import itertools
import time
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    """
    Solve the covering design problem C(v, k, t) using a structured probabilistic
    construction inspired by the Johnson scheme, followed by targeted local search.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    def time_ok():
        return time.time() < deadline - 5.0

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all blocks and their covered t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_covered_t = [None] * num_blocks
    block_covered_t_set = [None] * num_blocks
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for i, b in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            t_to_blocks[idx].append(i)
        block_covered_t[i] = covered_indices
        block_covered_t_set[i] = set(covered_indices)
    
    # Remove redundant blocks
    def try_remove_blocks(blocks):
        if not blocks:
            return blocks
        
        cover_count = [0] * num_t_subsets
        for b_idx in blocks:
            for t_idx in block_covered_t[b_idx]:
                cover_count[t_idx] += 1
        
        changed = True
        while changed and time_ok():
            changed = False
            for i in range(len(blocks)):
                if not time_ok():
                    return blocks
                
                b_idx = blocks[i]
                is_redundant = True
                for t_idx in block_covered_t[b_idx]:
                    if cover_count[t_idx] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        
        return blocks
    
    # 1-exchange local search with improved candidate selection
    def one_exchange_search(blocks, max_iters=1000):
        if not blocks:
            return blocks
        
        current_blocks = list(blocks)
        
        cover_count = [0] * num_t_subsets
        for b_idx in current_blocks:
            for t_idx in block_covered_t[b_idx]:
                cover_count[t_idx] += 1
        
        selected_set = set(current_blocks)
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iters and time_ok():
            improved = False
            iterations += 1
            
            # First pass: remove redundant blocks
            removed_any = False
            for i in range(len(current_blocks)):
                if not time_ok():
                    break
                b_idx = current_blocks[i]
                is_redundant = True
                for t_idx in block_covered_t[b_idx]:
                    if cover_count[t_idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    selected_set.discard(b_idx)
                    improved = True
                    removed_any = True
                    break
            
            if removed_any:
                continue
            
            # Try 1-exchanges with smarter candidate selection
            for i in range(len(current_blocks)):
                if not time_ok():
                    break
                
                b_idx = current_blocks[i]
                
                # Find essential t-subsets (only covered by this block)
                essential_t = [t_idx for t_idx in block_covered_t[b_idx] if cover_count[t_idx] == 1]
                
                if not essential_t:
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    selected_set.discard(b_idx)
                    improved = True
                    break
                
                # Build candidate set from blocks that cover at least one essential t-subset
                candidate_set = set()
                for t_idx in essential_t:
                    for cand in t_to_blocks[t_idx]:
                        if cand not in selected_set:
                            candidate_set.add(cand)
                
                # Limit candidates for efficiency
                if len(candidate_set) > 5000:
                    candidate_list = list(candidate_set)
                    random.shuffle(candidate_list)
                    candidate_list = candidate_list[:5000]
                else:
                    candidate_list = list(candidate_set)
                
                best_swap_block = -1
                best_swap_metric = -1
                
                for cand_idx in candidate_list:
                    # Check if candidate covers all essential t-subsets
                    if not all(t_idx in block_covered_t_set[cand_idx] for t_idx in essential_t):
                        continue
                    
                    # Calculate metric
                    new_cover = 0
                    overlap_penalty = 0
                    for t_idx in block_covered_t[cand_idx]:
                        if cover_count[t_idx] == 0:
                            new_cover += 1
                        elif cover_count[t_idx] == 1:
                            overlap_penalty += 0.5
                        else:
                            overlap_penalty += 1.0
                    
                    metric = new_cover * 2.0 - overlap_penalty
                    
                    if metric > best_swap_metric:
                        best_swap_metric = metric
                        best_swap_block = cand_idx
                
                if best_swap_block != -1 and best_swap_metric > 0:
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    for t_idx in block_covered_t[best_swap_block]:
                        cover_count[t_idx] += 1
                    selected_set.discard(b_idx)
                    selected_set.add(best_swap_block)
                    current_blocks[i] = best_swap_block
                    improved = True
                    break
        
        return current_blocks
    
    # 2-for-2 exchange search
    def two_exchange_search(blocks, max_iters=200):
        if len(blocks) < 2:
            return blocks
        
        current_blocks = list(blocks)
        
        cover_count = [0] * num_t_subsets
        for b_idx in current_blocks:
            for t_idx in block_covered_t[b_idx]:
                cover_count[t_idx] += 1
        
        selected_set = set(current_blocks)
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iters and time_ok():
            improved = False
            iterations += 1
            
            # Remove redundant
            for i in range(len(current_blocks)):
                if not time_ok():
                    break
                b_idx = current_blocks[i]
                is_redundant = True
                for t_idx in block_covered_t[b_idx]:
                    if cover_count[t_idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    selected_set.discard(b_idx)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 2-for-2 exchanges
            n = len(current_blocks)
            for i in range(n):
                if not time_ok():
                    break
                for j in range(i + 1, n):
                    if not time_ok():
                        break
                    
                    b1 = current_blocks[i]
                    b2 = current_blocks[j]
                    
                    # Find essential t-subsets for each
                    essential1 = [t_idx for t_idx in block_covered_t[b1] if cover_count[t_idx] == 1]
                    essential2 = [t_idx for t_idx in block_covered_t[b2] if cover_count[t_idx] == 1]
                    
                    if not essential1 and not essential2:
                        continue
                    
                    # Build candidate set
                    candidate_set = set()
                    for t_idx in essential1 + essential2:
                        for cand in t_to_blocks[t_idx]:
                            if cand not in selected_set:
                                candidate_set.add(cand)
                    
                    if len(candidate_set) > 3000:
                        candidate_list = list(candidate_set)
                        random.shuffle(candidate_list)
                        candidate_list = candidate_list[:3000]
                    else:
                        candidate_list = list(candidate_set)
                    
                    best_pair = None
                    best_metric = -1
                    
                    # Try pairs of candidates
                    # For efficiency, only try a limited number of pairs
                    num_pairs = min(500, len(candidate_list) * (len(candidate_list) - 1) // 2)
                    if len(candidate_list) < 2:
                        continue
                    
                    # Sample pairs
                    tried_pairs = set()
                    attempts = 0
                    max_attempts = min(num_pairs, 200)
                    
                    while attempts < max_attempts:
                        attempts += 1
                        c1, c2 = random.sample(candidate_list, 2)
                        if (c1, c2) in tried_pairs or (c2, c1) in tried_pairs:
                            continue
                        tried_pairs.add((c1, c2))
                        
                        # Check if c1 and c2 together cover all essential t-subsets
                        all_essential = essential1 + essential2
                        covered = block_covered_t_set[c1] | block_covered_t_set[c2]
                        if not all(t_idx in covered for t_idx in all_essential):
                            continue
                        
                        # Calculate metric
                        new_cover = 0
                        overlap_penalty = 0
                        for t_idx in block_covered_t[c1] + block_covered_t[c2]:
                            if cover_count[t_idx] == 0:
                                new_cover += 1
                            elif cover_count[t_idx] == 1:
                                overlap_penalty += 0.5
                            else:
                                overlap_penalty += 1.0
                        
                        metric = new_cover * 2.0 - overlap_penalty
                        
                        if metric > best_metric:
                            best_metric = metric
                            best_pair = (c1, c2)
                    
                    if best_pair is not None and best_metric > 0:
                        c1, c2 = best_pair
                        for t_idx in block_covered_t[b1]:
                            cover_count[t_idx] -= 1
                        for t_idx in block_covered_t[b2]:
                            cover_count[t_idx] -= 1
                        for t_idx in block_covered_t[c1]:
                            cover_count[t_idx] += 1
                        for t_idx in block_covered_t[c2]:
                            cover_count[t_idx] += 1
                        selected_set.discard(b1)
                        selected_set.discard(b2)
                        selected_set.add(c1)
                        selected_set.add(c2)
                        current_blocks[i] = c1
                        current_blocks[j] = c2
                        improved = True
                        break
                
                if improved:
                    break
        
        return current_blocks
    
    # Greedy construction
    def greedy_construct(seed):
        random.seed(seed)
        
        uncovered_set = set(range(num_t_subsets))
        selected = []
        
        while uncovered_set and time_ok():
            target_t_idx = random.choice(list(uncovered_set))
            candidates = t_to_blocks[target_t_idx]
            
            best_block = -1
            best_gain = -1
            best_blocks_with_gain = []
            
            for b_idx in candidates:
                gain = sum(1 for t_idx in block_covered_t[b_idx] if t_idx in uncovered_set)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_blocks_with_gain = [b_idx]
                elif gain == best_gain:
                    best_blocks_with_gain.append(b_idx)
            
            if best_block == -1:
                break
            
            best_block = random.choice(best_blocks_with_gain)
            selected.append(best_block)
            
            for t_idx in block_covered_t[best_block]:
                uncovered_set.discard(t_idx)
        
        return selected
    
    # Main search loop
    best_blocks = None
    best_len = float('inf')
    
    seeds = [42, 123, 999, 0, 1, 2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 100, 200, 300, 400, 500]
    
    for seed in seeds:
        if not time_ok():
            break
        
        greedy_res = greedy_construct(seed)
        if not greedy_res:
            continue
        
        processed_res = try_remove_blocks(greedy_res)
        ls_res = one_exchange_search(processed_res)
        ls_res = two_exchange_search(ls_res)
        
        if len(ls_res) < best_len:
            best_len = len(ls_res)
            best_blocks = ls_res
        
        if best_len <= 69:
            break
    
    if time_ok():
        for _ in range(20):
            if not time_ok():
                break
            
            seed = random.randint(0, 1000000)
            greedy_res = greedy_construct(seed)
            
            if not greedy_res:
                continue
            
            processed_res = try_remove_blocks(greedy_res)
            ls_res = one_exchange_search(processed_res)
            ls_res = two_exchange_search(ls_res)
            
            if len(ls_res) < best_len:
                best_len = len(ls_res)
                best_blocks = ls_res
                
            if best_len <= 69:
                break
    
    if best_blocks is None:
        greedy_res = greedy_construct(42)
        if greedy_res:
            best_blocks = try_remove_blocks(greedy_res)
        else:
            return []
    
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    
    seen = set()
    unique_blocks = []
    for b in final_blocks:
        b.sort()
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    
    return unique_blocks
