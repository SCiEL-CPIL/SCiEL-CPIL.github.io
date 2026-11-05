import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_sub_count = len(all_t_subsets)
    t_sub_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    k_block_count = len(all_k_blocks)
    
    # Efficient Precomputation of block_to_t_subs and t_sub_to_blocks
    block_to_t_subs = [[] for _ in range(k_block_count)]
    t_sub_to_blocks = [[] for _ in range(t_sub_count)]
    
    for bi, block in enumerate(all_k_blocks):
        # Generate all t-subsets within this block
        for t_sub in itertools.combinations(block, t):
            t_idx = t_sub_to_idx[t_sub]
            block_to_t_subs[bi].append(t_idx)
            t_sub_to_blocks[t_idx].append(bi)
            
    # Convert block_to_t_subs to sets for faster membership testing in Local Search
    block_t_sets = [set(lst) for lst in block_to_t_subs]
    
    # Greedy Covering
    cover_count = [0] * t_sub_count
    solution_blocks = []
    uncovered_count = t_sub_count
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
            
        # Find an uncovered t-subset
        target_t_idx = -1
        for i in range(t_sub_count):
            if cover_count[i] == 0:
                target_t_idx = i
                break
                
        if target_t_idx == -1:
            break
            
        candidates = t_sub_to_blocks[target_t_idx]
        
        best_block_idx = -1
        best_gain = -1
        
        for bi in candidates:
            gain = 0
            for t_idx in block_to_t_subs[bi]:
                if cover_count[t_idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = bi
                
        if best_block_idx == -1:
            break
            
        solution_blocks.append(best_block_idx)
        for t_idx in block_to_t_subs[best_block_idx]:
            if cover_count[t_idx] == 0:
                uncovered_count -= 1
            cover_count[t_idx] += 1
            
    # Deletion Pass
    for _ in range(10):
        if time.time() > deadline:
            break
        any_removed = False
        for i in range(len(solution_blocks) - 1, -1, -1):
            bi = solution_blocks[i]
            redundant = True
            for t_idx in block_to_t_subs[bi]:
                if cover_count[t_idx] == 1:
                    redundant = False
                    break
            if redundant:
                for t_idx in block_to_t_subs[bi]:
                    cover_count[t_idx] -= 1
                solution_blocks.pop(i)
                any_removed = True
        if not any_removed:
            break
            
    # Local Search: Swap-based improvement
    for ls_round in range(5):
        if time.time() > deadline:
            break
        
        random.shuffle(solution_blocks)
        
        improved = False
        for i in range(len(solution_blocks)):
            if time.time() > deadline:
                break
                
            bi = solution_blocks[i]
            critical_subs = []
            for t_idx in block_to_t_subs[bi]:
                if cover_count[t_idx] == 1:
                    critical_subs.append(t_idx)
            
            if not critical_subs:
                for t_idx in block_to_t_subs[bi]:
                    cover_count[t_idx] -= 1
                solution_blocks.pop(i)
                improved = True
                break
            
            if len(critical_subs) > 6:
                continue
                
            rarest = min(critical_subs, key=lambda x: len(t_sub_to_blocks[x]))
            
            potential_replacements = []
            crit_set = set(critical_subs)
            
            for b_idx in t_sub_to_blocks[rarest]:
                if b_idx == bi:
                    continue
                if crit_set.issubset(block_t_sets[b_idx]):
                    potential_replacements.append(b_idx)
            
            if not potential_replacements:
                continue
                
            best_rep = -1
            best_score = -1
            
            for b_idx in potential_replacements:
                score = 0
                for t_idx in block_to_t_subs[b_idx]:
                    if t_idx not in crit_set and cover_count[t_idx] == 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_rep = b_idx
            
            if best_rep == -1:
                continue
                
            for t_idx in block_to_t_subs[bi]:
                cover_count[t_idx] -= 1
            for t_idx in block_to_t_subs[best_rep]:
                cover_count[t_idx] += 1
            
            solution_blocks[i] = best_rep
            improved = True
            
            j = len(solution_blocks) - 1
            while j >= 0:
                bj = solution_blocks[j]
                redundant = True
                for t_idx in block_to_t_subs[bj]:
                    if cover_count[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for t_idx in block_to_t_subs[bj]:
                        cover_count[t_idx] -= 1
                    solution_blocks.pop(j)
                j -= 1
                
            if not improved:
                break
                
    # Final Deletion Pass
    for _ in range(5):
        if time.time() > deadline:
            break
        any_removed = False
        for i in range(len(solution_blocks) - 1, -1, -1):
            bi = solution_blocks[i]
            redundant = True
            for t_idx in block_to_t_subs[bi]:
                if cover_count[t_idx] == 1:
                    redundant = False
                    break
            if redundant:
                for t_idx in block_to_t_subs[bi]:
                    cover_count[t_idx] -= 1
                solution_blocks.pop(i)
                any_removed = True
        if not any_removed:
            break
            
    final_blocks = [list(all_k_blocks[bi]) for bi in solution_blocks]
    
    return final_blocks
