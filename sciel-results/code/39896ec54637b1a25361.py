import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precompute all t-subsets and map to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t = len(all_t_subs)
    
    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)
    
    # Map each k-subset to a bitset of t-subset indices it contains
    k_sub_bitsets = [0] * n_k
    # Map each t-subset to list of k-subset indices that contain it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    # Map each k-subset to list of t-subset indices it contains
    k_sub_to_t_idxs = [[] for _ in range(n_k)]
    
    for i, k_sub in enumerate(all_k_subs):
        bitset = 0
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idx = t_sub_to_idx[t_sub]
            bitset |= (1 << t_idx)
            t_idxs.append(t_idx)
            t_sub_to_k_idxs[t_idx].append(i)
        k_sub_bitsets[i] = bitset
        k_sub_to_t_idxs[i] = t_idxs

    def time_ok():
        return time.time() < deadline

    def greedy_cover(random_seed):
        random.seed(random_seed)
        uncovered = (1 << n_t) - 1
        blocks = []
        
        while uncovered and time_ok():
            # Find the uncovered t-subset with the fewest candidate blocks
            # This is a "most constrained" heuristic
            lowest_bit = uncovered & -uncovered
            t_idx = lowest_bit.bit_length() - 1
            
            # Check a few random uncovered t-subsets to find one with fewest candidates
            best_t_idx = t_idx
            best_candidate_count = len(t_sub_to_k_idxs[t_idx])
            
            # Sample a few random uncovered t-subsets
            sample_size = min(20, n_t)
            for _ in range(sample_size):
                # Pick a random bit from uncovered
                rand_pos = random.getrandbits(n_t.bit_length())
                mask = (1 << rand_pos) & uncovered
                if mask:
                    candidate_t_idx = mask.bit_length() - 1
                    cand_count = len(t_sub_to_k_idxs[candidate_t_idx])
                    if cand_count < best_candidate_count:
                        best_candidate_count = cand_count
                        best_t_idx = candidate_t_idx
                    if cand_count == 1:
                        break
            
            t_idx = best_t_idx
            candidates = t_sub_to_k_idxs[t_idx]
            
            if not candidates:
                break
            
            best_gain = -1
            best_candidates = []
            
            for b_idx in candidates:
                gain = (k_sub_bitsets[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [b_idx]
                elif gain == best_gain:
                    best_candidates.append(b_idx)
            
            if not best_candidates:
                break
                
            # Random tie-breaking with slight preference for diverse blocks
            chosen_b_idx = random.choice(best_candidates)
            blocks.append(chosen_b_idx)
            uncovered &= ~k_sub_bitsets[chosen_b_idx]
            
        return blocks

    def reduce_covering(blocks):
        if not blocks:
            return blocks
            
        block_set = set(blocks)
        cover_count = [0] * n_t
        
        for b_idx in blocks:
            for t_idx in k_sub_to_t_idxs[b_idx]:
                cover_count[t_idx] += 1
                
        changed = True
        while changed and time_ok():
            changed = False
            new_blocks = []
            for b_idx in blocks:
                is_essential = False
                for t_idx in k_sub_to_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    for t_idx in k_sub_to_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            blocks = new_blocks
            
        return blocks

    def local_search(blocks, max_time):
        if not blocks:
            return blocks
            
        blocks = blocks[:]
        block_set = set(blocks)
        
        cover_count = [0] * n_t
        for b_idx in blocks:
            for t_idx in k_sub_to_t_idxs[b_idx]:
                cover_count[t_idx] += 1
                
        search_deadline = time.time() + max_time
        improved = True
        while improved and time.time() < search_deadline:
            improved = False
            
            # Try 1-0 moves (remove a block)
            for i in range(len(blocks)):
                if time.time() >= search_deadline:
                    break
                b_idx = blocks[i]
                
                # Temporarily remove
                for t_idx in k_sub_to_t_idxs[b_idx]:
                    cover_count[t_idx] -= 1
                    
                valid = True
                for t_idx in k_sub_to_t_idxs[b_idx]:
                    if cover_count[t_idx] == 0:
                        valid = False
                        break
                        
                if valid:
                    blocks.pop(i)
                    block_set.remove(b_idx)
                    improved = True
                    break
                else:
                    for t_idx in k_sub_to_t_idxs[b_idx]:
                        cover_count[t_idx] += 1
                    
                    # Try 1-1 swap: replace b_idx with another block
                    uncovered_t_idxs = []
                    for t_idx in k_sub_to_t_idxs[b_idx]:
                        if cover_count[t_idx] == 0:
                            uncovered_t_idxs.append(t_idx)
                    
                    if uncovered_t_idxs:
                        # Pick the most constrained uncovered t-subset
                        first_t = uncovered_t_idxs[0]
                        candidates = t_sub_to_k_idxs[first_t]
                        
                        best_swap = -1
                        best_score = -1
                        for b_cand in candidates:
                            if b_cand in block_set:
                                continue
                            # Check if b_cand covers all uncovered_t_idxs
                            covers_all = True
                            for t_u in uncovered_t_idxs:
                                if not ((k_sub_bitsets[b_cand] >> t_u) & 1):
                                    covers_all = False
                                    break
                            if covers_all:
                                # Score: how many additional t-subsets does it cover?
                                extra = 0
                                for t_idx in k_sub_to_t_idxs[b_cand]:
                                    if cover_count[t_idx] == 0 and t_idx not in uncovered_t_idxs:
                                        extra += 1
                                if extra > best_score:
                                    best_score = extra
                                    best_swap = b_cand
                                if best_score >= 5:
                                    break
                            
                        if best_swap != -1:
                            block_set.remove(b_idx)
                            block_set.add(best_swap)
                            blocks[i] = best_swap
                            
                            for t_idx in k_sub_to_t_idxs[b_idx]:
                                cover_count[t_idx] -= 1
                            for t_idx in k_sub_to_t_idxs[best_swap]:
                                cover_count[t_idx] += 1
                            
                            improved = True
                            break
            
            # Try 2-1 moves: remove two blocks, add one
            if not improved and len(blocks) > 2 and time.time() < search_deadline:
                for i in range(len(blocks)):
                    if time.time() >= search_deadline:
                        break
                    for j in range(i + 1, len(blocks)):
                        if time.time() >= search_deadline:
                            break
                        b1, b2 = blocks[i], blocks[j]
                        
                        # Temporarily remove both
                        for t_idx in k_sub_to_t_idxs[b1]:
                            cover_count[t_idx] -= 1
                        for t_idx in k_sub_to_t_idxs[b2]:
                            cover_count[t_idx] -= 1
                        
                        # Find uncovered t-subsets
                        uncovered_t_idxs = []
                        for t_idx in k_sub_to_t_idxs[b1]:
                            if cover_count[t_idx] == 0:
                                uncovered_t_idxs.append(t_idx)
                        for t_idx in k_sub_to_t_idxs[b2]:
                            if cover_count[t_idx] == 0 and t_idx not in uncovered_t_idxs:
                                uncovered_t_idxs.append(t_idx)
                        
                        if not uncovered_t_idxs:
                            # Both can be removed
                            blocks.pop(j)
                            blocks.pop(i)
                            block_set.remove(b1)
                            block_set.remove(b2)
                            improved = True
                            break
                        else:
                            # Try to find a single block that covers all uncovered
                            first_t = uncovered_t_idxs[0]
                            candidates = t_sub_to_k_idxs[first_t]
                            
                            best_swap = -1
                            for b_cand in candidates:
                                if b_cand in block_set:
                                    continue
                                covers_all = True
                                for t_u in uncovered_t_idxs:
                                    if not ((k_sub_bitsets[b_cand] >> t_u) & 1):
                                        covers_all = False
                                        break
                                if covers_all:
                                    best_swap = b_cand
                                    break
                            
                            if best_swap != -1:
                                blocks.pop(j)
                                blocks.pop(i)
                                blocks.append(best_swap)
                                block_set.remove(b1)
                                block_set.remove(b2)
                                block_set.add(best_swap)
                                for t_idx in k_sub_to_t_idxs[best_swap]:
                                    cover_count[t_idx] += 1
                                improved = True
                                break
                            else:
                                # Restore
                                for t_idx in k_sub_to_t_idxs[b1]:
                                    cover_count[t_idx] += 1
                                for t_idx in k_sub_to_t_idxs[b2]:
                                    cover_count[t_idx] += 1
                    if improved:
                        break
            if improved:
                continue

        return blocks

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple greedy attempts with different strategies
    num_restarts = 15
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        blocks = greedy_cover(i)
        if not blocks:
            continue
        blocks = reduce_covering(blocks)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
            
    # Local Search on the best block with remaining time
    if best_blocks and time.time() < deadline - 15:
        remaining = deadline - time.time() - 15
        blocks = local_search(best_blocks, remaining)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
    
    if best_blocks:
        return [all_k_subs[i] for i in best_blocks]
    else:
        return []
