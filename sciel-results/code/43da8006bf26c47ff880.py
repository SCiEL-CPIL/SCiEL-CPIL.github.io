import itertools
import time
import random
import sys
from collections import defaultdict
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0
    
    # 1. Precomputation
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    blocks = []
    block_covers = []
    block_covers_set = []
    
    for block in itertools.combinations(range(v), k):
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        blocks.append(block)
        block_covers.append(covers)
        block_covers_set.append(frozenset(covers))
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    # Build NumPy matrix for fast gain computation
    # cover_matrix[b, s] = 1 if block b covers subset s
    cover_matrix = np.zeros((n_blocks, n_t), dtype=np.int32)
    for b_idx, covers in enumerate(block_covers):
        for s_idx in covers:
            cover_matrix[b_idx, s_idx] = 1
    
    # sub_to_blocks: for each t-subset, list of blocks covering it
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
            
    if not _deadline_ok(deadline):
        return []

    def greedy_init(random_seed):
        random.seed(random_seed)
        uncovered = np.ones(n_t, dtype=np.bool_)
        uncovered_count = n_t
        selected_indices = []
        
        while uncovered_count > 0:
            if not _deadline_ok(deadline):
                break
                
            # Find most constrained uncovered subset
            min_count = float('inf')
            target_sub = -1
            for s in range(n_t):
                if uncovered[s]:
                    c = len(sub_to_blocks[s])
                    if c < min_count:
                        min_count = c
                        target_sub = s
                        if c == 1:
                            break
            
            if target_sub == -1:
                break
                
            candidates = sub_to_blocks[target_sub]
            
            # Use NumPy for fast gain computation
            candidate_arr = np.array(candidates)
            gains = cover_matrix[candidate_arr].dot(uncovered.astype(np.int32))
            
            max_gain = gains.max()
            
            # Pick randomly from those with max gain
            best_mask = gains == max_gain
            best_candidates = candidate_arr[best_mask]
            
            best_block = int(random.choice(best_candidates))
            
            selected_indices.append(best_block)
            
            # Update uncovered
            new_covers = cover_matrix[best_block]
            newly_covered = new_covers & uncovered
            uncovered_count -= int(newly_covered.sum())
            uncovered[newly_covered] = False
                    
        return selected_indices

    def prune(selected_indices):
        selected_set = set(selected_indices)
        counts = np.zeros(n_t, dtype=np.int32)
        
        for i in selected_set:
            counts[block_covers[i]] += 1
            
        changed = True
        while changed and _deadline_ok(deadline):
            changed = False
            for i in list(selected_set):
                # Check if any subset covered by i has count <= 1
                if (counts[block_covers[i]] <= 1).any():
                    continue
                # i is redundant
                selected_set.remove(i)
                counts[block_covers[i]] -= 1
                changed = True
        return selected_set

    def local_search(selected_set, random_seed):
        random.seed(random_seed)
        counts = np.zeros(n_t, dtype=np.int32)
        for i in selected_set:
            counts[block_covers[i]] += 1
                
        no_improve = 0
        max_no_improve = 30
        T = 5.0
        cooling = 0.95
        min_T = 0.1
        
        while T > min_T and _deadline_ok(deadline) and no_improve < max_no_improve:
            if not selected_set:
                break
                
            improved = False
            # Pick random block
            i = random.choice(list(selected_set))
            
            # Try to remove i
            covers_i = block_covers[i]
            counts[covers_i] -= 1
            
            uncovered_mask = (counts == 0) & (cover_matrix[i] == 1)
            uncovered_list = np.where(uncovered_mask)[0].tolist()
            
            if not uncovered_list:
                # Success: i was redundant
                selected_set.remove(i)
                improved = True
                T = max(min_T, T * 0.9)
                no_improve = 0
                continue
            else:
                # Try to find a replacement
                # Sort uncovered by rarity
                uncovered_list.sort(key=lambda s: len(sub_to_blocks[s]))
                
                # Candidates for rarest
                candidates = set(sub_to_blocks[uncovered_list[0]])
                candidates -= selected_set  # Must be a new block
                
                if candidates:
                    # Filter candidates that cover all uncovered
                    valid_cands = []
                    for c in candidates:
                        c_set = block_covers_set[c]
                        if len(c_set) >= len(uncovered_list):
                            ok = True
                            for s in uncovered_list[1:]:
                                if s not in c_set:
                                    ok = False
                                    break
                            if ok:
                                valid_cands.append(c)
                    
                    if valid_cands:
                        # Pick best by score: how many count==1 subsets it covers
                        best_c = None
                        best_score = -1
                        # Sample if too many
                        if len(valid_cands) > 30:
                            valid_cands = random.sample(valid_cands, 30)
                        
                        for c in valid_cands:
                            score = int((counts[block_covers[c]] == 1).sum())
                            if score > best_score:
                                best_score = score
                                best_c = c
                                
                        if best_c is not None:
                            # Swap
                            selected_set.remove(i)
                            selected_set.add(best_c)
                            
                            counts[block_covers[best_c]] += 1
                            
                            improved = True
                            T = max(min_T, T * 0.9)
                            no_improve = 0
                            continue
                
                # Failed to swap, restore
                counts[covers_i] += 1
                no_improve += 1
                T = max(min_T, T * cooling)
                
        return selected_set

    best_result = None
    best_len = float('inf')
    
    # Run multiple restarts - increased count due to faster greedy
    num_restarts = 25
    for i in range(num_restarts):
        if not _deadline_ok(deadline):
            break
            
        selected = greedy_init(i)
        selected = prune(selected)
        selected = local_search(selected, i)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_result = selected
            
    if best_result is None:
        selected = greedy_init(0)
        selected = prune(selected)
        best_result = selected

    result = [list(blocks[i]) for i in best_result]
    
    # Final verification
    if _deadline_ok(deadline - 2.0):
        check_counts = np.zeros(n_t, dtype=np.int32)
        for i in best_result:
            check_counts[block_covers[i]] += 1
        if (check_counts == 0).any():
             pass
             
    return result
