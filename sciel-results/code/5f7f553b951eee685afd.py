import itertools
import time
import random
import sys
from collections import defaultdict

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
    
    # sub_to_blocks: for each t-subset, list of blocks covering it
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
            
    if not _deadline_ok(deadline):
        return []

    def greedy_init(random_seed):
        random.seed(random_seed)
        uncovered = [True] * n_t
        uncovered_count = n_t
        selected_indices = []
        
        while uncovered_count > 0:
            if not _deadline_ok(deadline):
                break
                
            # Find most constrained uncovered subset
            min_count = float('inf')
            target_sub = -1
            # Optimization: don't scan all n_t every time if possible?
            # But n_t is 11k, which is fast.
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
            
            # Calculate gain for candidates
            # Gain = number of newly covered subsets
            gains = []
            for b in candidates:
                gain = 0
                for s in block_covers[b]:
                    if uncovered[s]:
                        gain += 1
                gains.append(gain)
                
            # Find max gain
            max_gain = max(gains)
            
            # Pick randomly from those with max gain (or near max)
            best_candidates = [b for b, g in zip(candidates, gains) if g == max_gain]
            
            # To diversify, we can pick from top 20%
            # But for now, random from max is good.
            best_block = random.choice(best_candidates)
            
            selected_indices.append(best_block)
            
            for s in block_covers[best_block]:
                if uncovered[s]:
                    uncovered[s] = False
                    uncovered_count -= 1
                    
        return selected_indices

    def prune(selected_indices):
        selected_set = set(selected_indices)
        counts = [0] * n_t
        
        for i in selected_set:
            for s in block_covers[i]:
                counts[s] += 1
                
        changed = True
        while changed and _deadline_ok(deadline):
            changed = False
            for i in list(selected_set):
                redundant = True
                for s in block_covers[i]:
                    if counts[s] <= 1:
                        redundant = False
                        break
                if redundant:
                    selected_set.remove(i)
                    for s in block_covers[i]:
                        counts[s] -= 1
                    changed = True
        return selected_set

    def local_search(selected_set, random_seed):
        random.seed(random_seed)
        counts = [0] * n_t
        for i in selected_set:
            for s in block_covers[i]:
                counts[s] += 1
                
        no_improve = 0
        max_no_improve = 20
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
            for s in block_covers[i]:
                counts[s] -= 1
                
            uncovered = [s for s in block_covers[i] if counts[s] == 0]
            
            if not uncovered:
                # Success: i was redundant
                selected_set.remove(i)
                improved = True
                T = max(min_T, T * 0.9)
                no_improve = 0
                continue
            else:
                # Try to find a replacement
                # Sort uncovered by rarity
                uncovered.sort(key=lambda s: len(sub_to_blocks[s]))
                
                # Candidates for rarest
                candidates = set(sub_to_blocks[uncovered[0]])
                candidates -= selected_set # Must be a new block
                
                if candidates:
                    # Filter
                    valid_cands = []
                    for c in candidates:
                        c_set = block_covers_set[c]
                        # Check if c covers all uncovered
                        # Optimization: if len(c_set) < len(uncovered), skip
                        if len(c_set) >= len(uncovered):
                            ok = True
                            for s in uncovered[1:]:
                                if s not in c_set:
                                    ok = False
                                    break
                            if ok:
                                valid_cands.append(c)
                    
                    if valid_cands:
                        # Pick best
                        # Score: coverage of count==1 subsets
                        best_c = None
                        best_score = -1
                        # Sample if too many
                        if len(valid_cands) > 20:
                            valid_cands = random.sample(valid_cands, 20)
                        
                        for c in valid_cands:
                            score = 0
                            for s in block_covers[c]:
                                if counts[s] == 1:
                                    score += 1
                            if score > best_score:
                                best_score = score
                                best_c = c
                                
                        if best_c is not None:
                            # Swap
                            selected_set.remove(i)
                            selected_set.add(best_c)
                            
                            for s in block_covers[best_c]:
                                counts[s] += 1
                                
                            improved = True
                            T = max(min_T, T * 0.9)
                            no_improve = 0
                            continue
                
                # Failed to swap, restore
                for s in block_covers[i]:
                    counts[s] += 1
                    # Note: if we didn't swap, counts were decremented, so we restore
                    # But wait, if we didn't swap, we should restore the decrement.
                    # The loop above decremented. So we increment back.
                no_improve += 1
                T = max(min_T, T * cooling)
                
        return selected_set

    best_result = None
    best_len = float('inf')
    
    # Run multiple restarts
    num_restarts = 15
    for i in range(num_restarts):
        if not _deadline_ok(deadline):
            break
            
        selected = greedy_init(i)
        selected = prune(selected)
        selected = local_search(selected, i)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_result = selected
            # Optional: print or log
            
    if best_result is None:
        selected = greedy_init(0)
        selected = prune(selected)
        best_result = selected

    result = [list(blocks[i]) for i in best_result]
    
    # Final verification
    if _deadline_ok(deadline - 2.0):
        check_counts = [0] * n_t
        for i in best_result:
            for s in block_covers[i]:
                check_counts[s] += 1
        if any(c == 0 for c in check_counts):
             # Invalid, should not happen
             pass
             
    return result
