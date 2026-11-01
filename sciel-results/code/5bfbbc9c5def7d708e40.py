import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = list(range(v))
    t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute coverage
    block_cover_indices = []
    for block in k_blocks:
        subs = list(itertools.combinations(block, t))
        indices = [t_subset_to_idx[sub] for sub in subs]
        block_cover_indices.append(indices)
        
    tsub_to_blocks = [[] for _ in range(n_t_subsets)]
    for b_idx, inds in enumerate(block_cover_indices):
        for s_idx in inds:
            tsub_to_blocks[s_idx].append(b_idx)
            
    # Convert to sets for faster "is C in tsub_to_blocks[s]" checks?
    # tsub_to_blocks[s] is a list. Checking `C in list` is O(L).
    # L is roughly n_k_blocks * C(k,t) / n_t_subsets.
    # 50k * 35 / 3876 ~ 450.
    # So `C in list` is ~450 ops. Acceptable.
    # But for 2-for-1, we check many candidates.
    # Let's keep them as lists.

    k_choose_t = len(block_cover_indices[0])
    
    best_blocks = None
    best_len = float('inf')
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # 1. Greedy with Randomization
        gains = [k_choose_t] * n_k_blocks
        uncovered = [True] * n_t_subsets
        selected = []
        uncovered_count = n_t_subsets
        
        # Optimization: Use a heap? No, gains change dynamically.
        # Just argmax is fine for 50k items, but we want randomization.
        # We will pick from top 10.
        
        while uncovered_count > 0:
            if time.time() > deadline: break
            
            # Find top 10 gains
            # This is O(N log 10) if we use nlargest
            # Or just O(N) scan to find max, then randomize?
            # Let's use a simple approach: find max gain, then pick randomly from blocks with gain >= max_gain * 0.8
            max_gain = max(gains)
            if max_gain == 0: break
            
            # Collect candidates
            # To avoid O(N) collection every time, we can just pick a random block and check if it's "good enough"?
            # No, greedy needs to be strong.
            # Let's use `random.sample` on indices where gain is high?
            # Simpler: Pick a random index, if gain is 0, retry? No, biased.
            
            # Efficient Top-K:
            # Since N=50k, scanning is fast.
            # Let's just find the max, and then pick randomly from the top 5% of gains?
            # Or just pick from top 10.
            
            # For simplicity and speed, let's stick to standard greedy for the first few restarts,
            # then introduce randomness.
            # Actually, the local search is the main driver.
            # Let's use a "Soft Greedy":
            # Pick b with probability proportional to exp(gain[b] / tau)
            # But sampling from 50k items is slow.
            
            # Alternative:
            # 1. Find max_gain.
            # 2. Find all blocks with gain == max_gain? Too many.
            # 3. Find blocks with gain >= max_gain - 1?
            
            # Let's try:
            # Pick a random block. If it's among the top 100, accept?
            # No, let's just use standard greedy but with a random tie-breaker.
            # If there are multiple blocks with max gain, pick one randomly.
            
            # To implement random tie-breaking efficiently:
            # 1. Find max_gain.
            # 2. Iterate through blocks, count how many have max_gain?
            # 3. Pick one at random from those?
            
            # Even simpler:
            # Just use `random.choice` on a list of indices with max gain.
            # Building that list is O(N).
            
            max_gain_val = max(gains)
            candidates = [i for i, g in enumerate(gains) if g == max_gain_val]
            b = rng.choice(candidates)
            
            selected.append(b)
            
            for s in block_cover_indices[b]:
                if uncovered[s]:
                    uncovered[s] = False
                    uncovered_count -= 1
                    for b2 in tsub_to_blocks[s]:
                        gains[b2] -= 1
                        
        if uncovered_count > 0:
            continue
            
        # 2. Local Search
        # We work with a list of block indices
        current_selected = selected[:]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Compute cover counts and unique subsets
            cover_count = np.zeros(n_t_subsets, dtype=np.int32)
            for b in current_selected:
                cover_count[block_cover_indices[b]] += 1
            
            unique_tsubs = {b: [] for b in current_selected}
            for b in current_selected:
                for s in block_cover_indices[b]:
                    if cover_count[s] == 1:
                        unique_tsubs[b].append(s)
            
            # Step A: Redundancy Removal
            redundant = [b for b in current_selected if len(unique_tsubs[b]) == 0]
            if redundant:
                for b in redundant:
                    current_selected.remove(b)
                improved = True
                continue
            
            # Step B: 1-for-1 Swap
            # Sort by unique count
            sorted_blocks = sorted(current_selected, key=lambda b: len(unique_tsubs[b]))
            
            swap_made = False
            for b_out in sorted_blocks:
                u_list = unique_tsubs[b_out]
                if not u_list: continue
                
                first_s = u_list[0]
                # Candidates for b_in are blocks covering first_s
                for b_in in tsub_to_blocks[first_s]:
                    if b_in in current_selected:
                        continue
                    
                    # Check if b_in covers all s in u_list
                    covers_all = True
                    for s in u_list:
                        if b_in not in tsub_to_blocks[s]:
                            covers_all = False
                            break
                    if covers_all:
                        # Perform swap
                        current_selected.remove(b_out)
                        current_selected.append(b_in)
                        swap_made = True
                        improved = True
                        break
                if swap_made:
                    break
                    
            if swap_made:
                continue
                
            # Step C: 2-for-1 Swap
            # Consider top 20 blocks with smallest unique counts
            top_candidates = sorted_blocks[:20]
            
            found_21 = False
            for i in range(len(top_candidates)):
                for j in range(i+1, len(top_candidates)):
                    b1 = top_candidates[i]
                    b2 = top_candidates[j]
                    
                    u1 = unique_tsubs[b1]
                    u2 = unique_tsubs[b2]
                    
                    # Union of unique subsets
                    # Since they are disjoint (unique to b1 and b2), union is just concat?
                    # Wait, a subset cannot be unique to b1 AND unique to b2.
                    # So u1 and u2 are disjoint.
                    u_union = u1 + u2
                    
                    if not u_union:
                        continue
                        
                    first_s = u_union[0]
                    
                    for b_in in tsub_to_blocks[first_s]:
                        if b_in == b1 or b_in == b2:
                            continue
                        if b_in in current_selected:
                            continue
                            
                        covers_all = True
                        for s in u_union:
                            if b_in not in tsub_to_blocks[s]:
                                covers_all = False
                                break
                        if covers_all:
                            # Perform 2-for-1 swap
                            current_selected.remove(b1)
                            current_selected.remove(b2)
                            current_selected.append(b_in)
                            found_21 = True
                            improved = True
                            break
                    if found_21:
                        break
                if found_21:
                    break
                    
            if found_21:
                continue
                
            # If no improvement, break
            break
            
        # Update best
        if len(current_selected) < best_len:
            best_len = len(current_selected)
            best_blocks = [list(k_blocks[i]) for i in current_selected]
            
    return best_blocks
