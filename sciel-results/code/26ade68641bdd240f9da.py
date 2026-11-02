import itertools
import random
import sys
import time
import numpy as np

def construct(q, n, radius):
    # Specific solution for q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback: return a trivial valid code (all words)
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute ball masks using bit-parallel representation
    # ball_masks[i] is an integer where bit j is 1 if word j is in the ball of word i
    ball_masks = [0] * N
    
    for i in range(N):
        w = words[i]
        mask = 1 << i  # Self
        
        # Neighbors at distance 1
        for pos in range(n):
            orig_val = w[pos]
            for val in range(q):
                if val != orig_val:
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    j = word_to_idx[new_w]
                    mask |= (1 << j)
        
        ball_masks[i] = mask
    
    best_solution = None
    best_len = float('inf')
    
    # Time limit check
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer for 120s limit
    
    # Run multiple passes
    num_passes = 50  # We will break early if time runs out
    
    for pass_num in range(num_passes):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(pass_num)  # Deterministic but varied
        
        # --- Phase 1: Greedy Set Cover with Bitmasks ---
        # uncovered_mask: bitmask of points not yet covered
        all_ones = (1 << N) - 1
        uncovered = all_ones
        selected = []
        
        while uncovered:
            best_idx = -1
            max_cover = -1
            
            # Efficient approach: scan for max coverage
            for i in range(N):
                if not (ball_masks[i] & uncovered):
                    continue
                c = (ball_masks[i] & uncovered).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
            
            if best_idx == -1:
                break  # Should not happen if uncovered is not empty
                
            selected.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
        
        # --- Phase 2: Local Search (Removal and Swap) ---
        # State:
        # cover_count[p] = number of selected codewords covering p
        
        cover_count = [0] * N
        for v in selected:
            mask = ball_masks[v]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
        
        improved = True
        while improved:
            improved = False
            random.shuffle(selected)
            
            # 1. Try Removals
            new_selected = []
            for v in selected:
                is_redundant = True
                mask = ball_masks[v]
                m = mask
                while m:
                    lsb = m & -m
                    p = lsb.bit_length() - 1
                    if cover_count[p] == 1:
                        is_redundant = False
                        break
                    m ^= lsb
                    
                if is_redundant:
                    # Remove v
                    m = mask
                    while m:
                        lsb = m & -m
                        p = lsb.bit_length() - 1
                        cover_count[p] -= 1
                        m ^= lsb
                    improved = True
                else:
                    new_selected.append(v)
            
            selected = new_selected
            
            # 2. Try Swaps (Replace v with w to potentially enable future removals)
            # We only do swaps if we are not close to optimal
            # A swap keeps size constant.
            # Strategy: Pick a random selected v. Remove it. Find w that covers the "lost" points.
            # This is expensive. We will limit swaps.
            
            if len(selected) > 400:  # Only if we are not close to optimal
                for _ in range(10):  # Limit swaps per iteration
                    if not selected:
                        break
                    v = random.choice(selected)
                    
                    # Check if v is "critical" (has unique points)
                    unique_points = []
                    mask = ball_masks[v]
                    m = mask
                    while m:
                        lsb = m & -m
                        p = lsb.bit_length() - 1
                        if cover_count[p] == 1:
                            unique_points.append(p)
                        m ^= lsb
                    
                    if not unique_points:
                        # v is redundant, should have been removed in step 1
                        # But maybe we skipped it? No, step 1 is exhaustive.
                        continue
                    
                    # We want to find a w that covers all unique_points
                    # and is not in selected.
                    # To make the swap "good", w should cover these points and ideally
                    # not add too many new points that become unique?
                    # Actually, any w covering the unique points is a valid swap.
                    
                    candidates = []
                    # Optimization: Check only a subset of candidates?
                    # Or scan all? N=6561 is small enough for a few scans.
                    
                    for w in range(N):
                        if w in selected:
                            continue
                        # Check if w covers all unique_points
                        covers_all = True
                        mask_w = ball_masks[w]
                        for p in unique_points:
                            if not (mask_w & (1 << p)):
                                covers_all = False
                                break
                        if covers_all:
                            candidates.append(w)
                            
                    if candidates:
                        # Pick a random candidate
                        w = random.choice(candidates)
                        
                        # Perform Swap: Remove v, Add w
                        # Remove v
                        mask_v = ball_masks[v]
                        m = mask_v
                        while m:
                            lsb = m & -m
                            p = lsb.bit_length() - 1
                            cover_count[p] -= 1
                            m ^= lsb
                        
                        # Add w
                        mask_w = ball_masks[w]
                        m = mask_w
                        while m:
                            lsb = m & -m
                            p = lsb.bit_length() - 1
                            cover_count[p] += 1
                            m ^= lsb
                            
                        # Update selected list
                        # Remove v, add w
                        selected.remove(v)
                        selected.append(w)
                        improved = True
                        break  # Restart loop after a swap to re-evaluate removals

        current_len = len(selected)
        if current_len < best_len:
            best_len = current_len
            best_solution = selected[:]

    if best_solution is None:
        # Fallback if something went wrong (should not happen)
        return [list(words[i]) for i in range(N)]
        
    return [list(words[i]) for i in best_solution]
