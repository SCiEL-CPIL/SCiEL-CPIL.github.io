"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
import numpy as np

def _greedy_build(candidates, counts_init=None):
    """Greedy: add candidates in given order if B2[2] constraint holds."""
    chosen = []
    counts = defaultdict(int) if counts_init is None else dict(counts_init)
    for x in candidates:
        # Check all new sums
        ok = True
        for y in chosen:
            s = x + y
            if counts[s] + 1 > 2:
                ok = False
                break
        if ok:
            s2 = 2 * x
            if counts[s2] + 1 > 2:
                ok = False
        if ok:
            for y in chosen:
                counts[x + y] += 1
            counts[2 * x] += 1
            chosen.append(x)
    return chosen, counts

def _is_valid_b2(chosen):
    """Verify B2[2] property."""
    if not chosen:
        return True
    arr = np.array(chosen)
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(len(arr))
    sums = arr[i_idx] + arr[j_idx]
    # Count occurrences
    unique, counts_arr = np.unique(sums, return_counts=True)
    return np.all(counts_arr <= 2)

def _local_search(chosen, time_limit=10.0):
    """Try swaps to improve set size - optimized version."""
    chosen = sorted(chosen)
    n = len(chosen)
    
    # Precompute counts
    counts = defaultdict(int)
    for i in range(n):
        for j in range(i, n):
            counts[chosen[i] + chosen[j]] += 1
    
    chosen_set = set(chosen)
    start_time = time.time()
    
    # Generate candidate y values: focus on gaps and boundary regions
    # Try values that are far from existing elements to reduce sum collisions
    candidate_pool = list(range(1, 1001))
    rng_ls = random.Random(12345)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Shuffle candidate order for diversity
        candidate_pool_shuffled = candidate_pool[:]
        rng_ls.shuffle(candidate_pool_shuffled)
        
        for i in range(len(chosen)):
            if time.time() - start_time > time_limit:
                break
            x = chosen[i]
            
            # Temporarily remove x
            for j in range(len(chosen)):
                if j != i:
                    s = x + chosen[j]
                    counts[s] -= 1
            counts[2 * x] -= 1
            
            found_swap = False
            # Try candidates in shuffled order
            for y in candidate_pool_shuffled:
                if y in chosen_set:
                    continue
                
                # Check if adding y is valid
                ok = True
                for j in range(len(chosen)):
                    s = y + chosen[j]
                    if counts[s] + 1 > 2:
                        ok = False
                        break
                if ok:
                    s2 = 2 * y
                    if counts[s2] + 1 > 2:
                        ok = False
                
                if ok:
                    # Do the swap
                    for j in range(len(chosen)):
                        s = y + chosen[j]
                        counts[s] += 1
                    counts[2 * y] += 1
                    chosen[i] = y
                    chosen_set.discard(x)
                    chosen_set.add(y)
                    improved = True
                    found_swap = True
                    break
            
            if found_swap:
                break
            else:
                # Restore x
                for j in range(len(chosen)):
                    if j != i:
                        s = x + chosen[j]
                        counts[s] += 1
                counts[2 * x] += 1
    
    return sorted(chosen)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s
    
    # Strategy 1: Standard greedy (increasing order)
    candidates = list(range(1, 1001))
    chosen1, _ = _greedy_build(candidates)
    
    best = chosen1[:]
    best_size = len(best)
    
    # Strategy 2: Try structured and random orderings
    rng = random.Random(42)
    
    # Try some structured approaches
    # Approach A: Small numbers first (already done in Strategy 1)
    # Approach B: Large numbers first
    if time.time() < deadline:
        cand_large = list(range(1000, 0, -1))
        chosen_large, _ = _greedy_build(cand_large)
        if len(chosen_large) > best_size:
            best = chosen_large[:]
            best_size = len(best)
    
    # Approach C: Mixed - alternate small and large
    if time.time() < deadline:
        small = list(range(1, 501))
        large = list(range(500, 0, -1))
        cand_mixed = []
        si, li = 0, 0
        while si < len(small) or li < len(large):
            if si < len(small):
                cand_mixed.append(small[si])
                si += 1
            if li < len(large):
                cand_mixed.append(large[li])
                li += 1
        chosen_mixed, _ = _greedy_build(cand_mixed)
        if len(chosen_mixed) > best_size:
            best = chosen_mixed[:]
            best_size = len(best)
    
    # Strategy 3: Try random orderings with bias
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create biased candidate list: mix of structured and random
        if restart % 3 == 0:
            # Pure random
            cand = list(range(1, 1001))
            rng.shuffle(cand)
        elif restart % 3 == 1:
            # Bias towards smaller numbers
            cand = list(range(1, 1001))
            # Weighted shuffle: smaller numbers more likely to come first
            weights = [1.0 / (i + 1) for i in range(1000)]
            # Simple approach: sort by random key with bias
            biased = [(rng.random() ** 0.5, i) for i in range(1000)]
            biased.sort(key=lambda x: x[0])
            cand = [b[1] + 1 for b in biased]
        else:
            # Bias towards larger numbers
            cand = list(range(1, 1001))
            biased = [(rng.random() ** 0.3, i) for i in range(1000)]
            biased.sort(key=lambda x: x[0])
            cand = [b[1] + 1 for b in biased]
        
        chosen_r, _ = _greedy_build(cand)
        if len(chosen_r) > best_size:
            best = chosen_r[:]
            best_size = len(best)
    
    # Local search on the best solution
    if time.time() < deadline - 3.0:
        ls_time = min(15.0, deadline - time.time() - 2.0)
        if ls_time > 1.0:
            best = _local_search(best, time_limit=ls_time)
            best_size = len(best)
    
    # Verify validity
    if not _is_valid_b2(best):
        # Fallback to standard greedy result
        best = chosen1[:]
    
    return best
# EVOLVE-BLOCK-END
