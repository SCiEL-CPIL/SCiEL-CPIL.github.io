"""Combined B2[2] construction with vectorized local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
import numpy as np

def _greedy_build(candidates, counts_init=None):
    """Greedy: add candidates in given order if B2[2] constraint holds."""
    chosen = []
    counts = defaultdict(int) if counts_init is None else dict(counts_init)
    chosen_set = set()
    for x in candidates:
        if x in chosen_set:
            continue
        ok = True
        new_sums = []
        for y in chosen:
            s = x + y
            new_sums.append(s)
            if counts[s] + 1 > 2:
                ok = False
                break
        if ok:
            s2 = 2 * x
            new_sums.append(s2)
            if counts[s2] + 1 > 2:
                ok = False
        if ok:
            for s in new_sums:
                counts[s] += 1
            chosen.append(x)
            chosen_set.add(x)
    return chosen, counts

def _is_valid_b2(chosen):
    """Verify B2[2] property using vectorized NumPy."""
    if not chosen:
        return True
    arr = np.array(chosen, dtype=np.int32)
    n = len(arr)
    i_idx, j_idx = np.triu_indices(n)
    sums = arr[i_idx] + arr[j_idx]
    unique, counts_arr = np.unique(sums, return_counts=True)
    return bool(np.all(counts_arr <= 2))

def _compute_conflicts(chosen, counts):
    """Vectorized: for each candidate y not in chosen, count how many z in chosen
    have counts[y+z] >= 2 (i.e., adding y would create a 3rd representation).
    Returns dict y -> conflict_count."""
    chosen_arr = np.array(chosen, dtype=np.int32)
    chosen_set = set(chosen)
    # All possible sums from chosen: for each y in 1..1000, sums = y + chosen
    # We need: for each y, count of z in chosen where counts[y+z] >= 2
    # Vectorized approach: build a matrix of (y+z) for y=1..1000, z in chosen
    # But that's 1000*n which could be large. Instead, iterate y but use numpy for inner.
    
    # Better: precompute which sums are "full" (count >= 2)
    full_sums = set()
    for s, c in counts.items():
        if c >= 2:
            full_sums.add(s)
    
    # For each candidate y, conflict = |{z in chosen : y+z in full_sums}| + [2*y in full_sums]
    conflicts = {}
    for y in range(1, 1001):
        if y in chosen_set:
            continue
        conf = 0
        for z in chosen:
            if (y + z) in full_sums:
                conf += 1
        if (2 * y) in full_sums:
            conf += 1
        conflicts[y] = conf
    return conflicts

def _local_search(chosen, time_limit=10.0):
    """Optimized local search with vectorized conflict computation."""
    chosen = sorted(chosen)
    counts = defaultdict(int)
    n = len(chosen)
    for i in range(n):
        for j in range(i, n):
            counts[chosen[i] + chosen[j]] += 1
    
    start_time = time.time()
    improved = True
    rng_ls = random.Random(999)
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        remaining = time_limit - (time.time() - start_time)
        if remaining <= 0:
            break
        
        # Phase 1: Try pure insertions
        chosen_set = set(chosen)
        conflicts = _compute_conflicts(chosen, counts)
        
        # Sort candidates by conflict (lowest first), try those with conflict <= 2
        promising = [(c, y) for y, c in conflicts.items() if c <= 2]
        promising.sort()
        
        # Limit number to try based on remaining time
        max_inserts_to_try = min(len(promising), max(20, int(remaining * 30)))
        
        for conf, y in promising[:max_inserts_to_try]:
            if time.time() - start_time > time_limit:
                break
            # Verify insertion is valid (conflict count is necessary but verify exactly)
            ok = True
            for z in chosen:
                if counts[y + z] + 1 > 2:
                    ok = False
                    break
            if ok and counts[2 * y] + 1 > 2:
                ok = False
            if ok:
                # Do insert
                for z in chosen:
                    counts[y + z] += 1
                counts[2 * y] += 1
                chosen.append(y)
                chosen.sort()
                chosen_set.add(y)
                improved = True
        
        # Phase 2: Try swaps
        if (time.time() - start_time) < time_limit:
            n = len(chosen)
            chosen_set = set(chosen)
            
            # Recompute conflicts after potential insertions
            conflicts = _compute_conflicts(chosen, counts)
            promising_swaps = [(c, y) for y, c in conflicts.items() if c <= 3]
            promising_swaps.sort()
            
            max_swaps_to_try = min(len(promising_swaps), max(10, int((time_limit - (time.time() - start_time)) * 15)))
            
            # Try swapping each element with promising candidates
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                x = chosen[i]
                
                # Temporarily remove x
                for z in chosen:
                    if z != x:
                        counts[x + z] -= 1
                counts[2 * x] -= 1
                chosen_set.discard(x)
                
                swapped = False
                for conf, y in promising_swaps[:max_swaps_to_try]:
                    if y in chosen_set:
                        continue
                    # Check if y can be added to (chosen - {x})
                    ok = True
                    for z in chosen:
                        if z != x:
                            if counts[y + z] + 1 > 2:
                                ok = False
                                break
                    if ok and counts[2 * y] + 1 > 2:
                        ok = False
                    
                    if ok:
                        # Do the swap
                        for z in chosen:
                            if z != x:
                                counts[y + z] += 1
                        counts[2 * y] += 1
                        chosen[i] = y
                        chosen_set.add(y)
                        chosen.sort()
                        improved = True
                        swapped = True
                        break
                
                if not swapped:
                    # Restore x
                    for z in chosen:
                        if z != x:
                            counts[x + z] += 1
                    counts[2 * x] += 1
                    chosen_set.add(x)
                    chosen.sort()
    
    return sorted(chosen)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Strategy 1: Standard greedy (increasing order)
    candidates = list(range(1, 1001))
    chosen1, _ = _greedy_build(candidates)
    
    best = chosen1[:]
    best_size = len(best)
    
    # Strategy 2: Diverse greedy restarts
    rng = random.Random(42)
    
    # Large-first
    if time.time() < deadline:
        cand_large = list(range(1000, 0, -1))
        chosen_large, _ = _greedy_build(cand_large)
        if len(chosen_large) > best_size:
            best = chosen_large[:]
            best_size = len(best)
    
    # Mixed alternating
    if time.time() < deadline:
        small = list(range(1, 501))
        large = list(range(1000, 499, -1))
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
    
    # Prime-based orderings
    for p in [2, 3, 5, 7, 11]:
        if time.time() > deadline - 30.0:
            break
        cand_p = [x for x in range(1, 1001) if x % p != 0] + [x for x in range(1, 1001) if x % p == 0]
        chosen_p, _ = _greedy_build(cand_p)
        if len(chosen_p) > best_size:
            best = chosen_p[:]
            best_size = len(best)
    
    # Random restarts with various biases
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 40.0:  # Leave time for local search
            break
        
        cand = list(range(1, 1001))
        
        if restart % 6 == 0:
            # Pure random
            rng.shuffle(cand)
        elif restart % 6 == 1:
            # Bias towards smaller (power 0.5)
            biased = [(rng.random() ** 0.5, i) for i in range(1000)]
            biased.sort(key=lambda x: x[0])
            cand = [b[1] + 1 for b in biased]
        elif restart % 6 == 2:
            # Bias towards larger (power 0.3)
            biased = [(rng.random() ** 0.3, i) for i in range(1000)]
            biased.sort(key=lambda x: x[0])
            cand = [b[1] + 1 for b in biased]
        elif restart % 6 == 3:
            # Very small bias (power 0.8)
            biased = [(rng.random() ** 0.8, i) for i in range(1000)]
            biased.sort(key=lambda x: x[0])
            cand = [b[1] + 1 for b in biased]
        elif restart % 6 == 4:
            # Block shuffle (blocks of 20)
            for i in range(0, 1000, 20):
                block = cand[i:i+20]
                rng.shuffle(block)
                cand[i:i+20] = block
        else:
            # Two-phase: first 100 random, rest sorted
            first_100 = cand[:100]
            rng.shuffle(first_100)
            cand = first_100 + cand[100:]
        
        chosen_r, _ = _greedy_build(cand)
        if len(chosen_r) > best_size:
            best = chosen_r[:]
            best_size = len(best)
    
    # Local search on the best solution
    remaining = deadline - time.time() - 3.0
    if remaining > 1.0:
        ls_time = min(remaining, 80.0)
        best = _local_search(best, time_limit=ls_time)
        best_size = len(best)
    
    # Second local search pass with different random seed if time allows
    if time.time() < deadline - 5.0:
        remaining2 = deadline - time.time() - 3.0
        if remaining2 > 5.0:
            best = _local_search(best, time_limit=min(remaining2, 30.0))
            best_size = len(best)
    
    # Verify validity
    if not _is_valid_b2(best):
        best = chosen1[:]
    
    return best
# EVOLVE-BLOCK-END
