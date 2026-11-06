# EVOLVE-BLOCK-START
import time
import random
import math
from fractions import Fraction

DENOMINATORS = list(range(2, 49))
N = len(DENOMINATORS)
FRAC_DENOMS = [Fraction(1, d) for d in DENOMINATORS]
ONE = Fraction(1, 1)

def _best_subset_for_target(target: Fraction):
    """
    Find a subset of indices whose reciprocal sum is strictly below target,
    minimizing the deficit (target - sum).
    Uses a meet-in-the-middle style enumeration for the smaller half.
    """
    # Split denominators into two halves
    mid = N // 2
    left = list(range(mid))
    right = list(range(mid, N))
    
    # Enumerate all subsets of the left half
    left_sums = {}  # sum -> list of indices
    for mask in range(1 << len(left)):
        s = Fraction(0, 1)
        idxs = []
        for i in range(len(left)):
            if mask & (1 << i):
                s += FRAC_DENOMS[left[i]]
                idxs.append(left[i])
        if s < target:
            # Keep the best (smallest sum for a given... actually we want to keep all for combining)
            # But storing all is too much memory if we store lists. 
            # Instead, for each sum, we just need to know it exists. 
            # However, to reconstruct, we need indices.
            # Let's store: sum -> (indices_tuple)
            # To save memory, we can prune: if we have two subsets with the same sum, keep the one with fewer elements? 
            # No, for combining, any is fine. Let's just keep one representative.
            # Actually, we want to combine left_sum + right_sum < target.
            # So for a fixed right_sum, we want the largest left_sum < target - right_sum.
            # So we can store for each possible left_sum, we don't need to store ALL of them if we sort.
            # Let's collect all valid left sums.
            left_sums[s] = tuple(idxs)
    
    # Enumerate all subsets of the right half
    right_sums = []
    for mask in range(1 << len(right)):
        s = Fraction(0, 1)
        idxs = []
        for i in range(len(right)):
            if mask & (1 << i):
                s += FRAC_DENOMS[right[i]]
                idxs.append(right[i])
        right_sums.append((s, tuple(idxs)))
    
    # Sort left sums by value to allow binary search or two-pointer
    # We want to find pairs (L, R) such that L + R < target, minimizing (target - (L+R)).
    # Equivalent to maximizing L + R subject to L + R < target.
    
    # Extract sorted keys from left_sums
    left_keys = sorted(left_sums.keys())
    
    best_deficit = None
    best_indices = None
    
    # For each right sum, find the best left sum
    for r_sum, r_idxs in right_sums:
        if r_sum >= target:
            continue
        # We need L < target - r_sum
        limit = target - r_sum
        # Find largest L in left_keys such that L < limit
        # Binary search for limit in left_keys
        # left_keys is sorted ascending.
        lo, hi = 0, len(left_keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if left_keys[mid] < limit:
                lo = mid + 1
            else:
                hi = mid
        # lo is the index of the first element >= limit.
        # So lo-1 is the largest index with value < limit.
        if lo > 0:
            best_l = left_keys[lo - 1]
            l_idxs = left_sums[best_l]
            total_sum = best_l + r_sum
            deficit = target - total_sum
            
            if best_deficit is None or deficit < best_deficit:
                best_deficit = deficit
                best_indices = list(l_idxs) + list(r_idxs)
    
    # Also check if empty set is better? No, we want non-empty.
    # What if no pair found? (e.g. all single elements exceed target? Unlikely for target=1)
    if best_indices is None:
        # Fallback: find single element closest to target
        for i in range(N):
            if FRAC_DENOMS[i] < target:
                d = target - FRAC_DENOMS[i]
                if best_deficit is None or d < best_deficit:
                    best_deficit = d
                    best_indices = [i]
    
    if best_indices is None:
        return [0], target - FRAC_DENOMS[0]
        
    return best_indices, best_deficit

def _local_search_polish(idx, deficit, deadline):
    """
    Fast local search to polish a solution.
    Tries 1-swap and 1-add moves.
    """
    best_idx = list(idx)
    best_deficit = deficit
    best_sum = ONE - best_deficit
    in_set = set(best_idx)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        
        # Try 1-swap: remove i, add j
        for i in best_idx[:]:
            if time.time() >= deadline:
                break
            f_out = FRAC_DENOMS[i]
            sum_without = best_sum - f_out
            for j in range(N):
                if j in in_set:
                    continue
                f_in = FRAC_DENOMS[j]
                new_sum = sum_without + f_in
                if new_sum >= ONE:
                    continue
                new_def = ONE - new_sum
                if new_def < best_deficit:
                    best_idx.remove(i)
                    best_idx.append(j)
                    in_set.remove(i)
                    in_set.add(j)
                    best_sum = new_sum
                    best_deficit = new_def
                    changed = True
                    break
            if changed:
                break
        
        # Try 1-add: add j if it fits
        if not changed:
            for j in range(N):
                if j in in_set:
                    continue
                f_in = FRAC_DENOMS[j]
                new_sum = best_sum + f_in
                if new_sum >= ONE:
                    continue
                new_def = ONE - new_sum
                if new_def < best_deficit:
                    best_idx.append(j)
                    in_set.add(j)
                    best_sum = new_sum
                    best_deficit = new_def
                    changed = True
                    break
    
    return best_idx, best_deficit

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    best_idx = [0]
    best_deficit = ONE - FRAC_DENOMS[0]
    
    # Phase 1: Exhaustive search via meet-in-the-middle for target 1
    # This is exact for the given N=47 if we can afford the memory/time.
    # 2^23 ~ 8 million subsets. This might be too slow in Python.
    # Let's estimate: 8 million iterations, each doing Fraction addition.
    # Fraction addition is O(1) for small numbers, but memory is the issue.
    # Storing 8 million Fractions might take ~1GB? Maybe too much.
    # Let's try a smaller split or a heuristic.
    
    # Alternative: Use the meet-in-the-middle but with a limit on subset size?
    # Or just use a strong heuristic + local search.
    
    # Given the 60s limit, a full 2^23 enumeration in Python is risky.
    # Let's try a hybrid: 
    # 1. Generate many random seeds.
    # 2. Use a more effective local search (including 2-swaps).
    # 3. Try to find the specific known good solutions.
    
    # Known good: 1/2 + 1/3 + 1/7 + 1/43 = 1805/1806. Deficit 1/1806 ~ 0.000553
    # Let's see if we can beat this.
    # 1/2 + 1/3 + 1/8 + 1/24 + 1/48 = 0.5+0.333+0.125+0.0416+0.0208 = 0.999... 
    # Sum = 12/24 + 8/24 + 3/24 + 1/24 + 0.5/24 = 24.5/24 > 1. No.
    # 1/2 + 1/3 + 1/9 + 1/18 + 1/36 + 1/72? No 72 not in list.
    
    # Let's run a few random restarts with strong local search.
    
    rng = random.Random(42)
    
    # Seed 1: Greedy
    selected, total = [], Fraction(0,1)
    for i in range(N):
        if total + FRAC_DENOMS[i] < ONE:
            selected.append(i)
            total += FRAC_DENOMS[i]
    idx, deficit = _local_search_polish(selected, ONE - total, deadline)
    if deficit < best_deficit:
        best_idx, best_deficit = idx, deficit
    
    # Seed 2: Known good
    s1 = [0, 1, 5, 41] # 1/2, 1/3, 1/7, 1/43
    sum1 = Fraction(1,2) + Fraction(1,3) + Fraction(1,7) + Fraction(1,43)
    if sum1 < ONE:
        idx, deficit = _local_search_polish(s1, ONE - sum1, deadline)
        if deficit < best_deficit:
            best_idx, best_deficit = idx, deficit
    
    # Random restarts
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        # Generate a random subset
        size = rng.randint(3, 8)
        candidates = list(range(N))
        rng.shuffle(candidates)
        selected = []
        total = Fraction(0,1)
        for i in candidates:
            if len(selected) >= size:
                break
            if total + FRAC_DENOMS[i] < ONE:
                selected.append(i)
                total += FRAC_DENOMS[i]
        
        if not selected:
            continue
            
        idx, deficit = _local_search_polish(selected, ONE - total, deadline)
        
        # Add a 2-swap local search pass if time permits
        if time.time() < deadline - 2:
            in_set = set(idx)
            cur_sum = ONE - deficit
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for a in range(len(idx)):
                    if time.time() >= deadline - 2: break
                    for b in range(a+1, len(idx)):
                        if time.time() >= deadline - 2: break
                        i, j = idx[a], idx[b]
                        sum_without = cur_sum - FRAC_DENOMS[i] - FRAC_DENOMS[j]
                        for x in range(N):
                            if x in in_set: continue
                            s1 = sum_without + FRAC_DENOMS[x]
                            if s1 >= ONE: continue
                            for y in range(N):
                                if y in in_set or y == x: continue
                                s2 = s1 + FRAC_DENOMS[y]
                                if s2 >= ONE: continue
                                new_def = ONE - s2
                                if new_def < deficit:
                                    # Apply move
                                    idx.remove(i)
                                    idx.remove(j)
                                    idx.append(x)
                                    idx.append(y)
                                    in_set.remove(i)
                                    in_set.remove(j)
                                    in_set.add(x)
                                    in_set.add(y)
                                    cur_sum = s2
                                    deficit = new_def
                                    improved = True
                                    break
                            if improved: break
                        if improved: break
                    if improved: break
                
                if deficit < best_deficit:
                    best_idx = list(idx)
                    best_deficit = deficit
        
        if deficit < best_deficit:
            best_idx = list(idx)
            best_deficit = deficit
    
    return best_idx
# EVOLVE-BLOCK-END
