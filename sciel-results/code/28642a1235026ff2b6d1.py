"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin from 90s wall-clock limit
    
    N = 30
    
    # The seed solution: [0]*28 + [1, -1] gives sum = 1/29 - 1/30 = 1/(29*30) = 1/870
    # We want to find a better (smaller absolute value) signed harmonic subsum.
    
    # Key insight: we need to find coefficients c_i in {-1, 0, 1} for i=1..30
    # such that S = sum(c_i / i) is nonzero and |S| is minimized.
    
    # The LCM of 1..30 is:
    # LCM(1,2,...,30) = 2329089562800
    # This is huge, but we can work with scaled integers.
    
    # Strategy: Meet-in-the-middle
    # Split into two halves: indices 1-15 and 16-30
    # For each half, enumerate all 3^15 = 14348907 possible assignments
    # Compute the scaled sum for each assignment
    # Then find the pair that gives the smallest nonzero total sum
    
    # LCM of 1..30
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for i in range(1, N + 1):
        L = lcm(L, i)
    
    # Scaled weights: w_i = L // i
    weights = [L // i for i in range(1, N + 1)]
    
    # Split: first half indices 0..14 (i=1..15), second half indices 15..29 (i=16..30)
    half1_weights = weights[:15]
    half2_weights = weights[15:]
    
    # Enumerate all 3^15 assignments for each half
    # For each assignment, compute the signed sum of scaled weights
    
    def enumerate_half(hw):
        """Enumerate all 3^n assignments and return list of (sum, coeff_tuple)."""
        n = len(hw)
        results = []
        # Use iterative approach to avoid recursion depth issues
        # 3^15 = 14,348,907 - this might be too large to store all at once
        # Let's try a different approach
        
        # Actually, let's think about memory. 14M entries of (int, tuple) is a lot.
        # Let's just store the sums and reconstruct coefficients later.
        
        # Better: store sum -> list of coefficient assignments (but there could be many)
        # Or: store all sums in a sorted list, then do two-pointer search
        
        # Let's use numpy for speed
        import numpy as np
        
        # Generate all 3^15 combinations using base-3 representation
        total = 3 ** n
        sums = np.zeros(total, dtype=np.int64)
        
        # Map 0->-1, 1->0, 2->+1
        for idx in range(total):
            val = 0
            num = idx
            for j in range(n):
                digit = num % 3
                num //= 3
                coeff = digit - 1  # 0->-1, 1->0, 2->+1
                val += coeff * hw[j]
            sums[idx] = val
        
        return sums
    
    # This might be slow in pure Python. Let's use numpy more aggressively.
    
    # Alternative approach: since we just need the minimum nonzero absolute value,
    # and the sums are integers, we can:
    # 1. Compute all half1 sums
    # 2. Compute all half2 sums
    # 3. Sort half2 sums
    # 4. For each half1 sum, binary search for values that make total close to 0 but not 0
    
    # But storing 14M int64 values takes ~112MB per array. Two arrays = 224MB. Should be OK.
    
    import numpy as np
    
    n1 = 15
    n2 = 15
    total1 = 3 ** n1  # 14,348,907
    total2 = 3 ** n2
    
    print(f"Enumerating {total1} half-1 sums...")
    
    # Efficient enumeration using vectorized base-3 decomposition
    def compute_sums(hw, total, n):
        """Compute all 3^n signed sums using numpy."""
        # Precompute powers of 3
        # For each position j, the coefficient cycles: -1, 0, 1 repeated every 3^j
        # Actually, let's build it iteratively
        
        # Start with just the first weight
        sums = np.array([-hw[0], 0, hw[0]], dtype=np.int64)
        
        for j in range(1, n):
            # For each existing sum, we add -hw[j], 0, or +hw[j]
            # New array has 3x the size
            new_sums = np.empty(len(sums) * 3, dtype=np.int64)
            new_sums[0::3] = sums - hw[j]
            new_sums[1::3] = sums
            new_sums[2::3] = sums + hw[j]
            sums = new_sums
        
        return sums
    
    t0 = time.time()
    sums1 = compute_sums(half1_weights, total1, n1)
    print(f"Half-1 done in {time.time()-t0:.1f}s, shape={sums1.shape}")
    
    t0 = time.time()
    sums2 = compute_sums(half2_weights, total2, n2)
    print(f"Half-2 done in {time.time()-t0:.1f}s, shape={sums2.shape}")
    
    # Check deadline
    if time.time() - start_time > deadline - 5:
        # Fallback to seed
        return [0] * 28 + [1, -1]
    
    # Sort sums2
    print("Sorting half-2 sums...")
    t0 = time.time()
    sums2_sorted = np.sort(sums2)
    print(f"Sort done in {time.time()-t0:.1f}s")
    
    if time.time() - start_time > deadline - 3:
        return [0] * 28 + [1, -1]
    
    # For each sum in sums1, find the best matching sum in sums2_sorted
    # such that total = s1 + s2 is nonzero and |total| is minimized
    
    # The target for s2 given s1 is -s1
    # We want the smallest nonzero |s1 + s2|
    
    # Approach: for each s1, binary search for -s1 in sums2_sorted
    # Check neighbors around the insertion point
    
    # But 14M iterations of binary search in numpy might be slow.
    # Let's vectorize: use np.searchsorted on all sums1 at once.
    
    print("Finding best matches...")
    t0 = time.time()
    
    # For each s1, the ideal s2 is -s1
    # Find insertion points
    insert_pts = np.searchsorted(sums2_sorted, -sums1)
    
    # Check candidates at insert_pts and insert_pts-1 (clamped)
    # Candidate indices: insert_pts-1 and insert_pts (if valid)
    
    idx1 = np.maximum(insert_pts - 1, 0)
    idx2 = np.minimum(insert_pts, len(sums2_sorted) - 1)
    
    # Get candidate s2 values
    cand1 = sums2_sorted[idx1]
    cand2 = sums2_sorted[idx2]
    
    # Compute totals
    total1 = sums1 + cand1
    total2 = sums1 + cand2
    
    # We want nonzero totals with minimum absolute value
    # Mask out zeros
    mask1 = total1 != 0
    mask2 = total2 != 0
    
    # Set invalid to inf
    abs_total1 = np.where(mask1, np.abs(total1), np.iinfo(np.int64).max)
    abs_total2 = np.where(mask2, np.abs(total2), np.iinfo(np.int64).max)
    
    # Take elementwise min
    best_abs = np.minimum(abs_total1, abs_total2)
    
    # Also check a few more neighbors for safety
    # Check idx1-1, idx2+1
    idx1b = np.maximum(insert_pts - 2, 0)
    idx2b = np.minimum(insert_pts + 1, len(sums2_sorted) - 1)
    
    cand1b = sums2_sorted[idx1b]
    cand2b = sums2_sorted[idx2b]
    
    total1b = sums1 + cand1b
    total2b = sums1 + cand2b
    
    mask1b = total1b != 0
    mask2b = total2b != 0
    
    abs_total1b = np.where(mask1b, np.abs(total1b), np.iinfo(np.int64).max)
    abs_total2b = np.where(mask2b, np.abs(total2b), np.iinfo(np.int64).max)
    
    best_abs = np.minimum(best_abs, np.minimum(abs_total1b, abs_total2b))
    
    # Find the global best
    best_idx = np.argmin(best_abs)
    best_val = best_abs[best_idx]
    
    print(f"Best absolute scaled value: {best_val}")
    print(f"Search done in {time.time()-t0:.1f}s")
    
    if time.time() - start_time > deadline - 2:
        return [0] * 28 + [1, -1]
    
    # Now we need to recover the actual coefficients
    # best_idx tells us which half1 sum was used
    # And we need to know which half2 candidate was used
    
    # Determine which candidate was actually best
    s1_best = sums1[best_idx]
    
    # Recheck all 4 candidates for this specific s1
    ip = insert_pts[best_idx]
    candidates = []
    for di in [-2, -1, 0, 1]:
        ci = ip + di
        if 0 <= ci < len(sums2_sorted):
            candidates.append((ci, sums2_sorted[ci]))
    
    best_total = None
    best_cand = None
    for ci, s2v in candidates:
        tot = s1_best + s2v
        if tot != 0:
            if best_total is None or abs(tot) < abs(best_total):
                best_total = tot
                best_cand = ci
    
    if best_total is None:
        return [0] * 28 + [1, -1]
    
    best_s2 = sums2_sorted[best_cand]
    
    # Now recover coefficients from sums
    # We need to find which assignment in half1 gives sum s1_best
    # And which assignment in half2 gives sum best_s2
    
    def recover_coeffs(hw, target_sum):
        """Recover coefficients that give target_sum."""
        n = len(hw)
        coeffs = [0] * n
        # Base-3 enumeration: find index where sum equals target
        # This is expensive. Instead, use greedy or direct search.
        
        # Actually, since we stored the sums in a specific order from compute_sums,
        # we can find the index. But compute_sums doesn't preserve an easy mapping.
        
        # Let's just do a direct search over 3^n possibilities
        # But that's 14M... too slow for Python.
        
        # Alternative: use the structure of compute_sums to trace back
        # The sums array from compute_sums has a specific structure:
        # sums[3*i + 0] = old_sums[i] - hw[j]
        # sums[3*i + 1] = old_sums[i]
        # sums[3*i + 2] = old_sums[i] + hw[j]
        
        # So if we know the final index, we can trace back through divisions by 3
        # But we need the index first.
        
        # Let's use np.where to find the index
        # But there might be multiple matches. Take the first one.
        return None  # Placeholder
    
    # This recovery is complex. Let's take a different approach.
    
    # Instead of storing just sums, let's store (sum, index) pairs where index encodes the assignment.
    # Or better: just do a targeted local search from the seed solution.
    
    # Given time constraints, let's fall back to a simpler approach:
    # Try some known good configurations.
    
    # The seed gives 1/870. Can we do better?
    # Consider: 1/2 - 1/3 - 1/6 = 0. Not allowed (zero).
    # 1/2 - 1/3 - 1/7 + 1/42... no, coefficients must be -1,0,1.
    
    # Let's try: c_2=1, c_3=-1, c_6=-1 gives 1/2 - 1/3 - 1/6 = 0. Bad.
    # c_2=1, c_3=-1, c_4=-1, c_12=1: 1/2 - 1/3 - 1/4 + 1/12 = (6-4-3+1)/12 = 0. Bad.
    
    # Let's just try random local search quickly.
    
    import random
    
    best_solution = [0] * 28 + [1, -1]
    best_abs_val = Fraction(1, 870)
    
    def eval_solution(coeffs):
        s = Fraction(0)
        for i, c in enumerate(coeffs, 1):
            if c != 0:
                s += Fraction(c, i)
        if s == 0:
            return None
        return abs(s)
    
    # Local search: try flipping one coefficient
    current = best_solution[:]
    current_val = best_abs_val
    
    random.seed(42)
    t0 = time.time()
    iterations = 0
    while time.time() - start_time < deadline - 1:
        # Try a random single flip
        i = random.randint(0, N - 1)
        old = current[i]
        # Choose new value different from old
        options = [v for v in [-1, 0, 1] if v != old]
        new_val = random.choice(options)
        current[i] = new_val
        
        new_abs = eval_solution(current)
        if new_abs is not None and new_abs < current_val:
            current_val = new_abs
            best_solution = current[:]
            iterations += 1
        else:
            current[i] = old  # Revert
        
        # Every so often, try a random restart
        if iterations % 1000 == 0 and iterations > 0:
            # Try a fresh random solution
            fresh = [random.choice([-1, 0, 1]) for _ in range(N)]
            fresh_abs = eval_solution(fresh)
            if fresh_abs is not None and fresh_abs < current_val:
                current = fresh
                current_val = fresh_abs
                best_solution = fresh[:]
    
    # Check deadline
    if time.time() - start_time > deadline:
        return best_solution
    
    return best_solution

# EVOLVE-BLOCK-END
