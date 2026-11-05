"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s hard limit

    # LCM of 1..30 for integer conversion
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
    # L = 2329089562800

    # Weights: w[i-1] = L // i for i in 1..30
    weights = [L // i for i in range(1, 31)]

    # Meet-in-the-middle: split into two halves of 15
    n1 = 15
    n2 = 15

    def gen_half_weights(w_list):
        """Generate all 3^len(w_list) signed sums for the given weights."""
        n = len(w_list)
        # Use iterative generation
        sums = [0]
        for w in w_list:
            new_sums = []
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
        return sums

    def gen_half_coeffs(w_list):
        """Generate all 3^len(w_list) coefficient assignments."""
        n = len(w_list)
        coeffs = [[]]
        for _ in range(n):
            new_coeffs = []
            for c in coeffs:
                new_coeffs.append(c + [-1])
                new_coeffs.append(c + [0])
                new_coeffs.append(c + [1])
            coeffs = new_coeffs
        return coeffs

    # Generate first half
    w1 = weights[:n1]
    w2 = weights[n1:]

    print(f"Generating half 1 sums ({3**n1} entries)...", flush=True)
    sums1 = gen_half_weights(w1)
    print(f"Generating half 2 sums ({3**n2} entries)...", flush=True)
    sums2 = gen_half_weights(w2)

    # Sort sums2 with indices for binary search
    sums2_arr = np.array(sums2, dtype=np.int64)
    sorted_idx = np.argsort(sums2_arr)
    sorted_sums2 = sums2_arr[sorted_idx]

    # For each sum in sums1, find the sum in sums2 that makes total closest to 0
    # total = s1 + s2, want |total| minimal and nonzero
    best_total = None
    best_idx1 = None
    best_idx2 = None

    sums1_arr = np.array(sums1, dtype=np.int64)

    # Vectorized search: for each s1, find s2 such that s1+s2 is closest to 0
    # i.e., s2 closest to -s1
    neg_s1 = -sums1_arr
    # Binary search for insertion point in sorted_sums2
    insert_pts = np.searchsorted(sorted_sums2, neg_s1)

    # Check candidates around insertion point
    candidates = []
    for offset in [-1, 0, 1]:
        idxs = np.clip(insert_pts + offset, 0, len(sorted_sums2) - 1)
        s2_vals = sorted_sums2[idxs]
        totals = neg_s1 + s2_vals  # = -s1 + s2... wait
        # Actually total = s1 + s2, and we want s2 ≈ -s1
        # So total = s1 + s2, and s2 = sorted_sums2[idxs]
        # Let me redo: for each i, s1_i, we want s2 such that s1_i + s2 ≈ 0
        # s2 ≈ -s1_i
        # total = s1_i + s2
        pass

    # Simpler: compute totals directly
    # For each i in 0..len(sums1)-1, and for each candidate j near searchsorted position:
    best_abs = None
    best_i = -1
    best_j = -1

    for i in range(len(sums1_arr)):
        if time.time() > deadline:
            break
        target = -sums1_arr[i]
        pos = np.searchsorted(sorted_sums2, target)
        for j_offset in range(-2, 3):
            j = pos + j_offset
            if 0 <= j < len(sorted_sums2):
                total = sums1_arr[i] + sorted_sums2[j]
                if total == 0:
                    continue  # skip zero
                abs_total = abs(total)
                if best_abs is None or abs_total < best_abs:
                    best_abs = abs_total
                    best_i = i
                    best_j = j
                    if best_abs == 1:
                        break
        if best_abs == 1:
            break

    if best_i >= 0 and best_j >= 0:
        # Recover coefficients
        # We need to map index in sums1 to coefficients, and index in sorted_sums2 to original index in sums2
        orig_idx2 = sorted_idx[best_j]

        # Reconstruct coefficients from sums1 index
        # This requires knowing how the sums were generated
        # Better approach: regenerate with coefficient tracking

        # Let's use a different approach: enumerate with coefficients tracked
        pass

    # Alternative: just do full enumeration with coefficient tracking for the winning combination
    # Since we found best_i and best_j, let's reconstruct

    # Actually, let's redo this more carefully with coefficient tracking
    # Use a recursive/iterative approach that tracks both sum and coefficients

    def gen_with_coeffs(w_list):
        """Generate (sum, coeffs) pairs for all 3^n assignments."""
        results = [(0, [])]
        for w in w_list:
            new_results = []
            for s, c in results:
                new_results.append((s - w, c + [-1]))
                new_results.append((s, c + [0]))
                new_results.append((s + w, c + [1]))
            results = new_results
        return results

    print("Regenerating with coefficient tracking...", flush=True)
    if time.time() < deadline - 30:
        pairs1 = gen_with_coeffs(w1)
        pairs2 = gen_with_coeffs(w2)

        # Build lookup for half 2: sum -> list of (coeffs, original_index)
        # For efficiency, use dict mapping sum to coeffs (keep first or all)
        sum2_to_coeffs = {}
        for idx, (s, c) in enumerate(pairs2):
            if s not in sum2_to_coeffs:
                sum2_to_coeffs[s] = c

        # Sort unique sums from half 2
        unique_sums2 = sorted(sum2_to_coeffs.keys())
        unique_sums2_arr = np.array(unique_sums2, dtype=np.int64)

        # Find best combination
        best_abs_total = None
        best_c1 = None
        best_c2 = None

        for s1, c1 in pairs1:
            if time.time() > deadline - 5:
                break
            target = -s1
            pos = np.searchsorted(unique_sums2_arr, target)
            for j_off in range(-3, 4):
                j = pos + j_off
                if 0 <= j < len(unique_sums2_arr):
                    s2 = int(unique_sums2_arr[j])
                    total = s1 + s2
                    if total == 0:
                        continue
                    abs_t = abs(total)
                    if best_abs_total is None or abs_t < best_abs_total:
                        best_abs_total = abs_t
                        best_c1 = c1
                        best_c2 = sum2_to_coeffs[s2]
                        if best_abs_total == 1:
                            break
            if best_abs_total == 1:
                break

        if best_c1 is not None and best_c2 is not None:
            coeffs = best_c1 + best_c2
            # Verify
            s = Fraction(0, 1)
            for i, c in enumerate(coeffs, 1):
                if c != 0:
                    s += Fraction(c, i)
            if s != 0:
                return coeffs

    # Fallback: local search
    def compute_sum(coeffs):
        s = Fraction(0, 1)
        for i, c in enumerate(coeffs, 1):
            if c != 0:
                s += Fraction(c, i)
        return s

    def objective(coeffs):
        s = compute_sum(coeffs)
        if s == 0:
            return None
        return abs(s)

    best_coeffs = None
    best_val = None

    def update_best(coeffs):
        nonlocal best_coeffs, best_val
        val = objective(coeffs)
        if val is not None:
            if best_val is None or val < best_val:
                best_val = val
                best_coeffs = list(coeffs)

    import random
    random.seed(42)

    def local_search(coeffs, max_iter=100):
        nonlocal best_coeffs, best_val
        current = list(coeffs)
        current_val = objective(current)
        if current_val is not None:
            update_best(current)

        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            for i in range(30):
                if time.time() > deadline:
                    break
                old = current[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old:
                        continue
                    current[i] = new_val
                    new_obj = objective(current)
                    if new_obj is not None:
                        if current_val is None or new_obj < current_val:
                            current_val = new_obj
                            improved = True
                            update_best(current)
                            break
                    current[i] = old
                if improved:
                    break
            if not improved:
                break
        return current

    seed_coeffs = [0]*28 + [1, -1]
    local_search(seed_coeffs, max_iter=50)

    for r in range(50):
        if time.time() > deadline:
            break
        coeffs = [random.choice([-1, 0, 1]) for _ in range(30)]
        local_search(coeffs, max_iter=30)

    if best_coeffs is not None:
        return best_coeffs
    else:
        return [0]*28 + [1, -1]
# EVOLVE-BLOCK-END
