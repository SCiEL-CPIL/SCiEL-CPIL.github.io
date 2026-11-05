import time
import math
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    n = 30
    # We need coefficients c_1..c_30 in {-1,0,1}
    # Sum S = sum_{i=1}^{30} c_i / i
    # We want to minimize |S| subject to S != 0
    # S = N / D where D = LCM(1..30)
    # Let L = LCM(1..30)
    # Then S = (sum c_i * (L/i)) / L
    # Let a_i = L/i (integer)
    # We want to minimize |sum c_i * a_i| != 0

    # Compute LCM(1..30)
    from math import gcd
    L = 1
    for i in range(1, 31):
        L = L * i // gcd(L, i)

    # a_i = L / i
    a = [0] + [L // i for i in range(1, 31)]  # 1-indexed

    # Meet-in-the-middle:
    # Split into two halves: first 15 and last 15
    # For each half, enumerate all 3^15 = 14348907 assignments
    # That's about 14M each, total 28M. Might be tight in 85s.
    # Actually 3^15 = 14348907, so two halves = ~28.7M total enumerations.
    # Each is just a sum, so should be feasible.

    # Let's do meet-in-the-middle
    # Half 1: indices 1..15
    # Half 2: indices 16..30

    # For each half, compute all possible sums of c_i * a_i
    # Then find the pair (s1, s2) that minimizes |s1 + s2| != 0

    # 3^15 = 14,348,907
    # We'll enumerate all sums for each half, sort the second half's sums,
    # and for each sum in the first half, binary search for the closest match in the second half.

    # But storing 14M integers is fine memory-wise (~112 MB for Python ints, but we can use array)

    # Let's use a more efficient approach:
    # Generate all sums for half 2, sort them
    # For each sum in half 1, find the two closest values in half 2's sorted list
    # that make s1 + s2 closest to 0 (but not 0)

    # Time check: generating 14M sums takes some time. Let's estimate.
    # Each sum is computed incrementally. 14M * 15 operations = ~200M ops. Might be too slow in Python.

    # Alternative: use numpy for vectorized generation? Or use a smarter recursive generation.

    # Let's try a different approach: since 3^15 is large, let's use a smaller split.
    # Split into 3 groups of 10: 3^10 = 59049 each.
    # Generate all sums for each group (59049 each).
    # Then combine group 1 and group 2: 59049^2 = ~3.5B pairs - too many.

    # Back to 2-way split. Let's try to be efficient.
    # Use array module for compact storage.

    # Actually, let me reconsider. 3^15 = 14348907.
    # In Python, generating 14M integers might take ~5-10 seconds.
    # Sorting 14M integers takes ~5-10 seconds.
    # Binary searching 14M times takes ~14M * log2(14M) ≈ 14M * 24 = 336M ops - too slow in Python.

    # Better: sort both lists, then use two-pointer technique.
    # Sort list1 and list2. Use two pointers to find minimum |s1 + s2| != 0.
    # This is O(n log n) for sorting + O(n) for two-pointer.

    # Let's do that.

    # Generate all sums for first half (indices 1-15)
    def gen_sums(indices):
        """Generate all possible sums c_i * a_i for given indices."""
        sums = [0]
        for idx in indices:
            ai = a[idx]
            new_sums = []
            for s in sums:
                new_sums.append(s - ai)
                new_sums.append(s)
                new_sums.append(s + ai)
            sums = new_sums
        return sums

    # Time check
    if time.time() - start_time > DEADLINE:
        return [0] * 28 + [1, -1]

    half1 = list(range(1, 16))
    half2 = list(range(16, 31))

    # Generate sums for half1
    sums1 = gen_sums(half1)
    if time.time() - start_time > DEADLINE:
        return [0] * 28 + [1, -1]

    # Generate sums for half2
    sums2 = gen_sums(half2)
    if time.time() - start_time > DEADLINE:
        return [0] * 28 + [1, -1]

    # Sort both
    sums1.sort()
    sums2.sort()
    if time.time() - start_time > DEADLINE:
        return [0] * 28 + [1, -1]

    # Two-pointer to find min |s1 + s2| != 0
    # We want to minimize |s1 + s2| where s1 + s2 != 0
    # Use two pointers: i from 0 to len(sums1)-1, j from len(sums2)-1 to 0
    # Actually, standard two-pointer for closest pair sum to target (0 here).

    best_diff = None
    best_i = 0
    best_j = 0

    i = 0
    j = len(sums2) - 1
    while i < len(sums1) and j >= 0:
        diff = sums1[i] + sums2[j]
        if diff == 0:
            # Move to avoid zero
            # Try i+1 and j-1
            # We need to check neighbors
            # For now, skip and move both
            i += 1
            j -= 1
            continue
        if best_diff is None or abs(diff) < abs(best_diff):
            best_diff = diff
            best_i = i
            best_j = j
        if diff > 0:
            j -= 1
        else:
            i += 1

    # But this two-pointer only finds one candidate. We need to be more careful.
    # The standard approach for "closest pair to target" with two sorted arrays:
    # Start with i=0, j=len(sums2)-1
    # At each step, if sums1[i]+sums2[j] > 0, decrement j
    # if sums1[i]+sums2[j] < 0, increment i
    # Track the minimum absolute difference (nonzero)

    # Let me redo this properly
    best_abs = float('inf')
    best_diff = None
    best_i = 0
    best_j = 0

    i = 0
    j = len(sums2) - 1
    while i < len(sums1) and j >= 0:
        diff = sums1[i] + sums2[j]
        if diff != 0:
            if abs(diff) < best_abs:
                best_abs = abs(diff)
                best_diff = diff
                best_i = i
                best_j = j
        if diff > 0:
            j -= 1
        elif diff < 0:
            i += 1
        else:
            # diff == 0, try moving both to find nonzero
            i += 1
            j -= 1

    if time.time() - start_time > DEADLINE:
        return [0] * 28 + [1, -1]

    # Now recover the coefficients
    # sums1[best_i] is the sum for the first half
    # sums2[best_j] is the sum for the second half
    # We need to recover which c_i values produce these sums.

    # This requires storing the actual coefficient assignments, not just the sums.
    # That's 14M * 15 bytes = ~215 MB. Might be too much.

    # Alternative: store for each sum the index of the coefficient assignment.
    # Or, better: store the sums with their "path" (the tuple of coefficients).

    # Actually, let's store the sums along with the coefficient vector encoded as an integer.
    # Each coefficient is in {-1, 0, 1}, so we can encode as base-3 digit.
    # 15 coefficients -> 15 base-3 digits -> fits in an integer.

    # Let me redo the generation to store (sum, code) pairs.

    # Actually, this is getting complex. Let me use a simpler approach:
    # Store the sums, and after finding the best, regenerate the coefficients
    # by doing a DFS that tracks which sums are achievable.

    # Simpler: just store the coefficient vectors as tuples alongside the sums.
    # 14M tuples of 15 integers each... that's a lot of memory.

    # Alternative approach: since we just need to find the minimum, and the two-pointer
    # gives us the best sum values, we can do a second pass to recover the coefficients.

    # Let's use a different strategy: instead of meet-in-the-middle with full enumeration,
    # let's use a local search / simulated annealing approach which is simpler.

    # Actually, let me just do the full meet-in-the-middle but store the codes.

    # Regenerate with codes
    def gen_sums_with_codes(indices):
        """Generate (sum, code) pairs where code encodes coefficients in base 3."""
        results = [(0, 0)]
        for pos, idx in enumerate(indices):
            ai = a[idx]
            new_results = []
            for s, code in results:
                # c = -1, 0, 1 -> digit 0, 1, 2 in base 3
                for digit in range(3):
                    c = digit - 1  # -1, 0, 1
                    new_s = s + c * ai
                    new_code = code + digit * (3 ** pos)
                    new_results.append((new_s, new_code))
            results = new_results
        return results

    # This will use more memory but let's try
    if time.time() - start_time > 60.0:
        # Fall back to simple heuristic
        return [0] * 28 + [1, -1]

    # Actually, 14M * 2 integers = 28M integers. In Python, each int is ~28 bytes, so ~784 MB.
    # That's too much. Let's use arrays.

    # Let me use a different approach: use numpy to generate the sums more efficiently.

    # Actually, let me just use a smarter local search.

    # Simulated annealing / local search:
    # Start with a random valid assignment (nonzero sum)
    # Try flipping individual coefficients or swapping
    # Accept improvements, occasionally accept worse solutions

    import random
    random.seed(42)

    # Initial solution: random
    coeffs = [random.choice([-1, 0, 1]) for _ in range(30)]
    # Ensure nonzero sum
    while sum(c / (i+1) for i, c in enumerate(coeffs)) == 0:
        coeffs = [random.choice([-1, 0, 1]) for _ in range(30)]

    def compute_sum(c):
        return sum(c[i] / (i+1) for i in range(30))

    best_coeffs = coeffs[:]
    best_val = abs(compute_sum(coeffs))

    # Local search: try flipping each coefficient
    for iteration in range(100000):
        if time.time() - start_time > DEADLINE:
            break

        improved = False
        for i in range(30):
            old = coeffs[i]
            for new_val in [-1, 0, 1]:
                if new_val == old:
                    continue
                coeffs[i] = new_val
                s = compute_sum(coeffs)
                if s != 0:
                    val = abs(s)
                    if val < best_val:
                        best_val = val
                        best_coeffs = coeffs[:]
                        improved = True
                else:
                    # s == 0, revert
                    coeffs[i] = old
                    continue
                coeffs[i] = old
        if not improved:
            # Try random restarts or perturbations
            # Perturb 2-3 random positions
            for _ in range(3):
                i = random.randint(0, 29)
                coeffs[i] = random.choice([-1, 0, 1])
            s = compute_sum(coeffs)
            if s != 0:
                best_coeffs = coeffs[:]
                best_val = abs(s)

    return best_coeffs

# Test
if __name__ == "__main__":
    result = construct()
    print(result)
    s = sum(result[i] / (i+1) for i in range(30))
    print(f"Sum: {s}, |Sum|: {abs(s)}")
