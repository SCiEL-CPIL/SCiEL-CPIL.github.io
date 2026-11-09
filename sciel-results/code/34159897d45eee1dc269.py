"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
from itertools import product
import random

def construct():
    """
    Find 30 coefficients in {-1, 0, 1} minimizing |sum c[i-1]/i| (nonzero).
    Returns the list of 30 coefficients.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Leave margin before 90s kill

    N = 30
    # Denominators are 1..30, so coefficient index i corresponds to denominator i+1
    # c[0]/1 + c[1]/2 + ... + c[29]/30

    # We want to minimize |S| where S = sum_{i=1}^{30} c_i / i, c_i in {-1,0,1}, S != 0.
    # The value is a rational number with denominator dividing L = lcm(1..30).
    # L = lcm(1,2,...,30) = 2329089562800 (approx 2.3e12)
    # S = k / L for some integer k. We want to minimize |k|/L, i.e., minimize |k| with k != 0.
    # k = sum_{i=1}^{30} c_i * (L / i)

    # Compute L
    L = 1
    for i in range(1, N + 1):
        L = L * i // gcd(L, i)

    # weights[i] = L / i for denominator i (1-indexed)
    weights = [L // i for i in range(1, N + 1)]

    # We need to find min |sum c_i * w_i| != 0, c_i in {-1,0,1}.
    # This is a variant of the closest-to-zero subset sum problem with 3 ternary choices per item.

    # Strategy: Meet-in-the-middle with pruning.
    # Split into two halves: first 15 denominators, last 15 denominators.
    # For each half, compute all possible sums.
    # There are 3^15 = 14348907 possible sums per half. That's manageable.
    # But we need to combine them to find the closest to zero (excluding zero).

    # Actually, 3^15 ~ 14 million. Storing all is feasible in memory (~100MB per half).
    # Let's try a smarter approach: sort one half's sums, then for each sum in the other half,
    # binary search for the value that minimizes the combined absolute value.

    # But first, let's check if we can get |k| = 1. If so, that's optimal.
    # Let's use a DP or set-based approach to check reachability of small k values.

    # Alternative: Use a set of reachable sums for the first half, sort it.
    # For the second half, iterate through all sums and for each, find the best match in the first half.

    # Let's implement meet-in-the-middle.
    # Time to generate 3^15 sums: 14M iterations. In Python, this might be slow (~10-30s).
    # We need to be careful with the deadline.

    # Let's try a different approach: dynamic programming with a set of reachable residues modulo some large number?
    # No, we need exact minimization.

    # Let's try to find if k=1 is reachable. If not, k=2, etc.
    # We can use a BFS/DP approach where we track reachable sums in a set, but the sums can be huge (up to L).
    # However, we only care about small |k|. We can prune sums that are too far from 0.
    # But "too far" is hard to bound without knowing the optimum.

    # Let's stick with meet-in-the-middle but optimize the generation.
    # Use NumPy for vectorized generation?
    # Or just efficient Python loops with early termination if we find k=1.

    # Let's first try to find if any combination gives k=1 or k=-1.
    # We can do this by iterating through all 3^15 sums for the first half, storing them in a set.
    # Then for the second half, for each sum s2, check if (1 - s2) is in the set, or (-1 - s2) is in the set.
    # If yes, we have a solution with |k|=1.

    # Let's implement this.
    # Split indices: 0..14 (denoms 1..15) and 15..29 (denoms 16..30)

    def gen_sums(weights_subset):
        """Generate all possible sums for the given weights with coefficients in {-1,0,1}."""
        n = len(weights_subset)
        sums = set()
        sums.add(0)
        for w in weights_subset:
            new_sums = set()
            for s in sums:
                new_sums.add(s - w)
                new_sums.add(s)
                new_sums.add(s + w)
            sums = new_sums
            if time.time() - start_time > DEADLINE:
                break
        return sums

    # Generate sums for first half
    w1 = weights[:15]
    w2 = weights[15:]

    sums1 = gen_sums(w1)
    if time.time() - start_time > DEADLINE:
        # Fallback: return a known small solution
        # e.g., [0]*28 + [1, -1] gives 1/29 - 1/30 = 1/(29*30) = 1/870
        # k = L/870. This is not minimal but valid.
        return [0] * 28 + [1, -1]

    # Check if k=1 or k=-1 is reachable
    found_coeff = None
    for target in [1, -1]:
        # We need s1 + s2 = target => s1 = target - s2
        # We can iterate over sums2 and check if (target - s2) is in sums1
        # But sums2 is not generated yet. Let's generate it.
        pass

    # Let's generate sums2
    sums2 = gen_sums(w2)
    if time.time() - start_time > DEADLINE:
        return [0] * 28 + [1, -1]

    # Now check for k=1 and k=-1
    # To save time, iterate over the smaller set
    if len(sums1) < len(sums2):
        for s1 in sums1:
            if (1 - s1) in sums2:
                # Found k=1
                # Need to recover coefficients
                found_coeff = _recover_coefficients(w1, w2, s1, 1 - s1)
                break
            if (-1 - s1) in sums2:
                # Found k=-1
                found_coeff = _recover_coefficients(w1, w2, s1, -1 - s1)
                break
    else:
        for s2 in sums2:
            if (1 - s2) in sums1:
                found_coeff = _recover_coefficients(w1, w2, 1 - s2, s2)
                break
            if (-1 - s2) in sums1:
                found_coeff = _recover_coefficients(w1, w2, -1 - s2, s2)
                break

    if found_coeff is not None:
        return found_coeff

    # If k=1 or -1 not found, try k=2, -2, etc.
    # But this could be slow. Let's try a few small k values.
    for k in [2, -2, 3, -3, 4, -4, 5, -5]:
        if time.time() - start_time > DEADLINE - 5:
            break
        if len(sums1) < len(sums2):
            for s1 in sums1:
                if (k - s1) in sums2:
                    found_coeff = _recover_coefficients(w1, w2, s1, k - s1)
                    break
                if (-k - s1) in sums2:
                    found_coeff = _recover_coefficients(w1, w2, s1, -k - s1)
                    break
        else:
            for s2 in sums2:
                if (k - s2) in sums1:
                    found_coeff = _recover_coefficients(w1, w2, k - s2, s2)
                    break
                if (-k - s2) in sums1:
                    found_coeff = _recover_coefficients(w1, w2, -k - s2, s2)
                    break
        if found_coeff is not None:
            return found_coeff

    # If we couldn't find a small k, use a heuristic or fallback
    # Let's try a local search or just return the fallback
    return [0] * 28 + [1, -1]


def _recover_coefficients(w1, w2, s1, s2):
    """Recover the coefficients given the sums for each half."""
    def recover(weights, target_sum):
        n = len(weights)
        # Backtrack to find coefficients
        coeffs = [0] * n
        # Use recursive backtracking
        def backtrack(i, current_sum):
            if i == n:
                return current_sum == target_sum
            w = weights[i]
            # Try -1, 0, 1
            for c in [-1, 0, 1]:
                coeffs[i] = c
                if backtrack(i + 1, current_sum + c * w):
                    return True
            return False

        if backtrack(0, 0):
            return coeffs
        return None

    c1 = recover(w1, s1)
    c2 = recover(w2, s2)
    if c1 is None or c2 is None:
        return None
    return c1 + c2

# EVOLVE-BLOCK-END
