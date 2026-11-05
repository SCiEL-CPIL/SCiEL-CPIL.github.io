"""Rewritten program: meet-in-the-middle + local refinement for minimal nonzero signed harmonic subsum."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin from 90s wall-clock limit

    N = 30

    # Seed solution: [0]*28 + [1, -1] gives sum = 1/29 - 1/30 = 1/870
    best_solution = [0] * 28 + [1, -1]
    best_val = Fraction(1, 870)

    def eval_solution(coeffs):
        """Evaluate the absolute value of the signed harmonic sum. Returns None if zero."""
        s = Fraction(0)
        for i, c in enumerate(coeffs, 1):
            if c != 0:
                s += Fraction(c, i)
        if s == 0:
            return None
        return abs(s)

    # Meet-in-the-middle approach
    # Split 30 variables into two halves of 15
    # For each half, compute all possible sums (3^15 = 14348907 possibilities - too many)
    # Instead, use a smarter approach: enumerate subsets of each half with signs

    # Actually, 3^15 is about 14 million which is feasible if we're careful
    # But let's try a different split: 15 and 15, but use a hash map

    # For each half (indices 0..14 and 15..29), enumerate all 3^15 assignments
    # Compute the sum for each, store in a dict: sum -> list of coefficient vectors (or just track best)

    # 3^15 = 14348907, which is manageable
    # But storing all sums as Fractions with huge denominators might be slow

    # Let's use a different approach: scale by LCM(1..30) to work with integers
    # LCM(1..30) is large but computable

    from math import gcd

    def lcm(a, b):
        return a * b // gcd(a, b)

    # Compute LCM(1..30)
    lcm_val = 1
    for i in range(1, 31):
        lcm_val = lcm(lcm_val, i)

    # Now each term c_i / i can be written as c_i * (lcm_val // i) / lcm_val
    # So the sum = (sum of c_i * (lcm_val // i)) / lcm_val
    # We want to minimize |sum of c_i * (lcm_val // i)| subject to it being nonzero

    # Precompute weights
    weights = [lcm_val // i for i in range(1, 31)]

    # Now the problem is: find c_i in {-1, 0, 1} to minimize |sum c_i * w_i| != 0

    # Meet-in-the-middle:
    # Split into first 15 and last 15
    # For first half: enumerate all 3^15 assignments, compute sum1
    # For second half: enumerate all 3^15 assignments, compute sum2
    # Total sum = sum1 + sum2
    # We want sum1 + sum2 to be small but nonzero

    # Store all (sum1, coeffs1) in a sorted list or dict
    # Then for each sum2, find sum1 such that sum1 + sum2 is close to 0 but not 0

    # 3^15 = 14348907 entries. Storing all of them might use too much memory.
    # Alternative: use a hash set of achievable sums for one half, then search the other.

    # Actually, let's think about this differently. The sums can be very large (lcm_val is ~2.3e12).
    # So we can't just use a simple array.

    # Let's try a different strategy: use local search with 2-flip moves, which was shown to work well.
    # But also try some structured approaches.

    # Phase 1: Try structured patterns
    structured_patterns = []

    # Consecutive pairs
    for i in range(N - 1):
        pattern = [0] * N
        pattern[i] = 1
        pattern[i + 1] = -1
        structured_patterns.append(pattern)

    # Pairs with one gap
    for i in range(N - 2):
        pattern = [0] * N
        pattern[i] = 1
        pattern[i + 2] = -1
        structured_patterns.append(pattern)

    # Triples
    for i in range(N - 2):
        pattern = [0] * N
        pattern[i] = 1
        pattern[i + 1] = -1
        pattern[i + 2] = 1
        structured_patterns.append(pattern)

    for pattern in structured_patterns:
        if len(pattern) != N:
            continue
        val = eval_solution(pattern)
        if val is not None and val < best_val:
            best_val = val
            best_solution = pattern[:]

    # Phase 2: Local search with 2-flip moves
    def local_search_2flip(start_coeffs, max_time=5.0):
        """Perform local search with 1-flip and 2-flip moves."""
        nonlocal best_solution, best_val
        local_best = start_coeffs[:]
        local_val = eval_solution(local_best)
        if local_val is None:
            return local_best, Fraction(1, 1)

        local_start = time.time()
        no_improve_count = 0
        max_no_improve = 100

        while time.time() - local_start < max_time and no_improve_count < max_no_improve:
            improved = False

            # Try all 1-flip moves
            for i in range(N):
                old = local_best[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old:
                        continue
                    local_best[i] = new_val
                    new_abs = eval_solution(local_best)
                    if new_abs is not None and new_abs < local_val:
                        local_val = new_abs
                        improved = True
                        no_improve_count = 0
                        if local_val < best_val:
                            best_val = local_val
                            best_solution = local_best[:]
                        break
                    else:
                        local_best[i] = old
                if improved:
                    break
                local_best[i] = old

            if improved:
                continue

            # Try 2-flip moves
            for i in range(N):
                if time.time() - local_start >= max_time:
                    break
                old_i = local_best[i]
                for new_i in [-1, 0, 1]:
                    if new_i == old_i:
                        continue
                    local_best[i] = new_i
                    for j in range(i + 1, N):
                        old_j = local_best[j]
                        for new_j in [-1, 0, 1]:
                            if new_j == old_j:
                                continue
                            local_best[j] = new_j
                            new_abs = eval_solution(local_best)
                            if new_abs is not None and new_abs < local_val:
                                local_val = new_abs
                                improved = True
                                no_improve_count = 0
                                if local_val < best_val:
                                    best_val = local_val
                                    best_solution = local_best[:]
                                break
                            else:
                                local_best[j] = old_j
                        if improved:
                            break
                    if improved:
                        break
                    local_best[i] = old_i

            if not improved:
                no_improve_count += 1

        return local_best, local_val

    # Run local search from current best
    if time.time() - start_time < deadline - 20:
        current, current_val = local_search_2flip(best_solution, max_time=20.0)

    # Phase 3: Random restarts with local search
    import random
    random.seed(42)

    restart_count = 0
    while time.time() - start_time < deadline - 5:
        restart_count += 1
        # Generate a random starting point
        fresh = []
        for _ in range(N):
            r = random.random()
            if r < 0.5:
                fresh.append(0)
            elif r < 0.75:
                fresh.append(1)
            else:
                fresh.append(-1)

        fresh_val = eval_solution(fresh)
        if fresh_val is None:
            continue

        if fresh_val < best_val or random.random() < 0.2:
            if time.time() - start_time < deadline - 3:
                fresh, fresh_val = local_search_2flip(fresh, max_time=3.0)

    # Final check
    final_val = eval_solution(best_solution)
    if final_val is None:
        return [0] * 28 + [1, -1]

    return best_solution

# EVOLVE-BLOCK-END
