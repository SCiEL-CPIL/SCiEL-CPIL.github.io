"""Find Egyptian fraction of 5/31 minimizing the largest denominator."""
import time
from fractions import Fraction
from itertools import combinations
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85  # safety margin (90s kill)

    target = Fraction(5, 31)

    def check_time():
        return time.time() > deadline

    # Meet-in-the-middle approach:
    # Split candidate denominators into two halves, enumerate all subset sums,
    # then look for pairs that sum to target.
    #
    # Key insight: if we limit denominators to [7, M], we can find solutions
    # with max denominator M. We binary search on M to find the minimum.
    #
    # But first, let's try a direct approach: enumerate all subsets of
    # denominators from a small set and check if any sum to 5/31.
    #
    # For denominators up to ~100, the number of subsets is 2^94 which is too large.
    # Instead, use meet-in-the-middle: split into two groups of ~47 each.
    # 2^47 is still too large.
    #
    # Better approach: use the fact that we need EXACTLY 5/31.
    # Multiply everything by 31: sum 31/d_i = 5.
    # So we need sum of (31/d_i) = 5 where each 31/d_i is a rational.
    #
    # Alternative: work with common denominator. If all d_i divide LCM(7..M),
    # then sum 1/d_i = 5/31 means sum (LCM/d_i) = 5 * LCM / 31.
    # This requires 31 | LCM, which is true if M >= 31.
    #
    # Let's try a smarter approach: enumerate subsets of small denominators
    # using a priority queue / branch and bound, tracking the remaining sum.
    #
    # Actually, let me try the most direct approach first:
    # For a given max denominator M, check if there exists a subset of
    # {7, 8, ..., M} whose reciprocal sum is exactly 5/31.
    #
    # Use dynamic programming on the numerator/denominator after clearing fractions.
    # This is tricky because denominators grow.
    #
    # Let me try a different strategy: use the modular constraint.
    # sum 1/d_i = 5/31
    # Multiply by 31*prod(d_i): sum prod(d_i)/d_i = 5*prod(d_i)
    # This is an integer equation.
    #
    # For small sets, we can enumerate. Let's try denominators up to 50 first
    # with a recursive backtracking that prunes aggressively.

    best_max = float('inf')
    best_result = None

    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)

    # Strategy: For each candidate max value M from 20 upward,
    # try to find a subset of {7, ..., M} summing to 5/31.
    # Use meet-in-the-middle with Fraction arithmetic.
    #
    # For M up to ~40, the set has ~34 elements. Split into two halves of 17.
    # 2^17 = 131072 subsets per half. Manageable.
    #
    # For larger M, we need more clever pruning.

    # First, try small M values with meet-in-the-middle
    for M in range(20, 60):
        if check_time():
            break
        if M >= best_max:
            break

        candidates = list(range(7, M + 1))
        n = len(candidates)
        if n == 0:
            continue

        # Meet in the middle
        mid = n // 2
        left = candidates[:mid]
        right = candidates[mid:]

        # Enumerate all subset sums of left half
        left_sums = {}
        nl = len(left)
        for mask in range(1 << nl):
            if mask == 0:
                continue
            s = Fraction(0)
            for i in range(nl):
                if mask & (1 << i):
                    s += Fraction(1, left[i])
            # Store: for each sum, store the mask (to reconstruct)
            if s not in left_sums:
                left_sums[s] = mask

        # Enumerate all subset sums of right half and check for complement
        nr = len(right)
        found = False
        for mask in range(1 << nr):
            if mask == 0:
                continue
            s = Fraction(0)
            for i in range(nr):
                if mask & (1 << i):
                    s += Fraction(1, right[i])
            complement = target - s
            if complement in left_sums:
                # Found a solution! Reconstruct
                lmask = left_sums[complement]
                denoms = []
                for i in range(nl):
                    if lmask & (1 << i):
                        denoms.append(left[i])
                for i in range(nr):
                    if mask & (1 << i):
                        denoms.append(right[i])
                if len(denoms) >= 1 and len(denoms) <= 256:
                    # Verify
                    total = sum(Fraction(1, d) for d in denoms)
                    if total == target:
                        update_best(denoms)
                        found = True
                        break
            if check_time():
                break
        if found:
            break
        if check_time():
            break

    # If we haven't found anything with M < 60, try a recursive backtracking
    # approach with tighter bounds for larger M values.
    if best_result is None or best_max > 60:
        if not check_time():
            # Use recursive search with pruning
            # 1/a1 + 1/a2 + ... + 1/ak = 5/31, 7 <= a1 < a2 < ... < ak <= M
            # At each step, the remaining sum must be achievable with
            # remaining terms all >= current+1 and <= M.
            # Lower bound on remaining sum: if we need r more terms,
            # minimum sum is sum 1/(next_i) where next_i are consecutive.
            # Upper bound: maximum sum is r * 1/(current+1).

            def search(rem_sum, min_den, max_den, current, remaining_terms_min, remaining_terms_max):
                nonlocal best_max, best_result
                if check_time():
                    return False
                if rem_sum == 0:
                    if current:
                        mx = max(current)
                        if mx < best_max:
                            best_max = mx
                            best_result = list(current)
                    return True
                if rem_sum < 0:
                    return False
                if not current and remaining_terms_min > 0:
                    return False

                # We need at least 1 more term
                # The next denominator d must satisfy:
                # 1/d <= rem_sum (so d >= 1/rem_sum)
                # And we need enough remaining terms to make up the rest.

                # If we add one more term d, remaining is rem_sum - 1/d
                # That remaining must be >= 0 and achievable.

                # Lower bound on d: d >= ceil(1/rem_sum) but also d >= min_den
                # Upper bound on d: we need (remaining_terms_max - len(current)) terms
                # each at most 1/(d+1) ... this is complex.
                # Simpler: d <= floor(1/rem_sum) if we only use 1 term,
                # but with multiple terms d can be smaller.
                #
                # Key bound: if we use k more terms starting from d,
                # max possible sum is k/d (all equal to 1/d, but they must be distinct
                # and increasing, so max sum < k/d).
                # Min possible sum with k terms starting from d is
                # sum_{i=0}^{k-1} 1/(d+i).
                #
                # For feasibility: rem_sum <= max_sum(k, d) and rem_sum >= min_sum(k, d)

                # Try d from min_den upward
                # d must satisfy: 1/d < rem_sum (strict, since we need more terms or exact)
                # Actually 1/d <= rem_sum
                d_min = max(min_den, (1 / rem_sum.numerator) * rem_sum.denominator + 1 if rem_sum.denominator > 0 else min_den)
                # d_min = max(min_den, ceil(1/rem_sum)) but 1/rem_sum might not be integer
                # 1/rem_sum = rem_sum.denominator / rem_sum.numerator
                # d >= floor(rem_sum.denominator / rem_sum.numerator) + 1 if not divisible
                # Actually: 1/d <= rem_sum => d >= 1/rem_sum => d >= ceil(1/rem_sum)
                # But we also need 1/d < rem_sum if we have more terms, or 1/d == rem_sum if last term
                # Let's just use d >= min_den and d <= some upper bound

                # Upper bound: with at most (remaining_terms_max - len(current)) terms,
                # the largest d we can use is bounded by needing enough sum.
                # If we use exactly 1 more term: d = 1/rem_sum must be integer
                # If we use 2+ terms: d < 1/rem_sum * (number of terms)
                # Conservative: d <= 2 * (1/rem_sum) for 2 terms, etc.
                # Let's cap d at a reasonable value.

                num_terms_left_max = remaining_terms_max - len(current)
                # d * rem_sum <= num_terms_left_max (rough bound)
                # d <= num_terms_left_max / rem_sum
                d_upper = int(num_terms_left_max / rem_sum) + 1
                d_upper = min(d_upper, max_den)

                # Also, the minimum achievable sum with num_terms_left_max terms
                # starting from d is sum_{i=0}^{num_terms_left_max-1} 1/(d+i)
                # This must be <= rem_sum
                # And the maximum is num_terms_left_max / d (loose)

                for d in range(d_min, d_upper + 1):
                    if check_time():
                        return False
                    if d > max_den:
                        break
                    if d >= best_max and current:
                        # If d >= best_max, the max will be at least d >= best_max
                        # Only continue if d < best_max
                        break
                    new_rem = rem_sum - Fraction(1, d)
                    if new_rem < 0:
                        continue
                    if new_rem == 0:
                        # Found complete solution
                        new_current = current + [d]
                        mx = max(new_current)
                        if mx < best_max:
                            best_max = mx
                            best_result = list(new_current)
                        return True
                    # Recurse
                    # Minimum number of additional terms: 1 (if new_rem is a unit fraction)
                    # Maximum: num_terms_left_max - 1
                    # Check if new_rem can be achieved with 1 term: new_rem must be 1/e for integer e
                    # 1/e = new_rem => e = 1/new_rem = new_rem.denominator / new_rem.numerator
                    if new_rem.denominator % new_rem.numerator == 0:
                        e = new_rem.denominator // new_rem.numerator
                        if e > d and e <= max_den and e < best_max:
                            new_current = current + [d, e]
                            mx = max(new_current)
                            if mx < best_max:
                                best_max = mx
                                best_result = list(new_current)
                            return True
                    # Try with more terms
                    search(new_rem, d + 1, max_den, current + [d], 1, num_terms_left_max - 1)

            # Try with increasing max_den
            for max_d in range(30, 100):
                if check_time():
                    break
                if max_d >= best_max:
                    break
                search(Fraction(5, 31), 7, max_d, [], 1, 256)

    if best_result is not None:
        # Verify the result
        total = sum(Fraction(1, d) for d in best_result)
        assert total == Fraction(5, 31), f"Verification failed: {total} != 5/31"
        assert all(1 <= d <= 10000000 for d in best_result)
        assert len(set(best_result)) == len(best_result)
        assert 1 <= len(best_result) <= 256
        return best_result

    # Fallback: use the 3-term search from original
    target_num = 5
    target_den = 31
    for a in range(7, 19):
        rem_num = 5 * a - 31
        rem_den = 31 * a
        if rem_num <= 0:
            continue
        min_b = max(a + 1, rem_den // rem_num + 1)
        max_b = (2 * rem_den) // rem_num
        for b in range(min_b, max_b + 1):
            rem2_num = rem_num * b - rem_den
            rem2_den = rem_den * b
            if rem2_num <= 0:
                continue
            if rem2_den % rem2_num == 0:
                c = rem2_den // rem2_num
                if c > b and c <= 10000000:
                    return [a, b, c]

    return [7, 8, 9, 10000000]
