"""
Find a set of distinct denominators whose reciprocals sum to exactly 5/31,
minimizing the largest denominator.

Approach:
1. Quick baseline via greedy (Fibonacci) expansion.
2. DFS search with strong pruning:
   - Remaining value r with min_den d_min and max_den M:
     * If r == 0: success
     * If r < 0: prune
     * If r > (M - d_min + 1) / (d_min * M): even with all denominators from
       d_min to M, the sum 1/d_min + 1/(d_min+1) + ... + 1/M is insufficient.
     * If r < 1/M: the smallest possible single term exceeds r (can't use any term)
     * If r > (number_of_terms_remaining) * 1/d_min: even with k copies of 1/d_min
       (but we need distinct, so this is a looser bound; use sum of k smallest)
   - Try denominators from d_min to M in increasing order.
   - For speed, precompute prefix sums of 1/i for the harmonic series up to M.
"""
import time
from fractions import Fraction
import math

def construct():
    target = Fraction(5, 31)
    deadline = time.time() + 80  # safety margin before 90s limit

    # Quick greedy baseline
    def greedy_expand(frac, min_d=1, used=None):
        terms = []
        r = frac
        min_d = min_d
        used = set(used) if used else set()
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d < min_d:
                d = min_d
            while d in used:
                d += 1
            terms.append(d)
            r -= Fraction(1, d)
            min_d = d + 1
            used.add(d)
        return terms

    baseline = greedy_expand(target)
    best_solution = baseline[:] if baseline else None
    best_max_den = max(baseline) if baseline else float('inf')

    # Precompute harmonic sums: H[n] = sum_{i=1}^{n} 1/i as Fraction
    # We'll need this for pruning. But computing up to large n is expensive.
    # Instead, use a tighter analytical bound:
    # sum_{i=a}^{b} 1/i >= ln(b) - ln(a) (integral bound, lower bound)
    # sum_{i=a}^{b} 1/i <= 1/a + (b-a)/b (rough upper bound)
    # Actually, for pruning: if r > sum_{i=d_min}^{M} 1/i, prune.
    # We can approximate sum_{i=a}^{b} 1/i with harmonic numbers or use
    # the bound: sum_{i=a}^{b} 1/i <= 1/a + (b-a)/b
    # And a better lower bound: sum_{i=a}^{b} 1/i >= ln(b) - ln(a-1)
    
    # For exactness, let's precompute harmonic sums up to a reasonable limit.
    # But M could be large. Let's use a dynamic approach.
    
    # Better pruning: 
    # Given r, d_min, M, and k terms remaining (k can be up to 256):
    # Maximum possible sum with k distinct terms from [d_min, M] is the sum of the
    # k smallest reciprocals: 1/d_min + 1/(d_min+1) + ... + 1/(d_min+k-1)
    # (assuming d_min+k-1 <= M; if not, use 1/d_min + ... + 1/M)
    # Minimum possible sum with k distinct terms: 1/(M-k+1) + ... + 1/M
    # (if M-k+1 >= d_min; else 1/d_min + ... + 1/M)
    
    # But we don't fix k. We can use up to 256 terms.
    # So the maximum sum is sum_{i=d_min}^{M} 1/i (using all available).
    # The minimum sum with 1 term is 1/M.
    
    # Key pruning rules:
    # 1. If r == 0: success
    # 2. If r < 0: prune
    # 3. If r > sum_{i=d_min}^{M} 1/i: prune (even using all denominators isn't enough)
    # 4. If r < 1/M: prune (smallest possible term is too small... wait, no)
    #    Actually if r > 0, we need at least one term. The smallest term we can add
    #    is 1/M. If r < 1/M, we still need more terms. This isn't a prune condition
    #    by itself.
    # 5. If we need exactly 1 more term: check if 1/r is an integer d with d_min <= d <= M.
    
    # Let's implement DFS with memoization on (r, d_min, M) -> boolean whether solvable.
    # But r is a Fraction, so memoization might be expensive. Let's try without first.
    
    # To speed up sum_{i=d_min}^{M} 1/i computation, precompute prefix harmonic sums.
    # We'll precompute up to some reasonable max. If M exceeds it, use approximation.
    
    MAX_PRECOMPUTE = 100000
    harmonic = [Fraction(0, 1)] * (MAX_PRECOMPUTE + 1)
    for i in range(1, MAX_PRECOMPUTE + 1):
        harmonic[i] = harmonic[i-1] + Fraction(1, i)
    
    def harmonic_sum(a, b):
        """Return sum_{i=a}^{b} 1/i exactly."""
        if a > b:
            return Fraction(0, 1)
        if b <= MAX_PRECOMPUTE:
            return harmonic[b] - harmonic[a-1]
        else:
            # Fallback: approximate with ln(b) - ln(a-1) + small correction
            # This is a lower bound for the true sum.
            # For pruning, we want an upper bound on the max possible sum.
            # Actually for "can we reach r?", we need: is r <= max_possible?
            # max_possible = sum_{i=a}^{b} 1/i
            # We need an upper bound on this sum to safely prune.
            # sum_{i=a}^{b} 1/i <= 1/a + integral_a^b 1/x dx = 1/a + ln(b/a)
            import math as _math
            return Fraction(1, a) + Fraction(_math.log(b / a), 1)
    
    def can_complete(r, d_min, M, depth=0):
        """Check if r can be expressed as sum of distinct reciprocals in [d_min, M]."""
        if r == 0:
            return True
        if r < 0:
            return False
        if d_min > M:
            return False
        # Upper bound on achievable sum
        max_sum = harmonic_sum(d_min, M)
        if r > max_sum:
            return False
        # If r is very small, we might need many terms.
        # Lower bound: with 1 term, min is 1/M. With k terms, min is sum of k largest reciprocals
        # which are 1/(M-k+1) + ... + 1/M.
        # But we can use up to 256 terms.
        # If r < 1/M and we can only use 1 term, prune. But we can use more.
        # Actually, the minimum positive sum with distinct terms from [d_min, M] 
        # using at least 1 term is 1/M.
        # So if 0 < r < 1/M, we need at least 2 terms.
        # This gets complicated. Let's just try the DFS.
        
        # Try each possible next denominator
        for d in range(d_min, M + 1):
            new_r = r - Fraction(1, d)
            if new_r == 0:
                return True
            if new_r < 0:
                break  # Further d will only make new_r smaller
            if can_complete(new_r, d + 1, M, depth + 1):
                return True
            if depth > 20:  # Limit recursion depth for speed
                break
        return False

    def find_solution(r, d_min, M, max_terms=256):
        """Find a list of denominators summing to r, each in [d_min, M], distinct, at most max_terms."""
        if r == 0:
            return []
        if r < 0 or d_min > M or max_terms <= 0:
            return None
        max_sum = harmonic_sum(d_min, M)
        if r > max_sum:
            return None
        
        for d in range(d_min, M + 1):
            new_r = r - Fraction(1, d)
            if new_r < 0:
                break
            if new_r == 0:
                return [d]
            sub = find_solution(new_r, d + 1, M, max_terms - 1)
            if sub is not None:
                return [d] + sub
        return None

    # Strategy: Try decreasing max denominator M, starting from best_max_den - 1
    current_M = best_max_den - 1
    found_improvement = False

    while current_M >= 7 and time.time() < deadline:
        sol = find_solution(target, 7, current_M, 256)
        if sol is not None and len(sol) >= 1 and len(sol) <= 256:
            best_solution = sol
            best_max_den = current_M
            found_improvement = True
            current_M -= 1
        else:
            # If no solution found with this M, try a slightly smaller M? 
            # Or break to save time.
            # Let's try a few more values of M with different strategies.
            # For now, break to avoid spending too much time on infeasible M.
            # But maybe a solution exists with a different structure.
            # Let's do one more try with a randomized start.
            break

    # If time remains, try randomized local search to find better solutions
    if time.time() < deadline - 10 and best_solution is not None:
        import random
        random.seed(12345)
        # Try to find solutions with various M values
        for M in range(max(7, best_max_den - 20), best_max_den):
            if time.time() > deadline:
                break
            # Try different starting denominators
            for d1 in range(7, min(30, M + 1)):
                if time.time() > deadline:
                    break
                r1 = target - Fraction(1, d1)
                if r1 < 0:
                    continue
                if r1 == 0:
                    continue
                sol = find_solution(r1, d1 + 1, M, 255)
                if sol is not None and len(sol) <= 255:
                    full = [d1] + sol
                    if max(full) <= M and len(full) <= 256:
                        best_solution = full
                        best_max_den = max(full)
                        break

    if best_solution is None:
        best_solution = greedy_expand(target)

    # Validate
    assert sum(Fraction(1, d) for d in best_solution) == target
    assert len(set(best_solution)) == len(best_solution)
    assert all(1 <= d <= 10000000 for d in best_solution)
    assert 1 <= len(best_solution) <= 256

    return best_solution
