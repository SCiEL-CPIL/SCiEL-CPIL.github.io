import time
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before the 90s kill

    # Search for the minimum number of distinct denominators >= 5 whose
    # reciprocal sum is exactly 1.  k=3..7 are provably infeasible because
    # even the k smallest valid denominators (5,6,...) sum to less than 1.
    for k in range(3, 257):
        if time.time() > deadline - 3:
            break
        result = search_k(k, deadline)
        if result is not None:
            return result
    raise RuntimeError("no solution found within budget")


def search_k(k, deadline):
    target = Fraction(1, 1)

    def dfs(remaining_sum, min_den, depth, chosen):
        if time.time() > deadline:
            return None
        rem_terms = k - depth
        if rem_terms == 0:
            return chosen if remaining_sum == 0 else None

        # Upper bound: the largest sum achievable with rem_terms distinct
        # denominators all >= min_den is 1/min_den + 1/(min_den+1) + ...
        max_possible = Fraction(0, 1)
        for i in range(rem_terms):
            max_possible += Fraction(1, min_den + i)
        if remaining_sum > max_possible:
            return None

        # Bounds on the next denominator d (the smallest of the remaining terms):
        #   d >= min_den
        #   1/d <= remaining_sum          -> d >= ceil(1/remaining_sum)
        #   rem_terms/d >= remaining_sum  -> d <= floor(rem_terms/remaining_sum)
        lo = min_den
        lo2 = (remaining_sum.denominator + remaining_sum.numerator - 1) // remaining_sum.numerator
        if lo2 > lo:
            lo = lo2
        hi = (rem_terms * remaining_sum.denominator) // remaining_sum.numerator
        if lo > hi:
            return None

        for d in range(lo, hi + 1):
            new_sum = remaining_sum - Fraction(1, d)
            if new_sum < 0:
                break
            res = dfs(new_sum, d + 1, depth + 1, chosen + [d])
            if res is not None:
                return res
        return None

    return dfs(target, 5, 0, [])
