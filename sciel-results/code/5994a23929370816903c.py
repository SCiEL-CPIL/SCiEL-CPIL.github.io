"""Rewritten: generate powerful numbers up to a large bound, count collisions by sum, verify best by exact factorization."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def _is_powerful(n):
    """Exact check: every prime factor of n appears with exponent >= 2. 1 is powerful."""
    if n < 1:
        return False
    if n == 1:
        return True
    d = 2
    while d * d <= n:
        if n % d == 0:
            cnt = 0
            while n % d == 0:
                n //= d
                cnt += 1
            if cnt < 2:
                return False
        d += 1 if d == 2 else 2  # 2,3,5,7,...
    if n > 1:
        return False  # remaining prime factor has exponent 1
    return True

def _gen_powerful(limit):
    """Generate all powerful numbers <= limit.
    A powerful number is of the form prod p_i^{e_i} with e_i >= 2.
    Enumerate square part and cube part: n = a^2 * b^3 with b squarefree? 
    Simpler: n = x^2 * y where y is a perfect cube and x^2 * y has all exponents >=2.
    Standard: powerful numbers are exactly { a^2 * b^3 : a,b >= 1 }.
    """
    pw = set()
    # b^3 <= limit => b <= limit^(1/3)
    max_b = int(round(limit ** (1.0/3.0))) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        # a^2 <= limit // b3
        max_a = math.isqrt(limit // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= limit:
                pw.add(n)
    return sorted(pw)

def construct():
    t0 = time.time()
    DEADLINE = 130  # safety margin under 150s wall-clock

    # Generate powerful numbers up to a generous bound.
    # Collisions among powerful numbers give N = a + b with a,b powerful.
    # Larger bound -> more powerful numbers -> more possible sums -> higher max count.
    bound = 2_000_000_000  # 2e9
    pw = _gen_powerful(bound)
    # pw is sorted; 1 is included (a=1,b=1 -> 1).

    # Count how many pairs (a,b) with a<=b sum to each N.
    # Use a dict: for each a in pw, for each b in pw with a<=b, sum=a+b.
    # But that's O(len(pw)^2) which is too big. Instead: for each a, b = N - a.
    # We want to find N with the most representations.
    # Approach: iterate over all pairs? Too big. 
    # Better: for each a in pw, for each b in pw with b>=a, add to counter[a+b].
    # len(pw) for bound 2e9: let's estimate. Powerful numbers up to X ~ c * X^(2/3).
    # For X=2e9, X^(2/3) ~ (2e9)^0.666 ~ (2)^0.666 * (1e9)^0.666 ~ 1.587 * 1e6 ~ 1.5e6.
    # So len(pw) ~ 1.5e6. Pairs ~ (1.5e6)^2 / 2 ~ 1e12. Way too big.

    # Need a smarter approach. 
    # Alternative: for a fixed N, count = #{a in pw : a <= N/2 and N-a in pw}.
    # We can't iterate all N. But we can iterate over candidate N values that are sums.
    # 
    # Key insight: the number of representations of N as sum of two powerful numbers
    # is at most the number of powerful numbers <= N/2, which for N up to 2*bound is ~ bound^(2/3).
    # To find N with many representations, we can:
    #   For each a in pw (as the smaller summand), and each b in pw with b>=a,
    #   record N=a+b. But that's O(n^2).
    #
    # Instead, let's use a different structure: 
    #   We want to maximize over N the count f(N) = #{a in pw : a<=N/2, N-a in pw}.
    #   Note f(N) = #{a in pw : a <= N/2} intersected with (N - pw).
    #   This is a convolution-like count.
    #
    # Practical approach given time: pick a moderate bound so that O(n^2) is feasible,
    # OR use a randomized/heuristic search over N.
    #
    # Let's use a smaller bound for full O(n^2) collision counting, but go large enough
    # that we find N with many representations.
    # For bound = 10^7, len(pw) ~ (1e7)^(2/3) ~ 46400. Pairs ~ 1e9. Still too big in Python.
    #
    # Alternative: use NumPy for vectorized counting.
    # For each a in pw, compute N = a + b for all b in pw with b>=a. That's still O(n^2).
    #
    # Better structural approach: 
    #   The maximum number of representations is achieved by N that are "rich" in powerful summands.
    #   We can search over N by considering N = a + b where a,b are powerful, and use a
    #   dictionary to count, but restrict to a subset.
    #
    # Let's try: generate pw up to bound, then for the top-K largest powerful numbers,
    # compute all pair sums among them and count. The largest powerful numbers have
    # many divisors, so sums involving them may have many representations.
    #
    # Actually, a cleaner approach: use the fact that if N is a multiple of a large
    # powerful number structure, it has many representations. 
    #
    # Let's just do: for bound up to ~5e6, do full O(n^2) with early exit on time.
    bound2 = 5_000_000
    if time.time() - t0 > DEADLINE - 30:
        # Not enough time for large search; fall back to small
        bound2 = 1_000_000
    pw2 = _gen_powerful(bound2)
    pw_set = set(pw2)
    n_pw = len(pw2)

    # Count collisions: for i in range(n_pw), for j in range(i, n_pw), sum = pw2[i]+pw2[j].
    # This is O(n_pw^2 / 2). For n_pw ~ 50000, that's ~1.25e9. Too slow in Python.
    #
    # Use NumPy to vectorize: for each a, compute a + pw2[pw2 >= a], count.
    # Still O(n^2) memory if done naively.
    #
    # Alternative: iterate over a, and for each a, the valid b are pw2 values >= a and <= 2*bound2 - a.
    # Count how many such b exist such that a+b = N for some N... but we want to count per N.
    #
    # Let's use a dictionary and iterate, but with a time guard.
    from collections import Counter
    sum_count = defaultdict(int)
    best_N = None
    best_count = 0
    deadline_inner = t0 + DEADLINE - 10

    # Iterate over pairs, but break if we exceed time.
    # To make this feasible, use a smaller bound.
    # For bound2 = 1e6, n_pw ~ 2150. Pairs ~ 2.3e6. Feasible.
    # For bound2 = 2e6, n_pw ~ 3400. Pairs ~ 5.8e6. Feasible.
    # For bound2 = 5e6, n_pw ~ 5000. Pairs ~ 1.25e7. Feasible in ~30s.
    #
    # Let's do it with a time guard.
    for i in range(n_pw):
        a = pw2[i]
        # b must be >= a and a+b <= 2*bound2
        # b in pw2[i:] and b <= 2*bound2 - a
        for j in range(i, n_pw):
            b = pw2[j]
            s = a + b
            if s > 2 * bound2:
                break
            sum_count[s] += 1
            if sum_count[s] > best_count:
                best_count = sum_count[s]
                best_N = s
        if time.time() > deadline_inner:
            break

    if best_N is None or best_count == 0:
        # Fallback: return a trivial valid result
        return {'N': 2, 'pairs': [[1, 1]]}

    # Verify best_N by exact factorization and collect pairs.
    pairs = []
    for a in pw2:
        if a > best_N // 2:
            break
        b = best_N - a
        if b >= a and b in pw_set and _is_powerful(a) and _is_powerful(b):
            pairs.append([a, b])

    if not pairs:
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
