"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _evaluate_set(A):
    """Return max(|A+A|, |A*A|) for a sorted list of distinct positive integers."""
    n = len(A)
    if n < 2:
        return 0
    A_arr = np.array(A, dtype=np.int64)
    # Sumset A+A
    sums = A_arr[:, None] + A_arr[None, :]
    sum_set = set(sums.ravel().tolist())
    # Product set A*A
    prods = A_arr[:, None] * A_arr[None, :]
    prod_set = set(prods.ravel().tolist())
    return max(len(sum_set), len(prod_set))


def _try_geometric(a, p, q, n=20, max_val=10000):
    """
    Generate n distinct positive integers from a geometric-like progression a*(p/q)^i,
    rounded to integers, ensuring distinctness and <= max_val.
    Returns the list if valid, else None.
    """
    if a <= 0 or p <= 0 or q <= 0:
        return None
    vals = []
    val = float(a)
    for i in range(n):
        iv = int(round(val))
        if iv <= 0 or iv > max_val:
            return None
        if iv in vals:
            return None
        vals.append(iv)
        val = val * p / q
    return vals


def construct():
    start_time = time.time()
    deadline = 50.0  # safety margin before 60s kill

    best_set = None
    best_score = float('inf')

    # Strategy: Search over geometric progressions with rational ratio p/q
    # and integer starting value a.
    # We want small product set => multiplicative collisions => geometric structure.
    # We want small sum set => compact range => small ratio or small a.

    # Grid search over small p, q and a
    # p/q should be close to 1 to keep the set compact, but not 1 (distinctness).
    # Also try p/q > 1 (growing) and p/q < 1 (shrinking, but we need positive ints <= 10000, so shrinking from large a).
    
    # Let's focus on p/q > 1 first, as it's more natural for positive integers.
    # Ratio r = p/q.
    # Try small p, q.
    
    max_pq = 15
    max_a = 1000
    
    # Pre-compute candidate ratios
    ratios = []
    for q in range(1, max_pq + 1):
        for p in range(1, max_pq + 1):
            if p == q:
                continue
            # Avoid duplicates (p/q same as p'/q')
            # We can just use a set of fractions or tuples
            ratios.append((p, q))
    
    # Sort by ratio value to explore near 1 first? Or just iterate.
    # Let's iterate p, q in a structured way.
    
    # We'll do a few passes with increasing budget.
    # Pass 1: Small p, q, small a
    # Pass 2: Larger p, q, larger a
    
    # To speed up, we can vectorize the generation of candidate sets?
    # Not easily, because distinctness and bounds depend on the specific sequence.
    # But we can prune early.
    
    # Let's just loop efficiently.
    
    # We can also try to optimize 'a' for given p, q.
    # For fixed p, q, the set is determined by a.
    # The values are roughly a * (p/q)^i.
    # To keep them <= 10000, a must be small enough.
    # To keep them distinct, the ratio must not be too close to 1, or a must be small.
    
    # Let's iterate a from 1 to some limit, and p, q from 1 to some limit.
    
    # Optimization:
    # For a given p, q, what is the max a?
    # The largest value is a * (p/q)^(n-1) if p > q, or a if p < q (but then they shrink).
    # If p > q, a * (p/q)^19 <= 10000 => a <= 10000 * (q/p)^19.
    # If p < q, a <= 10000. But distinctness is harder.
    
    # Let's focus on p > q (growing) and p < q (shrinking).
    
    # We'll use a deadline check.
    
    # To make this faster, we can pre-filter (p,q) pairs that are "good".
    # A good ratio is one that creates many collisions in products.
    # Geometric progression A = {a, ar, ar^2, ...}
    # Products: a^2 r^(i+j). Distinct products correspond to distinct i+j.
    # If r is an integer, say r=k, then products are a^2 k^(i+j).
    # i+j ranges from 0 to 2n-2. So there are 2n-1 distinct products.
    # This is very small!
    # Sums: a(1+k+...+k^i + 1+k+...+k^j).
    # If k is large, sums are distinct. If k is small, maybe collisions?
    # For k=2, sums are a(2^i-1 + 2^j-1 + 1) = a(2^i + 2^j - 1).
    # Distinct pairs (i,j) give distinct sums unless 2^i+2^j = 2^k+2^l.
    # This happens only if {i,j} = {k,l}. So |A+A| = n(n+1)/2 = 210 for n=20.
    # |A*A| = 39.
    # Max = 210.
    # Current best (1..20): |A+A| = 39, |A*A| is much larger.
    # Let's calculate |A*A| for 1..20.
    # Products i*j. Max 400.
    # Number of distinct products is around 200-300.
    # So geometric progression with integer ratio k=2 is likely much better.
    
    # Let's test A = {1, 2, 4, ..., 2^19}.
    # 2^19 = 524288 > 10000. So we can't use k=2 for n=20 with max 10000.
    # We need a * 2^19 <= 10000 => a=0, impossible.
    # So integer ratio k>=2 is too big for n=20 within 10000.
    
    # What about rational ratio p/q close to 1?
    # e.g. p=5, q=4. Ratio 1.25.
    # a * 1.25^19. 1.25^19 ~ 68.
    # a * 68 <= 10000 => a <= 147.
    # Let's try a=1, p=5, q=4.
    # vals: 1, 1.25, 1.56, ... rounded.
    # Distinctness might fail for small a.
    
    # Let's systematically search p, q, a.
    
    # We'll prioritize integer ratios first (q=1), then rational.
    
    # Pass 1: Integer ratios q=1, p=2..10
    for p in range(2, 11):
        q = 1
        max_a = int(10000 * (q / p) ** 19)
        if max_a < 1:
            continue
        for a in range(1, min(max_a + 1, 200)):
            A = _try_geometric(a, p, q)
            if A is None:
                continue
            score = _evaluate_set(A)
            if score < best_score:
                best_score = score
                best_set = A
            if time.time() - start_time > deadline:
                if best_set is not None:
                    return best_set
                return list(range(1, 21))
    
    # Pass 2: Rational ratios p/q, p > q
    for q in range(2, max_pq + 1):
        for p in range(q + 1, max_pq + 1):
            ratio = p / q
            # Estimate max a
            max_a = int(10000 * (1 / ratio) ** 19)
            if max_a < 1:
                continue
            # Limit a search
            search_limit = min(max_a, 500)
            for a in range(1, search_limit + 1):
                A = _try_geometric(a, p, q)
                if A is None:
                    continue
                score = _evaluate_set(A)
                if score < best_score:
                    best_score = score
                    best_set = A
                if time.time() - start_time > deadline:
                    if best_set is not None:
                        return best_set
                    return list(range(1, 21))
    
    # Pass 3: Rational ratios p/q, p < q (shrinking)
    # This is less likely to be good because products shrink, but sums might collide?
    # Actually, if p < q, the set is {a, a*p/q, ...}.
    # For these to be integers, a must be divisible by q^(n-1) roughly.
    # So a must be large.
    # Let's skip for now, or try a few.
    
    if best_set is None:
        return list(range(1, 21))
    
    return best_set
# EVOLVE-BLOCK-END
