"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
# Precompute smallest prime factor for numbers up to 256
def _build_spf(limit=256):
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

_SPF = _build_spf(256)

def _prime_factors(n):
    """Return set of prime factors of n using precomputed SPF."""
    factors = set()
    while n > 1:
        p = _SPF[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _is_pairwise_coprime_fast(values):
    """Check if all distinct values are pairwise coprime using prime factor sets."""
    distinct = list(set(values))
    n = len(distinct)
    factor_sets = [_prime_factors(v) for v in distinct]
    for i in range(n):
        for j in range(i + 1, n):
            if factor_sets[i] & factor_sets[j]:
                return False
    return True

def _score_pair_fast(A, B):
    """Return number of distinct pairwise-coprime sums, or -1 if not valid."""
    sums = [a + b for a in A for b in B]
    distinct = list(set(sums))
    if not _is_pairwise_coprime_fast(distinct):
        return -1
    return len(distinct)

def construct():
    deadline = time.time() + 85  # 85s out of 90s budget
    best_count = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Verify the seed is valid
    seed_sums = [a + b for a in [1, 7, 13] for b in [4, 10, 16]]
    seed_distinct = list(set(seed_sums))
    if _is_pairwise_coprime_fast(seed_distinct):
        best_count = len(seed_distinct)
    
    rng = random.Random(42)
    
    # Strategy 1: Try to find 9 distinct pairwise coprime sums directly.
    # For A={a1,a2,a3}, B={b1,b2,b3}, the 9 sums are a_i+b_j.
    # Key insight: if we can make all 9 sums distinct and pairwise coprime, we get score 9.
    
    # Approach: Choose 9 pairwise coprime target sums s_ij in [2,256],
    # then try to decompose them into a_i + b_j form.
    
    # For a 3x3 grid of sums:
    # s11 s12 s13
    # s21 s22 s23
    # s31 s32 s33
    # We need s_ij = a_i + b_j, which means:
    # s11 - s12 = b1 - b2 (constant across rows)
    # s11 - s21 = a1 - a2 (constant across columns)
    # So the sumset must have the structure: s_ij = c + a_i + b_j where the
    # row differences and column differences are consistent.
    
    # More precisely: s_ij = a_i + b_j implies:
    # s_11 + s_22 = s_12 + s_21 (for any 2x2 subgrid)
    # This is a necessary and sufficient condition for the 3x3 matrix to be
    # decomposable as outer sum.
    
    # So: pick 9 pairwise coprime numbers, check if they form a valid sumset,
    # and if so, extract A and B.
    
    # Let's try: pick a base value and offsets.
    # Let A = {x, x+p, x+q}, B = {y, y+r, y+s}
    # Sums: x+y, x+y+r, x+y+s, x+p+y, x+p+y+r, x+p+y+s, x+q+y, x+q+y+r, x+q+y+s
    # = (x+y) + {0, r, s, p, p+r, p+s, q, q+r, q+s}
    
    # So we need 9 pairwise coprime numbers of the form t + d_i where d_i are from
    # {0, r, s, p, p+r, p+s, q, q+r, q+s}
    
    # Let's search over p, q, r, s and t (base) to find 9 pairwise coprime values.
    
    # For efficiency, limit p, q, r, s to reasonable ranges.
    # t = x+y ranges from 2 to 256.
    
    # Precompute: for given p,q,r,s, the 9 offsets are fixed.
    # We need t + offset_i to be pairwise coprime for all 9 offsets.
    
    # Strategy: iterate over small p,q,r,s and for each, try multiple t values.
    
    max_pqrs = 50  # max difference
    found_9 = False
    
    for p in range(1, max_pqrs):
        for q in range(p + 1, max_pqrs):  # q > p for distinct A
            for r in range(1, max_pqrs):
                for s in range(r + 1, max_pqrs):  # s > r for distinct B
                    offsets = sorted(set([0, r, s, p, p+r, p+s, q, q+r, q+s]))
                    if len(offsets) != 9:
                        continue  # need all 9 distinct
                    # For each t, check if t+offsets are pairwise coprime
                    # t ranges such that all values in [2, 256]
                    min_t = max(2 - min(offsets), 1)
                    max_t = min(256 - max(offsets), 255)
                    if min_t > max_t:
                        continue
                    
                    # Sample t values rather than trying all
                    t_candidates = list(range(min_t, max_t + 1))
                    rng.shuffle(t_candidates)
                    for t in t_candidates[:500]:  # try up to 500 t values
                        values = [t + off for off in offsets]
                        if all(2 <= v <= 256 for v in values):
                            if _is_pairwise_coprime_fast(values):
                                # Found 9 pairwise coprime sums!
                                # Now extract A and B
                                # offsets correspond to: 
                                # 0->a1+b1, r->a1+b2, s->a1+b3, p->a2+b1, p+r->a2+b2, p+s->a2+b3, q->a3+b1, q+r->a3+b2, q+s->a3+b3
                                # a1 = x, b1 = y, t = x+y
                                # a2 = x+p, a3 = x+q
                                # b2 = y+r, b3 = y+s
                                # We need x, x+p, x+q, y, y+r, y+s all in [1,128]
                                # x >= 1, x+p <= 128, x+q <= 128 => x <= 128-q
                                # y >= 1, y+r <= 128, y+s <= 128 => y <= 128-s
                                # t = x+y, so x = t-y
                                # Constraints: 1 <= t-y <= 128-q, 1 <= y <= 128-s
                                # => max(1, t-(128-s)) <= y <= min(128-s, t-1)
                                y_min = max(1, t - (128 - s))
                                y_max = min(128 - s, t - 1)
                                if y_min <= y_max:
                                    y = y_min
                                    x = t - y
                                    if 1 <= x <= 128 - q and 1 <= y <= 128 - s:
                                        A = [x, x+p, x+q]
                                        B = [y, y+r, y+s]
                                        if all(1 <= v <= 128 for v in A) and all(1 <= v <= 128 for v in B):
                                            if len(set(A)) == 3 and len(set(B)) == 3:
                                                best_count = 9
                                                best_A = A
                                                best_B = B
                                                found_9 = True
                                                break
                    if found_9:
                        break
                    if time.time() > deadline:
                        return [best_A, best_B]
        if found_9 or time.time() > deadline:
            break
        if time.time() > deadline:
            break
    
    if found_9:
        return [best_A, best_B]
    
    # If we didn't find 9, do random search for best < 9
    for _ in range(200000):
        if time.time() > deadline:
            break
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        s = _score_pair_fast(A, B)
        if s > best_count:
            best_count = s
            best_A = A[:]
            best_B = B[:]
            if best_count >= 9:
                break
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
