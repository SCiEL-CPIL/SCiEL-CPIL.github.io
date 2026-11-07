"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 128
DEADLINE = 75.0  # safety margin before 90s kill

def _has_bad_triple(S):
    """Return True if S (set of ints) contains distinct a,b,c with a+b,a+c,b+c all in S."""
    L = sorted(S)
    n = len(L)
    Sset = set(L)
    for i in range(n):
        a = L[i]
        for j in range(i+1, n):
            b = L[j]
            if (a+b) in Sset:
                for k in range(j+1, n):
                    c = L[k]
                    if (a+c) in Sset and (b+c) in Sset:
                        return True
    return False

def _validate(S):
    """Return True if S is a valid set (no bad triple)."""
    return not _has_bad_triple(set(S))

def _fast_check_add(S, x):
    """Check if adding x to S (as a set) creates a bad triple. S is a set."""
    # A bad triple involving x: need distinct a,b in S such that:
    #   x is one of {a,b,c} and the three pair sums are all in S∪{x}
    # Cases:
    # 1) x is the largest (c): need a,b in S, a<b, and a+b, a+x, b+x all in S∪{x}
    #    Since a+x > x and b+x > x, they must be in S.
    #    Also a+b must be in S (since a+b < x? not necessarily, but if a+b == x it's ok since x is added)
    #    Actually a+b could be < x or > x. If a+b == x, that's fine. If a+b != x, it must be in S.
    # 2) x is the middle (b): need a,c in S, a<x<c, and a+x, a+c, x+c all in S∪{x}
    #    a+x > x, x+c > x, so they must be in S. a+c could be < x, = x, or > x.
    # 3) x is the smallest (a): need b,c in S, b<x? No, x is smallest so b,c > x.
    #    b+c, b+x, c+x all > x, so must be in S.
    #
    # More efficient: iterate over pairs in S and check if the third element and sums work.
    # But for speed, let's think differently.
    #
    # A triple {a,b,c} with a<b<c is bad iff a+b, a+c, b+c are all in the set.
    # If we add x, the new bad triples must include x.
    # Let S' = S ∪ {x}.
    # For each pair (p,q) in S with p<q:
    #   If x < p: triple {x,p,q}. Need x+p, x+q, p+q all in S'.
    #     x+p, x+q, p+q are all > x (since p,q > x), so they must be in S.
    #   If p < x < q: triple {p,x,q}. Need p+x, p+q, x+q all in S'.
    #     p+x, x+q > x, so must be in S. p+q could be anything.
    #   If p < q < x: triple {p,q,x}. Need p+q, p+x, q+x all in S'.
    #     p+x, q+x > x, so must be in S. p+q could be anything.
    #
    # So for each pair (p,q) in S:
    #   if x < p: check x+p in S, x+q in S, p+q in S
    #   if p < x < q: check p+x in S, x+q in S, p+q in S
    #   if p < q < x: check p+x in S, q+x in S, p+q in S
    #
    # This is O(|S|^2) per check which is too slow. Let's use a smarter approach.
    #
    # Alternative: precompute for S which pairs (a,b) have a+b in S.
    # When adding x, a new bad triple exists iff there exist distinct a,b,c in S∪{x}
    # such that all three pair sums are in S∪{x}.
    #
    # Let's just do the O(n^2) check but with early exit and it should be fine for n<=128.
    L = sorted(S | {x})
    n = len(L)
    Sset = set(L)
    for i in range(n):
        a = L[i]
        for j in range(i+1, n):
            b = L[j]
            s_ab = a + b
            if s_ab not in Sset:
                continue
            for k in range(j+1, n):
                c = L[k]
                if (a+c) in Sset and (b+c) in Sset:
                    return False  # bad triple found
    return True  # safe to add

def _greedy_complete(S, deadline):
    """Try to add elements from 1..128 not in S in a greedy manner."""
    S = set(S)
    candidates = [i for i in range(1, N+1) if i not in S]
    random.shuffle(candidates)
    for x in candidates:
        if time.time() > deadline:
            break
        if _fast_check_add(S, x):
            S.add(x)
    return S

def _two_layer_construct(seed):
    """Construct a set using two-layer approach."""
    rng = random.Random(seed)
    # Upper block U: sum-free. Use odds in a range.
    # Let U be all odd numbers in [lo, hi]
    # For U to be sum-free: sum of two odds is even, so if U contains only odds,
    # no pair sum of U is in U (since pair sums are even). Good.
    # But we need U ⊂ 1..128.
    #
    # Lower block L: weakly sum-free, with max(L) small enough that
    #   2*max(L) < min(U)  => all pair sums of L are < min(U)
    # Also need: no element of U equals a+b for a,b in L.
    #   Since a+b <= 2*max(L) < min(U), this is automatically satisfied.
    # Also need: no element of L equals a+b for a,b in L (weakly sum-free).
    #   L must be weakly sum-free: no a,b,c in L (not necessarily distinct? 
    #   Actually the triple must be distinct a,b,c, so a+b=a+c=b+c would require a=b=c, 
    #   but a,b,c are distinct, so we need: no distinct a,b,c in L with a+b,a+c,b+c in L.
    #   A simpler sufficient condition: L is sum-free (a+b not in L for any a,b in L, a<b).
    #   But weakly sum-free (a+b not in L for distinct a,b) is enough for our purpose
    #   because if a=b, then a+a=2a, and we'd need 2a in L, but then the triple would 
    #   need three distinct elements, so a,b,c distinct. If a+b is in L with a=b, 
    #   that doesn't directly create a bad triple unless combined with another element.
    #   To be safe, let's make L sum-free.
    #
    # Strategy: pick L as a set of small numbers that is sum-free.
    # Simple sum-free set: all numbers in (n/3, n/2] are sum-free? No.
    # Actually, a classic sum-free set: odds in a range, or numbers in (m/2, m].
    # Let's pick L to be a subset of [1, M] where M < min(U)/2.
    #
    # Let min(U) = lo. Then we need max(L) < lo/2.
    # Let's try lo = 70, so max(L) < 35. L ⊂ [1, 34].
    # U = odds in [70, 128] = {71, 73, 75, ..., 127}
    # |U| = (127-71)/2 + 1 = 29
    #
    # L: sum-free subset of [1, 34].
    # Max sum-free subset of [1,n] has size ceil(n/2).
    # For [1,34], max sum-free is 17 (e.g., all odds 1..33, or all in [18,34]).
    # Let's use L = odds in [1, 34] = {1,3,5,...,33}, |L|=17.
    # Check: is {1,3,5,...,33} sum-free? 1+3=4 not in L. 1+5=6 not in L. 3+5=8 not in L.
    # Yes, sum of two distinct odds is even, not in L. Good.
    # But wait, we need L to be weakly sum-free for the triple condition.
    # Since L is sum-free, it's certainly weakly sum-free.
    #
    # Total: |L| + |U| = 17 + 29 = 46. That's less than 66.
    #
    # Let's try a different split.
    # What if U is larger? Let U = odds in [50, 128] = {51,53,...,127}, |U| = 39.
    # Then max(L) < 25. L ⊂ [1,24].
    # Max sum-free in [1,24] is 12 (odds 1..23).
    # Total: 12 + 39 = 51. Still less.
    #
    # What if U is not just odds? What if U = [70, 128]?
    # Is [70,128] sum-free? 70+70=140>128, 70+71=141>128. 
    # Min pair sum = 70+71=141 > 128. So [70,128] is sum-free!
    # |U| = 128-70+1 = 59.
    # max(L) < 35. L ⊂ [1,34], max sum-free = 17.
    # Total: 17+59 = 76. Better!
    #
    # Can we do even better? U = [65,128]? Min pair sum = 65+66=131>128. Sum-free.
    # |U| = 64. max(L) < 32.5, so L ⊂ [1,32]. Max sum-free = 16.
    # Total: 16+64 = 80.
    #
    # U = [64,128]? Min pair sum = 64+65=129>128. Sum-free.
    # |U| = 65. max(L) < 32. L ⊂ [1,31]. Max sum-free = 16 (odds 1..31).
    # Total: 16+65 = 81.
    #
    # U = [63,128]? Min pair sum = 63+64=127 <= 128. Not sum-free (63+64=127 in U).
    #
    # So U=[64,128], L=odds in [1,31] gives 81.
    # Let's verify no cross issues:
    # - Pair sums of L: max = 31+29=60 < 64. Good, no L pair sum in U.
    # - Pair sums of U: min = 64+65=129 > 128. No U pair sum in 1..128. Good.
    # - L element = sum of two U elements? Min U pair sum = 129 > 31. No.
    # - U element = sum of two L elements? Max L pair sum = 60 < 64. No.
    # - L element = sum of L and U? Min L+U = 1+64=65 > 31. No.
    # - U element = sum of L and U? Min L+U = 65, max U = 128. 
    #   Could a U element equal l+u for l in L, u in U?
    #   l+u >= 1+64 = 65, l+u <= 31+128 = 159.
    #   We need l+u in U = [64,128]. So l+u in [65,128].
    #   This IS possible! e.g., l=1, u=64, l+u=65 in U.
    #   But does this create a bad triple? The triple would need three DISTINCT elements a,b,c.
    #   If we have l, u, l+u in the set, that's three elements. 
    #   The pair sums: l+u = l+u (in set), l+(l+u) = 2l+u, u+(l+u) = l+2u.
    #   For a bad triple, we need all three pair sums in the set.
    #   So we need 2l+u in set and l+2u in set.
    #   2l+u: if l=1,u=64, 2+64=66. Is 66 in set? 66 is in U=[64,128]. Yes!
    #   l+2u = 1+128=129. Not in 1..128. So NOT a bad triple.
    #   But we need to check all combinations.
    #
    # This is getting complex. Let's just build the set and validate, then greedily complete.
    
    # U = [64, 128]
    U = set(range(64, 129))
    # L = odds in [1, 31]
    L = set(range(1, 32, 2))
    S = L | U
    
    # Validate
    if not _validate(S):
        # Fallback: just U
        S = set(U)
    
    # Greedy complete
    S = _greedy_complete(S, deadline)
    return S

def _random_construct(seed, deadline):
    """Randomized construction with greedy completion."""
    rng = random.Random(seed)
    # Start with a random subset biased toward larger numbers
    S = set()
    for i in range(1, N+1):
        if rng.random() < 0.5:
            S.add(i)
    # Greedy complete
    S = _greedy_complete(S, deadline)
    return S

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    best = set(range(65, 129))  # fallback: [65,128], sum-free, size 64
    
    # Try two-layer construction
    try:
        S1 = _two_layer_construct(42)
        if _validate(S1) and len(S1) > len(best):
            best = S1
    except:
        pass
    
    # Try a few random constructions
    for seed in range(10):
        if time.time() > deadline - 5:
            break
        try:
            S = _random_construct(seed, deadline)
            if _validate(S) and len(S) > len(best):
                best = S
        except:
            pass
    
    # Final greedy pass on best
    best = _greedy_complete(best, deadline)
    
    if not _validate(best):
        # Emergency fallback
        best = set(range(65, 129))
    
    return list(best)
# EVOLVE-BLOCK-END
