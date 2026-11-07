"""Solve: select distinct integers in (210, 10210] so that 210*product is a perfect square,
minimizing max(selected) - 210."""
import time
from math import isqrt
from functools import lru_cache
import numpy as np

N = 210
MAX_VAL = 10210
MAX_SIZE = 100

# ---- Precompute primes, squarefree-parity vector for each integer up to MAX_VAL ----
_PRIME_LIST = []
def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n+1) if sieve[i]]

_primes = _primes_upto(MAX_VAL)

# Parity-vector for each number 0..MAX_VAL: dict prime->exponent mod 2 (only odd exponents kept)
# We only need this for 211..MAX_VAL and for N.
_pf = {}
for _p in _primes:
    for _m in range(_p, MAX_VAL+1, _p):
        # count exponent of _p in _m
        e = 0
        t = _m
        while t % _p == 0:
            t //= _p
            e += 1
        if e & 1:
            _pf.setdefault(_m, []).append(_p)

# Vector for each candidate value: frozenset of primes with odd exponent
def _vec(m):
    return frozenset(_pf.get(m, ()))

VEC = {}
for _m in range(N+1, MAX_VAL+1):
    VEC[_m] = _vec(_m)
TARGET = _vec(N)  # frozenset of primes with odd exponent in 210


# ---- Build candidate list ----
# candidates: list of (value, vec)
_CAND = [(m, VEC[m]) for m in range(N+1, MAX_VAL+1)]


# ---- Verification (arbitrary-precision integer sqrt) ----
def _is_square(x):
    r = isqrt(x)
    return r * r == x


def _valid(result):
    if not isinstance(result, (list, tuple)):
        return False
    if not (1 <= len(result) <= MAX_SIZE):
        return False
    vals = []
    for v in result:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if not (N < v <= MAX_VAL):
            return False
        vals.append(v)
    if len(set(vals)) != len(vals):
        return False
    prod = N
    for v in vals:
        prod *= v
    return _is_square(prod)


# ---- Parity check for a chosen set ----
def _parity_ok(chosen):
    acc = TARGET
    for v in chosen:
        acc = acc ^ VEC[v]
    return acc == frozenset()


# ---- Meet-in-the-middle with Schroeppel-Shamir style min-max search ----
# We'll do a direct 4-way MITM over the candidates restricted to (210, 210+M].
# Since |candidates| can be up to ~10000, 2^5000 is too big for a naive MITM.
# Instead, we use a smarter approach: we search over small subsets (size <= k) using
# a heuristic. Given the structure (210 = 2*3*5*7), many small solutions exist.

# Let's first try a fast, targeted search:
# 1) Look for a single a with sf(a) == sf(210) => a = 210*square. 
# 2) Look for pairs (a,b) with sf(a) ^ sf(b) == TARGET.
# 3) Then general small subsets.

# For the binary search on M, we'll:
#   - Build list L = [(value, vec) for value in 211..210+M]
#   - Try to find a subset whose XOR == TARGET, minimizing max.
#   - Use a heuristic: sort by value, use a greedy + local search.

# Actually, a cleaner approach: since we want to MINIMIZE max(selected), and the set
# must satisfy the parity condition, we can do:
#   For M from 1 upward:
#     candidates = [v for v in 211..210+M]
#     check if there's a subset with XOR == TARGET
#     if yes, return it
# But checking subset XOR in a 10000-element set is hard in general.

# However, the parity space is limited. The vector is over primes. The number of
# distinct parity vectors among 211..10210 is bounded by 2^(#primes <= 10210) but in
# practice much smaller.

# Key observation: we only need ANY subset, not necessarily minimal-size.
# A subset can be of size 1 (if v has the same parity as 210), size 2, etc.

# Let's use a practical approach:
# 1) For each M, build the candidate list.
# 2) Build a dict: parity_vec -> list of values having that parity (within 211..210+M).
# 3) Check size-1: is TARGET in the dict?
# 4) Check size-2: for each pair of vectors (v1, v2) with v1 ^ v2 == TARGET, check if
#    there exist distinct values. Use the dict.
# 5) Check size-3: v1 ^ v2 ^ v3 == TARGET.
# 6) For larger sizes, use a randomized greedy.

# Actually, let me think about the structure more carefully.
# 210 = 2*3*5*7. So TARGET = {2,3,5,7}.
# A number v has parity {2,3,5,7} iff v = 2*a^2 * 3*b^2 * 5*c^2 * 7*d^2 for some a,b,c,d,
# i.e., v is 210 times a perfect square. The smallest such v > 210 is 210*4 = 840.
# So the size-1 solution gives max = 840, score = 630.

# For size-2: we need sf(a) ^ sf(b) = {2,3,5,7}.
# E.g., sf(a) = {2}, sf(b) = {3,5,7}: a = 2*square, b = 105*square.
# Smallest a > 210 with sf(a)={2}: a = 2*11^2 = 242. sf(242)={2}? 242=2*121=2*11^2, yes.
# Smallest b > 210 with sf(b)={3,5,7}: b = 105*4=420. sf(420)=sf(2^2*3*5*7)={3,5,7}, yes.
# max = max(242, 420) = 420. Score = 210. Better!

# Even better: sf(a)={2,3}, sf(b)={5,7}: a=6*square, b=35*square.
# Smallest a>210: 6*20^2=2400? No, 6*6^2=216>210, sf(216)=sf(2^3*3^3)={2,3}, yes! a=216.
# Smallest b>210: 35*8^2=2240? 35*2^2=140<210, 35*3^2=315>210, sf(315)=sf(3^2*5*7)={5,7}, yes! b=315.
# max = max(216,315)=315. Score = 105.

# sf(a)={2,5}, sf(b)={3,7}: a=10*square, b=21*square.
# a: 10*5^2=250, sf(250)=sf(2*5^3)={2,5}, yes. b: 21*4^2=336, sf(336)=sf(2^4*3*7)={3,7}, yes.
# max=336. Score=126. Worse.

# sf(a)={2,7}, sf(b)={3,5}: a=14*square, b=15*square.
# a: 14*4^2=224, sf(224)=sf(2^5*7)={2,7}, yes. b: 15*4^2=240, sf(240)=sf(2^4*3*5)={3,5}, yes.
# max=240. Score=30! Even better.

# Can we do even better? Let's check size-1 and size-2 more carefully.
# For size-2, we want to minimize max(a,b) where sf(a)^sf(b)=TARGET.
# The 6 partitions of {2,3,5,7} into two subsets (up to symmetry):
#   {2}|{3,5,7}, {3}|{2,5,7}, {5}|{2,3,7}, {7}|{2,3,5},
#   {2,3}|{5,7}, {2,5}|{3,7}, {2,7}|{3,5}
# For each partition (S, T), we need smallest a>210 with sf(a)=S and smallest b>210 with sf(b)=T.

# Let me compute these systematically.


def _smallest_with_parity(target_set, lo, hi):
    """Find the smallest integer in (lo, hi] whose squarefree-parity equals target_set."""
    for v in range(lo+1, hi+1):
        if VEC[v] == target_set:
            return v
    return None


# Precompute: for each partition of TARGET into (S, T) (unordered), find min max(a,b)
# where sf(a)=S, sf(b)=T, both in (210, MAX_VAL].

from itertools import combinations

TARGET_PRIMES = sorted(TARGET)  # [2, 3, 5, 7]
best_score = MAX_VAL - N  # fallback
best_result = [4 * N]

# Size-1
v1 = _smallest_with_parity(TARGET, N, MAX_VAL)
if v1 is not None:
    s = v1 - N
    if s < best_score:
        best_score = s
        best_result = [v1]

# Size-2: all ways to split TARGET into (S, T) with S nonempty, S != T (avoid double count)
seen_splits = set()
for r in range(1, len(TARGET_PRIMES)//2 + 1):
    for S in combinations(TARGET_PRIMES, r):
        T = tuple(p for p in TARGET_PRIMES if p not in S)
        pair = frozenset([frozenset(S), frozenset(T)])
        if pair in seen_splits:
            continue
        seen_splits.add(pair)
        fs, ft = frozenset(S), frozenset(T)
        a = _smallest_with_parity(fs, N, MAX_VAL)
        b = _smallest_with_parity(ft, N, MAX_VAL)
        if a is not None and b is not None:
            m = max(a, b) - N
            if m < best_score:
                best_score = m
                best_result = [a, b]

# Now try to improve with a more systematic search for small M.
# Binary search on M: for each M, check if a valid subset exists in (210, 210+M].
# We'll use a randomized + greedy approach for the subset XOR problem.

def _find_subset_in_range(M, max_iter=500000):
    """Try to find a subset of (210, 210+M] whose parity XOR equals TARGET.
    Returns a list of values or None."""
    cand = list(range(N+1, N+M+1))
    if not cand:
        return None
    
    # Quick check: size-1
    for v in cand:
        if VEC[v] == TARGET:
            return [v]
    
    # Build parity->values map
    from collections import defaultdict
    pmap = defaultdict(list)
    for v in cand:
        pmap[VEC[v]].append(v)
    
    # Size-2
    for v in cand:
        need = VEC[v] ^ TARGET
        if need in pmap:
            for w in pmap[need]:
                if w != v:
                    return [v, w]
    
    # Size-3: v1 ^ v2 ^ v3 == TARGET
    # v1 ^ v2 == TARGET ^ v3
    # For each v3, check if there exist v1, v2 with v1^v2 = TARGET^v3
    # This is O(n^2) per v3 which is too slow. Instead, use the pmap.
    # For a pair (v1, v2), v1^v2 = need. We need need = TARGET ^ v3 for some v3.
    # So: is there a pair (v1, v2) and a v3 such that v1^v2^v3 = TARGET?
    # Equivalently: v1^v2 = TARGET ^ v3.
    # For each v3, need = TARGET ^ VEC[v3]. Check if need is a sum of two parities present.
    # We can precompute all pairwise XORs? That's O(n^2) which is too much for large M.
    # Instead, for each v3, iterate over v1 and check if TARGET^VEC[v3]^VEC[v1] is in pmap.
    # That's O(n^2) per v3... too slow.
    
    # Alternative: use a randomized approach.
    # Randomly pick elements and maintain XOR, backtracking.
    import random
    random.seed(42)
    
    # Try a greedy + local search
    # Start with empty set, greedily add elements that reduce "distance" to TARGET.
    # Then try swapping to improve.
    
    # Actually, let's just try random subsets of small sizes.
    for trial in range(max_iter):
        # Random size 3-5
        sz = random.choice([3, 3, 3, 4, 4, 5])
        if sz > len(cand):
            sz = len(cand)
        subset = random.sample(cand, sz)
        acc = TARGET
        for v in subset:
            acc = acc ^ VEC[v]
        if acc == frozenset():
            return subset
    
    return None


# Let's do a more efficient search using the structure.
# Since we already found score=30 from {224, 240}, let's verify and try to improve.

def construct():
    global best_score, best_result
    
    start_time = time.time()
    deadline = 75.0  # safety margin
    
    # We already have a candidate from the precomputed search above.
    # Let's verify it.
    if not _valid(best_result):
        best_result = [4 * N]
        best_score = 4 * N - N
    
    # Now try to improve by searching for smaller M.
    # Binary search: find the minimum M such that a valid subset exists in (210, 210+M].
    # Lower bound: 1 (if 211 works, which it doesn't), upper bound: best_score.
    
    lo = 1
    hi = best_score
    best_found = best_result
    
    # First, let's try a direct search from small M upward, but only if the range is small.
    # For each M, we need to check if a subset XOR == TARGET exists in (210, 210+M].
    
    # Efficient approach: precompute all subset XORs for small M using DP.
    # For M up to ~500, the candidate set is ~500 elements. 
    # We can use a dictionary to track reachable XORs with the corresponding subset.
    # But the XOR space can be large.
    
    # Let's use a different approach: for each M, use a randomized search + greedy.
    
    # First, let's do a systematic search for M from 1 to min(best_score, 200).
    # For each M, try to find a subset.
    
    for M in range(1, min(best_score + 1, 200)):
        if time.time() - start_time > deadline:
            break
        result = _find_subset_in_range(M, max_iter=200000)
        if result is not None:
            if _valid(result):
                best_found = result
                best_score = M
                break  # can't do better than M
    
    # Try a few more random searches with slightly larger M if we have time.
    if time.time() - start_time < deadline - 5:
        # Try to improve by searching in a window around the current best.
        for M in range(max(1, best_score - 20), best_score):
            if time.time() - start_time > deadline:
                break
            result = _find_subset_in_range(M, max_iter=500000)
            if result is not None and _valid(result):
                best_found = result
                best_score = M
    
    return list(best_found)


# Pre-run the initial search to set best_score and best_result
# (This runs at module load time)
if not _valid(best_result):
    best_result = [4 * N]
    best_score = 4 * N - N

# Also, let's verify the hardcoded good solution
_test = [224, 240]
if _valid(_test):
    s = max(_test) - N
    if s < best_score:
        best_score = s
        best_result = _test


# Let's also try a few more known-good small solutions:
# Check {216, 315}: sf(216)={2,3}, sf(315)={5,7}, XOR={2,3,5,7}=TARGET. max=315, score=105.
# Check {224, 240}: sf(224)={2,7}, sf(240)={3,5}, XOR={2,3,5,7}=TARGET. max=240, score=30.
# Already checked.

# Can we do better than 30? Let's check M < 30.
# For M < 30, candidates are 211..239.
# We need a subset whose XOR = {2,3,5,7}.
# Let's just let the binary search handle it.


# Final construct
def construct():
    global best_score, best_result
    
    start_time = time.time()
    deadline = 75.0
    
    # Start with our best known
    current_best = list(best_result)
    current_score = best_score
    
    if not _valid(current_best):
        current_best = [4 * N]
        current_score = 4 * N - N
    
    # Systematic search for M from 1 to min(current_score, 300)
    for M in range(1, min(current_score + 1, 300)):
        if time.time() - start_time > deadline:
            break
        # Quick check: build candidate list
        cand = list(range(N + 1, N + M + 1))
        if not cand:
            continue
        
        # Size-1
        found = None
        for v in cand:
            if VEC[v] == TARGET:
                found = [v]
                break
        
        # Size-2
        if found is None:
            from collections import defaultdict
            pmap = defaultdict(list)
            for v in cand:
                pmap[VEC[v]].append(v)
            for v in cand:
                need = VEC[v] ^ TARGET
                if need in pmap:
                    for w in pmap[need]:
                        if w != v:
                            found = [v, w]
                            break
                if found:
                    break
        
        # Size-3 (randomized)
        if found is None and len(cand) >= 3:
            import random
            rng = random.Random(M)  # deterministic per M
            for trial in range(50000):
                if time.time() - start_time > deadline:
                    break
                sz = 3
                if sz > len(cand):
                    sz = len(cand)
                subset = rng.sample(cand, sz)
                acc = TARGET
                for v in subset:
                    acc = acc ^ VEC[v]
                if acc == frozenset():
                    found = subset
                    break
        
        if found is not None:
            if _valid(found):
                current_best = found
                current_score = M
                break  # found the minimum M
    
    # If we have time, try to verify and potentially improve with a broader search.
    if time.time() - start_time < deadline - 10 and current_score > 1:
        # Try random searches for M slightly below current_score
        import random
        rng = random.Random(12345)
        for _ in range(50):
            if time.time() - start_time > deadline:
                break
            M = rng.randint(1, max(1, current_score - 1))
            cand = list(range(N + 1, N + M + 1))
            if len(cand) < 2:
                continue
            # Try random subsets of size 2-5
            for trial in range(10000):
                sz = rng.randint(2, min(5, len(cand)))
                subset = rng.sample(cand, sz)
                acc = TARGET
                for v in subset:
                    acc = acc ^ VEC[v]
                if acc == frozenset():
                    if _valid(subset):
                        current_best = subset
                        current_score = M
                    break
            if current_score == M:
                break
    
    return list(current_best)


# Update global best with the known good solution
_test2 = [224, 240]
if _valid(_test2):
    s = max(_test2) - N
    if s < best_score:
        best_score = s
        best_result = _test2
