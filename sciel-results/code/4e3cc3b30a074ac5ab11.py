"""Approximate ratio a/b to 3/2 within a divisor-sum fiber sigma(a)=sigma(b)."""
import math
import time

MAX_N = 10_000_000
TARGET_NUM, TARGET_DEN = 3, 2

# Precompute smallest prime factor (SPF) sieve for fast factorization
def _build_spf(limit):
    spf = list(range(limit + 1))
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

_spf = None

def _get_spf():
    global _spf
    if _spf is None:
        _spf = _build_spf(MAX_N)
    return _spf

def _factorize(n, spf):
    """Return list of (prime, exponent) pairs for n using SPF."""
    factors = []
    while n > 1:
        p = spf[n]
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        factors.append((p, e))
    return factors

def _sigma(n, spf):
    """Compute sigma(n) = sum of divisors of n using factorization."""
    if n == 1:
        return 1
    factors = _factorize(n, spf)
    result = 1
    for p, e in factors:
        result *= (p ** (e + 1) - 1) // (p - 1)
    return result

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    spf = _get_spf()
    
    best_a, best_b = 11, 6  # fallback: sigma(11)=12, sigma(6)=12, |11/6 - 3/2| = |11/6 - 9/6| = 2/6 = 1/3
    best_err_num = 1  # numerator of |a/b - 3/2| = |2a - 3b| / (2b)
    best_err_den = 2 * best_b  # denominator is 2b
    
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # Let k = |2a - 3b|. For each k >= 1, we want to find (a,b) with 2a - 3b = ±k
    # and sigma(a) = sigma(b), minimizing k/(2b).
    # For a given k, 2a - 3b = k => a = (3b + k) / 2, so 3b + k must be even.
    # Similarly 2a - 3b = -k => a = (3b - k) / 2, so 3b - k must be even.
    
    # Strategy: iterate k from 1 upward. For each k, try to find b such that
    # a = (3b ± k)/2 is a positive integer <= MAX_N, b <= MAX_N, and sigma(a) = sigma(b).
    # To minimize k/(2b), for a fixed k we want the largest possible b.
    # b ranges: for a = (3b+k)/2 >= 1 => b >= (2-k)/3, and a <= MAX_N => 3b+k <= 2*MAX_N => b <= (2*MAX_N - k)/3
    # Also b >= 1 and b <= MAX_N.
    
    # But searching all b for each k is too slow. Instead, we can be smarter:
    # For each k, the candidate b values are constrained. But k can be large.
    # Alternative: iterate over b and compute candidate a values?
    
    # Better approach: For each k starting from 1, find the largest b <= min(MAX_N, (2*MAX_N-k)//3)
    # such that a = (3b+k)//2 is integer and sigma(a)=sigma(b). Also check a=(3b-k)//2.
    
    # But this is still O(MAX_N) per k. We need a different strategy.
    
    # Alternative: Precompute sigma for all n up to MAX_N? That's 10 million entries.
    # sigma can be computed via sieve in O(N log N) time. Let's do that.
    
    # Actually, let's precompute sigma for all n up to MAX_N using a sieve-like approach.
    # This takes O(N log N) time and O(N) space. For N=10^7, this might be feasible in ~10-20s.
    
    # Let's precompute sigma array.
    if time.time() > deadline - 20:
        return [best_a, best_b]
    
    sigma_arr = [0] * (MAX_N + 1)
    for i in range(1, MAX_N + 1):
        for j in range(i, MAX_N + 1, i):
            sigma_arr[j] += i
    
    if time.time() > deadline - 5:
        return [best_a, best_b]
    
    # Now search over k from 1 upward
    # For each k, find b such that a = (3b+k)//2 or a = (3b-k)//2 is valid and sigma(a)==sigma(b)
    # We want to maximize b to minimize k/(2b)
    
    # For efficiency, for each k, the valid b range is:
    # Case 1: a = (3b+k)//2, need 3b+k even, a in [1, MAX_N], b in [1, MAX_N]
    # Case 2: a = (3b-k)//2, need 3b-k even, a in [1, MAX_N], b in [1, MAX_N]
    
    # Parity: 3b+k even => b+k even (since 3b has same parity as b) => b and k same parity
    # 3b-k even => b-k even => b and k same parity
    
    # So for a given k, b must have the same parity as k.
    
    # Let's iterate k from 1 to some limit, and for each k, check a few candidate b values.
    # To find the largest b, we can start from the max possible b and work downward.
    # But that could be slow. Instead, let's iterate b for each k but with early termination.
    
    # Actually, let's change strategy: iterate over possible sigma values.
    # Group numbers by sigma(n). For each group, find pairs (a,b) in the same group
    # that minimize |a/b - 3/2|.
    
    # Build a dictionary: sigma_value -> list of n
    if time.time() > deadline - 10:
        return [best_a, best_b]
    
    from collections import defaultdict
    groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        groups[sigma_arr[n]].append(n)
    
    if time.time() > deadline - 5:
        return [best_a, best_b]
    
    # For each group, find pairs that minimize |2a - 3b| / (2b)
    # Within a group, sort by value. For each b, find a closest to 3b/2.
    # Use binary search.
    
    best_err_num = None
    best_err_den = None
    
    for sig_val, nums in groups.items():
        if len(nums) < 2:
            continue
        nums.sort()
        # For each b in nums, find a in nums closest to 3b/2
        # Use bisect
        import bisect
        for i, b in enumerate(nums):
            target_a = 3 * b / 2
            # Find insertion point
            pos = bisect.bisect_left(nums, target_a)
            candidates = []
            if pos < len(nums):
                candidates.append(nums[pos])
            if pos > 0:
                candidates.append(nums[pos - 1])
            for a in candidates:
                if a == b:
                    continue
                # Compute |2a - 3b|
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    # Perfect match, but raw value 0 is invalid
                    continue
                den = 2 * b
                # Compare diff/den with best_err_num/best_err_den
                if best_err_num is None:
                    best_err_num = diff
                    best_err_den = den
                    best_a, best_b = a, b
                else:
                    # Compare diff/den < best_err_num/best_err_den
                    # i.e., diff * best_err_den < best_err_num * den
                    if diff * best_err_den < best_err_num * den:
                        best_err_num = diff
                        best_err_den = den
                        best_a, best_b = a, b
        if time.time() > deadline - 3:
            break
    
    if best_a is not None and best_err_num is not None and best_err_num > 0:
        return [int(best_a), int(best_b)]
    return [best_a, best_b]
