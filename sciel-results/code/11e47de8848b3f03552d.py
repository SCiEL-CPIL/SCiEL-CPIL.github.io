"""Strong algorithm for bounded-lcm set maximization on [1,512]."""
import time
from math import gcd, isqrt

N = 512
DEADLINE = 85.0  # safety margin before 90s kill

def _lcm(a, b):
    return a * b // gcd(a, b)

def _validate(s):
    s = set(s)
    if not s:
        return False
    for x in s:
        if x < 1 or x > N:
            return False
    lst = sorted(s)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if _lcm(lst[i], lst[j]) > N:
                return False
    return True

def _divisors_of(n, limit=N):
    res = set()
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            if i <= limit:
                res.add(i)
            j = n // i
            if j != i and j <= limit:
                res.add(j)
    return res

def construct():
    start = time.time()

    # --- Precompute LCM compatibility matrix using bitmasks ---
    # compat_mask[v]: bitmask where bit u is set iff u is in [1,512] and lcm(v,u) <= 512
    compat_mask = [0] * (N + 1)
    for v in range(1, N + 1):
        mask = 0
        for u in range(1, N + 1):
            if _lcm(v, u) <= N:
                mask |= (1 << (u - 1))
        compat_mask[v] = mask

    def is_compatible(v, S_mask):
        return (S_mask & ~compat_mask[v]) == 0

    def set_to_mask(s):
        m = 0
        for v in s:
            m |= (1 << (v - 1))
        return m

    def greedy_by_degree(seed_set):
        S = set(seed_set)
        S_mask = set_to_mask(S)
        remaining_mask = 0
        for v in range(1, N + 1):
            if v not in S and is_compatible(v, S_mask):
                remaining_mask |= (1 << (v - 1))
        
        while remaining_mask:
            best_v = None
            best_deg = -1
            r = remaining_mask
            while r:
                lb = r & (-r)
                v = lb.bit_length()
                deg = (compat_mask[v] & remaining_mask).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                r &= r - 1
            
            if best_v is None:
                break
            S.add(best_v)
            S_mask |= (1 << (best_v - 1))
            remaining_mask &= compat_mask[best_v]
            remaining_mask &= ~(1 << (best_v - 1))
        
        return S

    def local_search_polish(S):
        """Try pairwise swaps: remove x, add y if y is compatible with S\{x} and |S| doesn't decrease."""
        improved = True
        while improved:
            improved = False
            S_list = sorted(S)
            for i, x in enumerate(S_list):
                for y in range(1, N + 1):
                    if y in S:
                        continue
                    S_minus_x = S - {x}
                    if is_compatible(y, set_to_mask(S_minus_x)):
                        # Check if y is "better" than x in some sense, or just try the swap
                        new_S = S_minus_x | {y}
                        # Only accept if it allows adding more later, or if y > x (prefer larger numbers?)
                        # For now, accept any swap that maintains validity
                        S = new_S
                        improved = True
                        break
                if improved:
                    break
        return S

    # --- Find good starting seeds using divisor structure ---
    # Precompute divisor sets for all M
    M_sorted = sorted(range(1, N + 1), key=lambda m: len(_divisors_of(m, N)), reverse=True)
    
    best_S = set()
    best_size = 0
    
    # Try top M values as seeds with degree-based greedy
    for M in M_sorted[:40]:
        if time.time() - start > DEADLINE - 20:
            break
        d = _divisors_of(M, N)
        if not d:
            continue
        
        S2 = greedy_by_degree(d)
        if len(S2) > best_size:
            best_size = len(S2)
            best_S = S2
    
    # Also try from empty set
    if time.time() - start < DEADLINE - 25:
        S3 = greedy_by_degree(set())
        if len(S3) > best_size:
            best_size = len(S3)
            best_S = S3
    
    # Try increasing-order greedy
    if time.time() - start < DEADLINE - 20:
        S = set()
        S_mask = 0
        for v in range(1, N + 1):
            if is_compatible(v, S_mask):
                S.add(v)
                S_mask |= (1 << (v - 1))
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Local search polish: try to improve via pairwise swaps
    if time.time() - start < DEADLINE - 5:
        polished_S = local_search_polish(best_S)
        if len(polished_S) > best_size:
            best_size = len(polished_S)
            best_S = polished_S
    
    # Final polish: try to add one more element
    if time.time() - start < DEADLINE - 2:
        S_mask = set_to_mask(best_S)
        for v in range(1, N + 1):
            if v not in best_S and is_compatible(v, S_mask):
                best_S.add(v)
                break
    
    # Validate
    if not _validate(best_S):
        best_S = set(range(1, 21))
    
    return list(best_S)
