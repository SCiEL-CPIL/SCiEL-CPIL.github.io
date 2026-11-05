"""Strong algorithm for bounded-lcm set maximization on [1,512]."""
import time
from math import gcd, isqrt

N = 512
DEADLINE = 85.0

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

    # Precompute LCM compatibility as bitmasks
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

    def greedy_augment(seed_set):
        S = set(seed_set)
        S_mask = set_to_mask(S)
        changed = True
        while changed:
            changed = False
            for v in range(1, N + 1):
                if v in S:
                    continue
                if is_compatible(v, S_mask):
                    S.add(v)
                    S_mask |= (1 << (v - 1))
                    changed = True
        return S

    # Find best initial solution
    best_S = set()
    best_size = 0

    # Try divisor sets of highly composite numbers
    M_sorted = sorted(range(1, N + 1), key=lambda m: len(_divisors_of(m, N)), reverse=True)
    tried_M = set()
    for M in M_sorted[:40]:
        if time.time() - start > DEADLINE - 20:
            break
        if M in tried_M:
            continue
        tried_M.add(M)
        d = _divisors_of(M, N)
        if not d:
            continue
        S1 = greedy_augment(d)
        if len(S1) > best_size:
            best_size = len(S1)
            best_S = S1
        S2 = greedy_by_degree(d)
        if len(S2) > best_size:
            best_size = len(S2)
            best_S = S2

    # Try from empty
    if time.time() - start < DEADLINE - 25:
        S3 = greedy_by_degree(set())
        if len(S3) > best_size:
            best_size = len(S3)
            best_S = S3

    # Local search: try to improve by removing one and adding more
    def local_search(S, max_iters=150):
        S = set(S)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and time.time() - start < DEADLINE - 5:
            improved = False
            iter_count += 1
            S_mask = set_to_mask(S)
            for v in sorted(S):
                S_no_v_mask = S_mask & ~(1 << (v - 1))
                # Find candidates compatible with S\{v}
                cand_mask = 0
                for u in range(1, N + 1):
                    if u != v and is_compatible(u, S_no_v_mask):
                        cand_mask |= (1 << (u - 1))
                
                # Try single additions
                r = cand_mask
                while r:
                    lb = r & (-r)
                    u = lb.bit_length()
                    if u != v:
                        S.add(u)
                        S.remove(v)
                        improved = True
                        break
                    r &= r - 1
                if improved:
                    break
                
                # Try double additions: remove v, add two
                cand_list = []
                r = cand_mask
                while r:
                    lb = r & (-r)
                    u = lb.bit_length()
                    if u != v:
                        cand_list.append(u)
                    r &= r - 1
                
                if len(cand_list) >= 2:
                    found = False
                    for i in range(len(cand_list)):
                        if time.time() - start > DEADLINE - 5:
                            break
                        for j in range(i + 1, len(cand_list)):
                            u1, u2 = cand_list[i], cand_list[j]
                            if _lcm(u1, u2) <= N:
                                S.add(u1)
                                S.add(u2)
                                S.remove(v)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if improved:
                        break
        return S

    # Apply local search
    if time.time() - start < DEADLINE - 10:
        improved_S = local_search(best_S, max_iters=150)
        if len(improved_S) > best_size:
            best_size = len(improved_S)
            best_S = improved_S

    # Validate
    if not _validate(best_S):
        best_S = set(range(1, 21))

    return list(best_S)
