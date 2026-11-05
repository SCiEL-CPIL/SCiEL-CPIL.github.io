"""Strong algorithm for bounded-lcm set maximization on [1,512]."""
import time
from math import gcd, isqrt
from collections import defaultdict

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

    # Precompute divisor relations for down-set enforcement
    # divisors[v] = set of proper divisors of v in [1, N]
    divisors = [set() for _ in range(N + 1)]
    for v in range(1, N + 1):
        for d in range(1, isqrt(v) + 1):
            if v % d == 0:
                if d != v:
                    divisors[v].add(d)
                j = v // d
                if j != v and j != d:
                    divisors[v].add(j)

    def is_compatible(v, S_mask):
        return (S_mask & ~compat_mask[v]) == 0

    def set_to_mask(s):
        m = 0
        for v in s:
            m |= (1 << (v - 1))
        return m

    def mask_to_set(m):
        s = set()
        r = m
        while r:
            lb = r & (-r)
            s.add(lb.bit_length())
            r &= r - 1
        return s

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

    def greedy_fill(seed_set):
        """Fill by adding elements in order of smallest LCM with current set."""
        S = set(seed_set)
        S_mask = set_to_mask(S)
        remaining_mask = 0
        for v in range(1, N + 1):
            if v not in S and is_compatible(v, S_mask):
                remaining_mask |= (1 << (v - 1))
        
        while remaining_mask:
            best_v = None
            best_score = float('inf')
            r = remaining_mask
            while r:
                lb = r & (-r)
                v = lb.bit_length()
                max_lcm = 0
                for u in S:
                    l = _lcm(v, u)
                    if l > max_lcm:
                        max_lcm = l
                if max_lcm < best_score:
                    best_score = max_lcm
                    best_v = v
                r &= r - 1
            
            if best_v is None:
                break
            S.add(best_v)
            S_mask |= (1 << (best_v - 1))
            remaining_mask &= compat_mask[best_v]
            remaining_mask &= ~(1 << (best_v - 1))
        
        return S

    # Down-set closure: if v is in S, ensure all divisors of v are also in S
    def downset_closure(S):
        S = set(S)
        changed = True
        while changed:
            changed = False
            for v in list(S):
                for d in divisors[v]:
                    if d not in S:
                        S.add(d)
                        changed = True
        return S

    # Find best initial solution
    best_S = set()
    best_size = 0

    # Try divisor sets of highly composite numbers
    M_sorted = sorted(range(1, N + 1), key=lambda m: len(_divisors_of(m, N)), reverse=True)
    tried_M = set()
    for M in M_sorted[:80]:
        if time.time() - start > DEADLINE - 25:
            break
        if M in tried_M:
            continue
        tried_M.add(M)
        d = _divisors_of(M, N)
        if not d:
            continue
        # Apply downset closure to ensure structural optimality
        d_closed = downset_closure(d)
        S1 = greedy_augment(d_closed)
        if len(S1) > best_size:
            best_size = len(S1)
            best_S = S1
        S2 = greedy_by_degree(d_closed)
        if len(S2) > best_size:
            best_size = len(S2)
            best_S = S2
        S3 = greedy_fill(d_closed)
        if len(S3) > best_size:
            best_size = len(S3)
            best_S = S3

    # Try from empty
    if time.time() - start < DEADLINE - 30:
        S4 = greedy_by_degree(set())
        if len(S4) > best_size:
            best_size = len(S4)
            best_S = S4
        S5 = greedy_fill(set())
        if len(S5) > best_size:
            best_size = len(S5)
            best_S = S5

    # Structured local search using down-set property
    def local_search_downset(S, max_iters=300):
        """Local search that maintains down-set property.
        
        Key insight: In the optimal solution, if a number is included,
        all its divisors should also be included (down-set in divisibility poset).
        This dramatically reduces the search space.
        """
        S = set(S)
        S = downset_closure(S)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters and time.time() - start < DEADLINE - 5:
            improved = False
            iter_count += 1
            S_mask = set_to_mask(S)
            
            # Try removing one element and adding compatible ones
            # Focus on "expensive" elements (high value, few divisors benefit)
            candidates_to_remove = sorted(S, key=lambda x: (-x, len(divisors[x])))
            
            for v in candidates_to_remove[:30]:  # Limit to top 30 most expensive
                if time.time() - start > DEADLINE - 5:
                    break
                    
                S_no_v = S - {v}
                S_no_v_mask = set_to_mask(S_no_v)
                
                # Find all candidates compatible with S\{v}
                cand_mask = 0
                for u in range(1, N + 1):
                    if u not in S_no_v and is_compatible(u, S_no_v_mask):
                        cand_mask |= (1 << (u - 1))
                
                cand_set = mask_to_set(cand_mask)
                
                # Try adding 1, 2, or 3 elements
                for k in [3, 2, 1]:
                    if k == 1:
                        for u in cand_set:
                            new_S = S_no_v | {u}
                            if len(new_S) > len(S):
                                S = new_S
                                improved = True
                                break
                    elif k == 2:
                        cand_list = list(cand_set)
                        for i in range(len(cand_list)):
                            if time.time() - start > DEADLINE - 5:
                                break
                            u1 = cand_list[i]
                            for j in range(i + 1, len(cand_list)):
                                u2 = cand_list[j]
                                if _lcm(u1, u2) <= N:
                                    new_S = S_no_v | {u1, u2}
                                    if len(new_S) > len(S):
                                        S = new_S
                                        improved = True
                                        break
                            if improved:
                                break
                    elif k == 3:
                        cand_list = list(cand_set)
                        if len(cand_list) < 3:
                            continue
                        for i in range(len(cand_list)):
                            if time.time() - start > DEADLINE - 5:
                                break
                            u1 = cand_list[i]
                            for j in range(i + 1, len(cand_list)):
                                u2 = cand_list[j]
                                if _lcm(u1, u2) > N:
                                    continue
                                for l in range(j + 1, len(cand_list)):
                                    u3 = cand_list[l]
                                    if _lcm(u1, u3) <= N and _lcm(u2, u3) <= N:
                                        new_S = S_no_v | {u1, u2, u3}
                                        if len(new_S) > len(S):
                                            S = new_S
                                            improved = True
                                            break
                                if improved:
                                    break
                            if improved:
                                break
                    if improved:
                        break
                
                if improved:
                    # Re-apply downset closure
                    S = downset_closure(S)
                    break
        
        return S

    # Apply structured local search
    if time.time() - start < DEADLINE - 10:
        improved_S = local_search_downset(best_S, max_iters=300)
        if len(improved_S) > best_size:
            best_size = len(improved_S)
            best_S = improved_S

    # Final polish: try to add any missing compatible elements
    if time.time() - start < DEADLINE - 3:
        final_S = greedy_augment(best_S)
        if len(final_S) > best_size:
            best_size = len(final_S)
            best_S = final_S

    # Validate
    if not _validate(best_S):
        best_S = set(range(1, 21))

    return list(best_S)
