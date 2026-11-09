import time
from collections import defaultdict

N = 256
DEADLINE = time.time() + 85  # 5s safety margin before 90s kill


def _validate(S):
    """Return True iff S satisfies: for all a<b<=c in S, a does not divide b+c."""
    L = sorted(S)
    m = len(L)
    for i in range(m):
        a = L[i]
        for j in range(i + 1, m):
            b = L[j]
            for k in range(j, m):
                c = L[k]
                if (b + c) % a == 0:
                    return False
    return True


def _interval_seed():
    """Construct a valid set using the interval [floor(2N/3)+1, N]."""
    n = N
    start = (2 * n) // 3 + 1
    if start < 1:
        start = 1
    return list(range(start, n + 1))


def _greedy_with_bitsets():
    """
    Greedy construction using bitset-based forward checking.
    
    For each candidate x (processed in descending order), we check if adding x
    would create a violation. A violation occurs if there exist b, c in S with
    b >= c >= x+1 (since x is smallest) such that x | (b+c).
    
    We maintain for each potential divisor d, the set of sums s = b+c (where b,c in S, b>=c)
    that are multiples of d. When considering adding x, we check if any such sum exists
    where both b and c > x.
    """
    n = N
    S = []
    S_set = set()
    
    # For each element x in S, we track which sums b+c (b,c in S, b>=c, b>c or b=c)
    # are divisible by x. But more efficiently: for each potential new element x,
    # we need to know if any pair (b,c) in S has b+c ≡ 0 (mod x) and b,c > x.
    
    # Precompute: for each x in 1..N, which residues mod x are "dangerous"
    # Actually, let's use a different approach: maintain a set of all pair sums.
    
    pair_sums = set()
    
    for x in range(n, 0, -1):
        if time.time() > DEADLINE:
            break
        
        # Check if x can be added: no pair (b,c) in S with b,c > x and x | (b+c)
        valid = True
        # We need to check pairs where both elements are > x
        # Since S is built in descending order, elements in S that are > x are those added earlier
        candidates_gt_x = [s for s in S if s > x]
        
        for i in range(len(candidates_gt_x)):
            b = candidates_gt_x[i]
            for j in range(i, len(candidates_gt_x)):
                c = candidates_gt_x[j]
                if (b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.append(x)
            S_set.add(x)
            # Add new pair sums involving x
            for s in S:
                if s > x:
                    pair_sums.add(x + s)
                elif s == x:
                    pair_sums.add(2 * x)
    
    return S


def _greedy_descending():
    """Construct a valid set using greedy descending order."""
    n = N
    S = []
    for x in range(n, 0, -1):
        valid = True
        # Check if x divides b+c for any b,c in S
        # Since we add in descending order, all b,c in S are > x.
        # Thus x is the smallest element in any new triple (x, b, c).
        for i in range(len(S)):
            b = S[i]
            for j in range(i, len(S)):
                c = S[j]
                if (b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.append(x)
    return S


def _greedy_from_interval():
    """
    Start from the valid interval [171, 256] and try to add smaller elements.
    """
    n = N
    S = list(range(171, n + 1))
    S_set = set(S)
    
    for x in range(170, 0, -1):
        if time.time() > DEADLINE:
            break
        
        # Check if x can be added
        valid = True
        # x must not divide b+c for any b,c in S with b,c > x
        # Since all elements in S are > x (we're adding in descending order),
        # we need to check all pairs in S
        for i in range(len(S)):
            b = S[i]
            for j in range(i, len(S)):
                c = S[j]
                if (b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.append(x)
            S_set.add(x)
    
    return S


def _greedy_from_interval_ascending():
    """
    Start from the valid interval [171, 256] and try to add smaller elements
    in ascending order (try 1, 2, 3, ... up to 170).
    """
    n = N
    S = list(range(171, n + 1))
    S_set = set(S)
    
    for x in range(1, 171):
        if time.time() > DEADLINE:
            break
        
        # Check if x can be added
        valid = True
        # x must not divide b+c for any b,c in S with b,c > x
        # All elements in S are > x, so check all pairs
        for i in range(len(S)):
            b = S[i]
            for j in range(i, len(S)):
                c = S[j]
                if (b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.append(x)
            S_set.add(x)
    
    return S


def construct():
    # Generate multiple candidates using different strategies
    candidates = []
    
    # Strategy 1: Pure greedy descending
    cand1 = _greedy_descending()
    if cand1:
        candidates.append(cand1)
    
    # Strategy 2: Interval seed
    cand2 = _interval_seed()
    if cand2:
        candidates.append(cand2)
    
    # Strategy 3: Start from interval, add smaller elements descending
    cand3 = _greedy_from_interval()
    if cand3:
        candidates.append(cand3)
    
    # Strategy 4: Start from interval, add smaller elements ascending
    cand4 = _greedy_from_interval_ascending()
    if cand4:
        candidates.append(cand4)
    
    # Validate and pick the largest valid one
    best = []
    for cand in candidates:
        if _validate(set(cand)) and len(cand) > len(best):
            best = cand
    
    # Fallback if all fail (should not happen)
    if not best:
        best = _interval_seed()
    
    return sorted(best)


# Self-check at import (cheap, deterministic).
if __name__ == "__main__":
    _s = construct()
    assert _validate(set(_s)), "constructed set is invalid"
    print("size", len(_s), "first", _s[:5], "last", _s[-5:])
