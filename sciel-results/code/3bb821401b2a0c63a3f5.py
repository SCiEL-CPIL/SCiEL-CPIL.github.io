import math
import time

# Global state for the cap set greedy algorithm
_cap_set = set()
_conflict_cache = {}
_deadline = time.time() + 280.0  # Safety margin: 300s limit, start with 280s buffer
# We'll compute deadline at first call to be precise

def _init_deadline():
    global _deadline
    _deadline = time.time() + 285.0  # 15s margin before 300s hard limit

_init_deadline()

def _is_cap_set_check(vec):
    """Check if adding vec to _cap_set keeps it a cap set (no 3-term AP)."""
    v = tuple(vec)
    for u in _cap_set:
        # Check if u and v form a line: w = (u+v)/2 in Z_3^n
        # w_i = (u_i + v_i) * 2^{-1} mod 3 = (u_i + v_i) * 2 mod 3
        # Since 2^{-1} = 2 mod 3
        # But more simply: for each coordinate, if u_i != v_i, then the midpoint has value 1
        # Actually: (a+b)/2 mod 3. If a=b, result is a. If a!=b, result is the third value.
        # Values are 0,1,2. 0+1=1, 1/2=2 (since 2*2=4=1). Wait: 2*(1)=2, 2*2=4=1 mod 3.
        # So multiply by 2 mod 3: 
        # (0+0)*2=0, (0+1)*2=2, (0+2)*2=4=1, (1+1)*2=2, (1+2)*2=6=0, (2+2)*2=4=1
        # So midpoint values:
        # (0,0)->0, (0,1)->2, (0,2)->1, (1,1)->2, (1,2)->0, (2,2)->1
        w = []
        for ui, vi in zip(u, v):
            s = (ui + vi) % 3
            wi = (s * 2) % 3
            w.append(wi)
        if tuple(w) in _cap_set or tuple(w) == v or tuple(w) == u:
            # If w is u or v, that's fine (degenerate, but u!=v so w can't equal u or v unless u=v)
            # Actually if u != v, w is distinct from both in each coordinate where they differ.
            # The only issue is if w is already in _cap_set
            if tuple(w) in _cap_set:
                return False
    return True

def _count_conflicts(vec):
    """Count how many existing vectors would form a line with this vector."""
    v = tuple(vec)
    conflicts = 0
    for u in _cap_set:
        w = []
        for ui, vi in zip(u, v):
            s = (ui + vi) % 3
            wi = (s * 2) % 3
            w.append(wi)
        if tuple(w) in _cap_set:
            conflicts += 1
    return conflicts

def priority(vector, n):
    global _cap_set, _deadline
    
    # Check deadline
    if time.time() > _deadline:
        return -999999.0  # Reject all remaining vectors
    
    v = tuple(vector)
    
    # First, check if this vector can be added at all
    if not _is_cap_set_check(v):
        return -999998.0  # Can't add this vector
    
    # Compute conflict count: how many pairs in current set would be "blocked"
    # i.e., how many existing vectors u where (u+v)/2 is also in the set
    conflicts = _count_conflicts(v)
    
    # We want to MINIMIZE conflicts. Lower conflict = better priority.
    # Primary score: negative conflicts (so lower conflict -> higher priority)
    # Secondary: balance of symbols (prefer balanced vectors)
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    ideal = n / 3.0
    deviation = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # Tertiary: deterministic hash for tie-breaking
    h = 0
    base = 31
    for val in vector:
        h = (h * base + val) % (10**9 + 7)
    hash_score = h / (10**9 + 7)
    
    # Score: primary is minimizing conflicts, secondary is balance, tertiary is hash
    # Conflict count can range from 0 to len(_cap_set)
    # We want conflicts=0 to be best, so score = -conflicts * large_weight
    # Balance deviation is small (at most n), so use smaller weight
    score = -conflicts * 10000.0 - deviation * 10.0 + hash_score
    
    # Accept the vector into the cap set
    _cap_set.add(v)
    
    return float(score)
