import time
import random
from math import gcd
from itertools import combinations

_DEADLINE = 60.0
_START = None
_CAND = list(range(1, 129))
_PRE = None
_BALL = None
_PAIRS_GCD = None  # PAIRS_GCD[g] = list of (a,b) with gcd(a,b)=g
_GCD1 = None
_GCD_MASKS = None  # For each x, for each g, a bitmask of elements with gcd(x,.)=g
_PAIRS_GCD_MASKS = None  # For each g, a dict mapping element->bitmask of partners with gcd=g

def _ensure_pre():
    global _PRE, _BALL, _PAIRS_GCD, _GCD1, _GCD_MASKS, _PAIRS_GCD_MASKS
    if _PRE is not None:
        return
    n = 128
    PRE = [[1] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            PRE[i][j] = PRE[j][i] = gcd(i, j)
    _PRE = PRE

    BALL = {}
    for x in range(1, n + 1):
        d = {}
        for y in range(1, n + 1):
            g = PRE[x][y]
            d.setdefault(g, []).append(y)
        BALL[x] = d
    _BALL = BALL
    _GCD1 = {x: BALL[x].get(1, []) for x in range(1, n + 1)}

    # Precompute pairs with gcd g
    PAIRS_GCD = {}
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g = PRE[a][b]
            if g not in PAIRS_GCD:
                PAIRS_GCD[g] = []
            PAIRS_GCD[g].append((a, b))
    _PAIRS_GCD = PAIRS_GCD

    # Precompute bitmasks for fast feasibility checks
    # GCD_MASKS[x][g] = bitmask of all y such that gcd(x,y) = g
    GCD_MASKS = {}
    for x in range(1, n + 1):
        d = {}
        for y in range(1, n + 1):
            g = PRE[x][y]
            if g not in d:
                d[g] = 0
            d[g] |= (1 << y)
        GCD_MASKS[x] = d
    _GCD_MASKS = GCD_MASKS

    # For each g, for each element a, bitmask of b > a such that gcd(a,b) = g
    # Actually, let's precompute: for each g, a list of (a, b) pairs
    # And for fast checking: for each g, for each a, bitmask of all b (b != a) with gcd(a,b)=g
    PAIRS_GCD_MASKS = {}
    for g, pairs in PAIRS_GCD.items():
        elem_mask = {}
        for a, b in pairs:
            if a not in elem_mask:
                elem_mask[a] = 0
            if b not in elem_mask:
                elem_mask[b] = 0
            elem_mask[a] |= (1 << b)
            elem_mask[b] |= (1 << a)
        PAIRS_GCD_MASKS[g] = elem_mask
    _PAIRS_GCD_MASKS = PAIRS_GCD_MASKS

def _bad_triples_in(S):
    P = _PRE
    s = list(S)
    n = len(s)
    cnt = 0
    for i in range(n):
        a = s[i]
        for j in range(i + 1, n):
            b = s[j]
            g_ab = P[a][b]
            for k in range(j + 1, n):
                c = s[k]
                if P[a][c] == g_ab and P[b][c] == g_ab:
                    cnt += 1
    return cnt

def _can_add(Sset, x):
    """
    Check if x can be added to Sset.
    Sset is a set.
    """
    P = _PRE
    # Group elements of Sset by gcd with x
    groups = {}
    for u in Sset:
        g = P[x][u]
        if g not in groups:
            groups[g] = []
        groups[g].append(u)
    
    for g, lst in groups.items():
        if len(lst) < 2:
            continue
        # Check if any pair in lst has gcd g
        # Use precomputed pairs for this g
        pairs = _PAIRS_GCD.get(g, [])
        # Create a set of lst for O(1) lookup
        lst_set = set(lst)
        for a, b in pairs:
            if a in lst_set and b in lst_set:
                return False
    return True

def _can_add_fast(Smask, x):
    """
    Check if x can be added to the set represented by Smask (bitmask).
    Uses precomputed bitmasks for speed.
    """
    GCD_MASKS_x = _GCD_MASKS[x]
    for g, mask in GCD_MASKS_x.items():
        # Elements in S that have gcd g with x
        common = Smask & mask
        if common == 0:
            continue
        # Check if common contains any pair (a,b) with gcd(a,b) = g
        # Use precomputed PAIRS_GCD_MASKS
        elem_mask = _PAIRS_GCD_MASKS.get(g, {})
        # For each element a in common, check if any of its partners (with gcd g) are also in common
        # We can do: for each a in common, if elem_mask[a] & common has any bit set other than a itself, return False
        # But we need to be careful: elem_mask[a] includes b where gcd(a,b)=g
        # So if elem_mask[a] & common has a bit set, it means there's a partner b in common
        # We need at least one such partner that is not a itself
        # Since gcd(a,a) is not g (unless g=a), and we only consider distinct elements, this should be fine
        # Actually, let's iterate through bits in common
        c = common
        while c:
            # Get lowest set bit
            lb = c & (-c)
            a = lb.bit_length() - 1
            # Check if a has any partner in common with gcd g
            partners = elem_mask.get(a, 0)
            if partners & common:
                return False
            c ^= lb
    return True

def _greedy(order):
    S = []
    Sset = set()
    Smask = 0
    for x in order:
        if _can_add_fast(Smask, x):
            S.append(x)
            Sset.add(x)
            Smask |= (1 << x)
    return S

def _greedy_remove_worst(S):
    """
    Remove the element involved in the most potential bad triples,
    then greedily re-add elements.
    """
    P = _PRE
    S = list(S)
    Sset = set(S)
    Smask = 0
    for x in S:
        Smask |= (1 << x)
    
    # Find the element that is in the most "would-be bad triples"
    # For each element x in S, count how many pairs (a,b) in S\{x} 
    # would form a bad triple with x
    # A triple (x,a,b) is bad if gcd(x,a)=gcd(x,b)=gcd(a,b)
    
    # For efficiency, for each x, group other elements by gcd with x
    # For each group with >= 2 elements, count pairs in that group that have gcd equal to the group's gcd
    # Actually, let's just count: for each x, for each g, let C_g = {u in S\{x} : gcd(x,u)=g}
    # Then the number of bad triples involving x with gcd g is the number of pairs (a,b) in C_g with gcd(a,b)=g
    
    worst_x = None
    worst_count = -1
    
    for x in S:
        count = 0
        groups = {}
        for u in S:
            if u == x:
                continue
            g = P[x][u]
            if g not in groups:
                groups[g] = []
            groups[g].append(u)
        
        for g, lst in groups.items():
            if len(lst) < 2:
                continue
            # Count pairs in lst with gcd = g
            lst_set = set(lst)
            pairs = _PAIRS_GCD.get(g, [])
            for a, b in pairs:
                if a in lst_set and b in lst_set:
                    count += 1
        
        if count > worst_count:
            worst_count = count
            worst_x = x
    
    if worst_x is None or worst_count <= 0:
        return S, Sset, Smask
    
    # Remove worst_x
    S.remove(worst_x)
    Sset.discard(worst_x)
    Smask &= ~(1 << worst_x)
    
    # Greedily re-add elements
    added = True
    while added:
        added = False
        for x in _CAND:
            if x not in Sset and _can_add_fast(Smask, x):
                S.append(x)
                Sset.add(x)
                Smask |= (1 << x)
                added = True
    
    return S, Sset, Smask

def _local_search(S):
    P = _PRE
    S = list(S)
    Sset = set(S)
    Smask = 0
    for x in S:
        Smask |= (1 << x)
    
    max_rounds = 200
    for _ in range(max_rounds):
        if time.time() - _START > _DEADLINE - 5:
            break
        
        # Try adds
        added = True
        while added:
            added = False
            for x in _CAND:
                if x not in Sset and _can_add_fast(Smask, x):
                    S.append(x)
                    Sset.add(x)
                    Smask |= (1 << x)
                    added = True
        if time.time() - _START > _DEADLINE - 5:
            break
        
        # Try swaps
        improved = False
        for r in list(S):
            if time.time() - _START > _DEADLINE - 5:
                break
            Sset.discard(r)
            Smask &= ~(1 << r)
            for x in _CAND:
                if x in Sset or x == r:
                    continue
                if _can_add_fast(Smask, x):
                    # Swap
                    S.remove(r)
                    S.append(x)
                    Sset.add(x)
                    Smask |= (1 << x)
                    improved = True
                    break
            if improved:
                break
            else:
                Sset.add(r)
                Smask |= (1 << r)
        if not improved:
            # Try remove-worst-offender
            S_new, Sset_new, Smask_new = _greedy_remove_worst(S)
            if len(S_new) > len(S):
                S = S_new
                Sset = Sset_new
                Smask = Smask_new
                continue
            else:
                break
    
    return S

def _construct():
    global _START
    _ensure_pre()
    _START = time.time()
    
    best = []
    best_len = -1
    
    # Deterministic greedy
    S = _greedy(list(range(1, 129)))
    S = _local_search(S)
    if len(S) > best_len:
        best = S
        best_len = len(S)
        
    # Random restarts
    rng = random.Random(42)
    while time.time() - _START < _DEADLINE - 10:
        order = list(range(1, 129))
        rng.shuffle(order)
        S = _greedy(order)
        S = _local_search(S)
        if len(S) > best_len:
            best = S
            best_len = len(S)
            
    # Final polish: try more aggressive local search on the best solution
    if time.time() - _START < _DEADLINE - 3:
        S = list(best)
        S = _local_search(S)
        if len(S) > best_len:
            best = S
            best_len = len(S)
        
        # Try remove-worst a few more times
        for _ in range(5):
            if time.time() - _START > _DEADLINE - 2:
                break
            Sset = set(best)
            Smask = 0
            for x in best:
                Smask |= (1 << x)
            S_new, Sset_new, Smask_new = _greedy_remove_worst(best)
            if len(S_new) > len(best):
                best = S_new
                best_len = len(S_new)
            else:
                break
    
    return best

def construct():
    try:
        res = _construct()
    except Exception:
        res = [2]
    if not res:
        res = [2]
    return res
