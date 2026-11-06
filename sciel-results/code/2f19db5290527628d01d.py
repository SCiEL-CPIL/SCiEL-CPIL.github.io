import time
import random
from math import gcd

_DEADLINE = 85.0
_START = None
_CAND = list(range(1, 129))
_PRE = None
_PAIRS_GCD = None

def _ensure_pre():
    global _PRE, _PAIRS_GCD
    if _PRE is not None:
        return
    n = 128
    PRE = [[1] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            PRE[i][j] = PRE[j][i] = gcd(i, j)
    _PRE = PRE

    PAIRS_GCD = {}
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g = PRE[a][b]
            if g not in PAIRS_GCD:
                PAIRS_GCD[g] = []
            PAIRS_GCD[g].append((a, b))
    _PAIRS_GCD = PAIRS_GCD

def _bad_triples_count(S):
    """Count number of bad triples in S."""
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
    """Check if x can be added to Sset without creating a bad triple."""
    P = _PRE
    groups = {}
    for u in Sset:
        g = P[x][u]
        if g not in groups:
            groups[g] = []
        groups[g].append(u)
    
    for g, lst in groups.items():
        if len(lst) < 2:
            continue
        lst_set = set(lst)
        pairs = _PAIRS_GCD.get(g, [])
        for a, b in pairs:
            if a in lst_set and b in lst_set:
                return False
    return True

def _greedy(order):
    S = []
    Sset = set()
    for x in order:
        if _can_add(Sset, x):
            S.append(x)
            Sset.add(x)
    return S

def _local_search(S):
    """Improve S by trying adds and swaps."""
    S = list(S)
    Sset = set(S)
    
    # Phase 1: Try to add elements
    improved = True
    while improved and (time.time() - _START < _DEADLINE - 5):
        improved = False
        for x in _CAND:
            if time.time() - _START > _DEADLINE - 5:
                break
            if x not in Sset and _can_add(Sset, x):
                S.append(x)
                Sset.add(x)
                improved = True
    
    # Phase 2: Try swaps
    for _ in range(20):
        if time.time() - _START > _DEADLINE - 5:
            break
        found = False
        for r in list(S):
            if time.time() - _START > _DEADLINE - 5:
                break
            Sset.discard(r)
            for x in _CAND:
                if x in Sset or x == r:
                    continue
                if _can_add(Sset, x):
                    S.remove(r)
                    S.append(x)
                    Sset.add(x)
                    found = True
                    break
            if found:
                break
            else:
                Sset.add(r)
        if not found:
            break
    
    return S

def _local_search2(S):
    """Alternative local search: remove worst element, then re-add."""
    S = list(S)
    Sset = set(S)
    P = _PRE
    
    for _ in range(10):
        if time.time() - _START > _DEADLINE - 5:
            break
        
        # Find element involved in most bad triples
        if not S:
            break
        elem_counts = {x: 0 for x in S}
        s = list(S)
        n = len(s)
        for i in range(n):
            a = s[i]
            for j in range(i + 1, n):
                b = s[j]
                g_ab = P[a][b]
                for k in range(j + 1, n):
                    c = s[k]
                    if P[a][c] == g_ab and P[b][c] == g_ab:
                        elem_counts[a] += 1
                        elem_counts[b] += 1
                        elem_counts[c] += 1
        
        worst = max(elem_counts, key=elem_counts.get)
        if elem_counts[worst] == 0:
            break
        
        S.remove(worst)
        Sset.discard(worst)
        
        # Re-add greedily
        for x in _CAND:
            if x not in Sset and _can_add(Sset, x):
                S.append(x)
                Sset.add(x)
    
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
        best = S[:]
        best_len = len(S)
    
    S = _local_search2(best)
    if len(S) > best_len:
        best = S[:]
        best_len = len(S)
    
    # Random restarts
    rng = random.Random(42)
    while time.time() - _START < _DEADLINE - 10:
        order = list(range(1, 129))
        rng.shuffle(order)
        S = _greedy(order)
        S = _local_search(S)
        if len(S) > best_len:
            best = S[:]
            best_len = len(S)
        
        S = _local_search2(S)
        if len(S) > best_len:
            best = S[:]
            best_len = len(S)
    
    return best

def construct():
    try:
        res = _construct()
    except Exception:
        res = [2]
    if not res:
        res = [2]
    return res
