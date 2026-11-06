import time
import random
from math import gcd

_DEADLINE = 60.0
_START = None
_CAND = list(range(1, 129))
_PRE = None
_BALL = None
_PAIRS_GCD = None
_GCD1 = None
_GCD_MASKS = None
_PAIRS_GCD_MASKS = None
_PAIRS_GCD_SET = None

def _ensure_pre():
    global _PRE, _BALL, _PAIRS_GCD, _GCD1, _GCD_MASKS, _PAIRS_GCD_MASKS, _PAIRS_GCD_SET
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

    PAIRS_GCD = {}
    PAIRS_GCD_SET = {}
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g = PRE[a][b]
            if g not in PAIRS_GCD:
                PAIRS_GCD[g] = []
                PAIRS_GCD_SET[g] = set()
            PAIRS_GCD[g].append((a, b))
            PAIRS_GCD_SET[g].add((a, b))
    _PAIRS_GCD = PAIRS_GCD
    _PAIRS_GCD_SET = PAIRS_GCD_SET

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

def _can_add_fast(Smask, x):
    GCD_MASKS_x = _GCD_MASKS[x]
    for g, mask in GCD_MASKS_x.items():
        common = Smask & mask
        if common == 0:
            continue
        elem_mask = _PAIRS_GCD_MASKS.get(g, {})
        c = common
        while c:
            lb = c & (-c)
            a = lb.bit_length() - 1
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
            break
    
    return S

def _simulated_annealing(S, start_temp=1.0, end_temp=0.01, cooling=0.95, max_iters=5000):
    """
    Simulated annealing with O(n^2) incremental conflict counting.
    State: bitmask Smask representing the current set.
    Move: swap one element in S with one element not in S.
    Cost: number of bad triples (should be 0 for valid sets).
    We want to maximize cardinality, so we track (cardinality, -bad_triples) as objective.
    """
    P = _PRE
    S = list(S)
    Sset = set(S)
    Smask = 0
    for x in S:
        Smask |= (1 << x)
    
    # Precompute for each g, for each pair (a,b) with gcd(a,b)=g
    # We need to efficiently count bad triples.
    # A bad triple is (a,b,c) with gcd(a,b)=gcd(a,c)=gcd(b,c)=g.
    # For a given g, let E_g be the set of pairs (a,b) with gcd(a,b)=g.
    # A triple (a,b,c) is bad with gcd g iff all three pairs are in E_g.
    # So for each g, we need to count triangles in the graph G_g = (V, E_g) induced by S.
    
    # To make this fast, we can maintain for each g, the induced subgraph on S,
    # and count triangles incrementally.
    # But triangle counting is O(n^3) in the worst case per g.
    # Alternative: for each g, maintain adjacency as bitmasks, and count triangles by
    # for each vertex v in S, count pairs (u,w) in S such that u~v and w~v and u~w in G_g.
    # This is O(sum_v deg_g(v)^2) which is manageable.
    
    # Let's precompute for each g, the adjacency bitmask for each vertex.
    # adj_g[v] = bitmask of u such that gcd(u,v)=g.
    # This is already in _PAIRS_GCD_MASKS.
    
    def count_bad_triples(Smask):
        total = 0
        for g in _PAIRS_GCD_MASKS:
            # For each vertex v in S, count triangles involving v in G_g
            # For each v, let N_v = Smask & _PAIRS_GCD_MASKS[g].get(v, 0)
            # Then for each pair (u,w) in N_v with u<w, check if gcd(u,w)=g
            # i.e., check if w is in _PAIRS_GCD_MASKS[g].get(u, 0)
            # This is O(n * deg^2) per g, which might be too slow.
            # Alternative: for each g, iterate over pairs in E_g that are in S,
            # and for each pair (a,b), count c in S such that c is connected to both a and b in G_g.
            # That is, c is in S & adj_g[a] & adj_g[b].
            # Sum over all pairs (a,b) in E_g ∩ S.
            # Number of such pairs is O(n^2) per g in the worst case, but usually much less.
            pairs_in_S = 0
            for a, b in _PAIRS_GCD.get(g, []):
                if (Smask >> a) & 1 and (Smask >> b) & 1:
                    # Count c in S such that c is connected to both a and b in G_g
                    common = Smask & _PAIRS_GCD_MASKS[g].get(a, 0) & _PAIRS_GCD_MASKS[g].get(b, 0)
                    # Exclude a and b themselves
                    common &= ~(1 << a)
                    common &= ~(1 << b)
                    pairs_in_S += bin(common).count('1')
            # Each bad triple is counted 3 times (once for each pair)
            total += pairs_in_S // 3
        return total
    
    # This is still O(n^3) in the worst case. Let's use a different approach.
    # Maintain the number of bad triples incrementally.
    # When we remove r and add x, the change in bad triples can be computed
    # by considering only triples that involve r or x.
    
    def delta_bad_triples(Smask, r, x):
        """
        Compute the change in the number of bad triples when removing r and adding x.
        Smask is the current set including r but not x.
        """
        # Triples involving r that will be removed:
        # (r, a, b) where a,b in S\{r} and gcd(r,a)=gcd(r,b)=gcd(a,b)
        # Triples involving x that will be added:
        # (x, a, b) where a,b in S\{r} and gcd(x,a)=gcd(x,b)=gcd(a,b)
        # Also, triples involving both r and x are not possible since r is removed and x is added.
        
        S_without_r = Smask & ~(1 << r)
        
        # Count bad triples involving r in the original set
        removed = 0
        for g in _GCD_MASKS[r]:
            # a,b in S_without_r such that gcd(r,a)=g, gcd(r,b)=g, gcd(a,b)=g
            common = S_without_r & _GCD_MASKS[r][g]
            # For each pair (a,b) in common, check if gcd(a,b)=g
            c = common
            while c:
                lb = c & (-c)
                a = lb.bit_length() - 1
                # Count b in common such that b > a and gcd(a,b)=g
                partners = _PAIRS_GCD_MASKS.get(g, {}).get(a, 0)
                removed += bin((partners & common) & ~(1 << a)).count('1')
                c ^= lb
        removed //= 2  # Each pair counted twice
        
        # Count bad triples involving x in the new set
        added = 0
        for g in _GCD_MASKS[x]:
            common = S_without_r & _GCD_MASKS[x][g]
            c = common
            while c:
                lb = c & (-c)
                a = lb.bit_length() - 1
                partners = _PAIRS_GCD_MASKS.get(g, {}).get(a, 0)
                added += bin((partners & common) & ~(1 << a)).count('1')
                c ^= lb
        added //= 2
        
        return added - removed
    
    best_S = S
    best_Smask = Smask
    best_card = len(S)
    
    temp = start_temp
    for it in range(max_iters):
        if time.time() - _START > _DEADLINE - 5:
            break
        if temp < end_temp:
            break
        
        # Choose a random swap
        r = random.choice(S)
        candidates = [x for x in _CAND if x not in Sset]
        if not candidates:
            break
        x = random.choice(candidates)
        
        # Check if the swap is valid (no new bad triples)
        # Actually, we allow invalid states in simulated annealing, but we need to track the cost.
        # For simplicity, let's only consider valid swaps (cost = 0).
        # This is more like a random walk with restarts.
        
        S_without_r = Smask & ~(1 << r)
        if not _can_add_fast(S_without_r, x):
            temp *= cooling
            continue
        
        # Perform the swap
        S.remove(r)
        Sset.discard(r)
        S.append(x)
        Sset.add(x)
        Smask = S_without_r | (1 << x)
        
        # Since we only accept valid swaps, the cardinality stays the same.
        # To improve cardinality, we need to try adds.
        # Let's try to add an element after the swap.
        improved = False
        for y in _CAND:
            if y not in Sset and _can_add_fast(Smask, y):
                S.append(y)
                Sset.add(y)
                Smask |= (1 << y)
                improved = True
                break
        
        if improved:
            if len(S) > best_card:
                best_S = list(S)
                best_Smask = Smask
                best_card = len(S)
        else:
            # Revert the swap if it didn't lead to an improvement
            S.remove(x)
            Sset.discard(x)
            S.append(r)
            Sset.add(r)
            Smask = S_without_r | (1 << r)
        
        temp *= cooling
    
    return best_S

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
            
    # Simulated annealing on the best solution
    if time.time() - _START < _DEADLINE - 3:
        S = list(best)
        S = _simulated_annealing(S)
        if len(S) > best_len:
            best = S
            best_len = len(S)
            
        # Try local search again on the SA result
        if time.time() - _START < _DEADLINE - 2:
            S = list(best)
            S = _local_search(S)
            if len(S) > best_len:
                best = S
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
