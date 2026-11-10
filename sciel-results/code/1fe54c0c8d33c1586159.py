"""MWIS on 3-AP conflict graph with branch-and-bound + fractional LP relaxation bound."""
import time
import math

N = 256
WEIGHT = [1.0 / (i + 1) for i in range(N)]

def build_conflicts():
    """Build conflict graph: two numbers conflict if they form part of a 3-AP."""
    conf = [0] * N
    for i in range(N):
        x = i + 1
        for j in range(i + 1, N):
            y = j + 1
            d = y - x
            # Check if x,y can be part of any 3-term AP
            # Case 1: x,y are terms 1,2 -> term3 = y+d <= N
            # Case 2: x,y are terms 2,3 -> term1 = x-d >= 1
            # Case 3: x,y are terms 1,3 -> term2 = (x+y)/2 must be integer
            hit = False
            if y + d <= N:
                hit = True
            elif x - d >= 1:
                hit = True
            elif (x + y) % 2 == 0:
                m = (x + y) // 2
                if 1 <= m <= N and m != x and m != y:
                    hit = True
            if hit:
                conf[i] |= (1 << j)
                conf[j] |= (1 << i)
    return conf

CONF = build_conflicts()

_start_time = None
_best_set = None
_best_val = 0.0
_time_limit = 55.0

def _time_left():
    return _start_time is None or (time.time() - _start_time) < _time_limit

def _greedy_is(cand_mask):
    """Greedy maximal independent set by weight descending."""
    verts = [i for i in range(N) if (cand_mask >> i) & 1]
    verts.sort(key=lambda i: -WEIGHT[i])
    chosen = []
    avail = cand_mask
    for v in verts:
        if (avail >> v) & 1:
            chosen.append(v)
            avail &= ~CONF[v]
            avail &= ~(1 << v)
    return chosen

def _fractional_bound(cand_mask):
    """
    Fractional LP relaxation upper bound.
    For each vertex v, let deg_v be its degree within cand_mask.
    In a fractional independent set, x_v <= 1/(1 + deg_v * something)...
    
    Better approach: Use the fact that for any clique in the conflict graph,
    sum of x_v <= 1. We can use greedy coloring to get a set of cliques (color classes
    in the complement... actually use the conflict graph structure).
    
    Simplest strong bound: For each vertex, the LP constraint is x_v <= 1 and for each
    edge (u,v), x_u + x_v <= 1. A simple bound: sort vertices by weight desc, and
    use the fact that in an independent set, no two adjacent vertices can both be chosen.
    
    Use a greedy sequential bound: process vertices in weight-desc order.
    Maintain a 'budget' - when we pick vertex v, we must exclude its neighbors.
    This gives an upper bound by assuming we can fractionally take from remaining.
    
    Actually, let's use a simpler but effective bound: 
    ub = sum of weights of all vertices in cand_mask, but for each connected component
    of the conflict graph induced by cand_mask, the independent set can take at most
    the maximum weight matching... 
    
    Let me use a practical bound: greedy coloring of the conflict graph induced by cand_mask.
    The number of colors times... no.
    
    Best practical bound: Use the fact that for each vertex v in cand_mask, 
    x_v + sum_{u in N(v) cap cand} x_u <= 1 is not quite right.
    
    Let me use: ub = sum over all v in cand of WEIGHT[v] / (1 + deg_v_in_cand)
    This is a valid fractional bound because for any independent set S,
    sum_{v in S} WEIGHT[v] <= sum_{v in cand} WEIGHT[v] / (1 + deg_v)
    ... this isn't obviously valid.
    
    Simpler: Use the Lovasz theta function approximation via greedy coloring.
    Color the conflict graph; the independence number <= sum over color classes of max weight.
    """
    verts = [i for i in range(N) if (cand_mask >> i) & 1]
    if not verts:
        return 0.0
    verts.sort(key=lambda i: -WEIGHT[i])
    
    # Greedy coloring of conflict graph induced by cand_mask
    color_of = {}
    col_max = {}
    for v in verts:
        used = set()
        cm = CONF[v] & cand_mask
        while cm:
            lsb = cm & -cm
            nb = lsb.bit_length() - 1
            cm ^= lsb
            if nb in color_of:
                used.add(color_of[nb])
        c = 0
        while c in used:
            c += 1
        color_of[v] = c
        if c not in col_max or WEIGHT[v] > col_max[c]:
            col_max[c] = WEIGHT[v]
    return sum(col_max.values())

def _search(cand_mask, cur_val, chosen, depth):
    global _best_set, _best_val
    if not _time_left():
        return
    if cand_mask == 0:
        if cur_val > _best_val:
            _best_val = cur_val
            _best_set = list(chosen)
        return
    
    # Upper bound
    ub = _fractional_bound(cand_mask)
    if cur_val + ub <= _best_val + 1e-12:
        return
    
    # Pick branching vertex: highest weight in cand_mask
    verts = [i for i in range(N) if (cand_mask >> i) & 1]
    v = max(verts, key=lambda i: WEIGHT[i])
    
    # Branch 1: include v
    new_cand = cand_mask & ~CONF[v] & ~(1 << v)
    _search(new_cand, cur_val + WEIGHT[v], chosen + [v], depth + 1)
    if not _time_left():
        return
    
    # Branch 2: exclude v
    _search(cand_mask & ~(1 << v), cur_val, chosen, depth + 1)

def construct():
    global _start_time, _best_set, _best_val
    _start_time = time.time()
    _best_val = 0.0
    _best_set = None
    full = (1 << N) - 1
    
    # Seed with greedy solution
    g = _greedy_is(full)
    _best_set = g
    _best_val = sum(WEIGHT[i] for i in g)
    
    # Run branch and bound
    _search(full, 0.0, [], 0)
    
    if _best_set is None:
        _best_set = [0]
    
    return [i + 1 for i in _best_set]
