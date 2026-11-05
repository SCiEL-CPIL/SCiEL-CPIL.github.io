import random
import itertools

_N = 20
_CYCLES = None
_EDGE_TO_IDX = None

def _init_data():
    global _CYCLES, _EDGE_TO_IDX
    if _CYCLES is not None:
        return
    
    n = _N
    edge_to_idx = {}
    idx = 0
    # We don't need to store edges here, just the mapping
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = idx
            edge_to_idx[(v, u)] = idx
            idx += 1
            
    _EDGE_TO_IDX = edge_to_idx
    
    cycles = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e1 = edge_to_idx[(a,b)]
        e2 = edge_to_idx[(b,c)]
        e3 = edge_to_idx[(c,d)]
        e4 = edge_to_idx[(d,a)]
        cycles.append([e1, e2, e3, e4])
        
        e1 = edge_to_idx[(a,b)]
        e2 = edge_to_idx[(b,d)]
        e3 = edge_to_idx[(d,c)]
        e4 = edge_to_idx[(c,a)]
        cycles.append([e1, e2, e3, e4])
        
        e1 = edge_to_idx[(a,c)]
        e2 = edge_to_idx[(c,b)]
        e3 = edge_to_idx[(b,d)]
        e4 = edge_to_idx[(d,a)]
        cycles.append([e1, e2, e3, e4])
        
    _CYCLES = cycles

def _solve_with_permutation(perm):
    n = _N
    _init_data()
    edge_to_idx = _EDGE_TO_IDX
    cycles = _CYCLES
    
    # Build edges list with colors
    edges = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edges.append([u, v, 0])
            idx += 1
            
    # Color
    k = n // 2
    center = n - 1
    for i in range(len(edges)):
        u, v, _ = edges[i]
        pu = perm[u]
        pv = perm[v]
        
        if pu == center or pv == center:
            other = pu if pv == center else pv
            edges[i][2] = other % (n - 1)
        else:
            edges[i][2] = (pu + pv) % (n - 1)

    active = [True] * len(edges)
    
    for _ in range(200):
        conflict_count = [0] * len(edges)
        bad_cycles_count = 0
        
        for cycle in cycles:
            e1, e2, e3, e4 = cycle
            if not (active[e1] and active[e2] and active[e3] and active[e4]):
                continue
            
            c1 = edges[e1][2]
            c2 = edges[e2][2]
            c3 = edges[e3][2]
            c4 = edges[e4][2]
            
            if c1 == c3 or c2 == c4:
                bad_cycles_count += 1
                conflict_count[e1] += 1
                conflict_count[e2] += 1
                conflict_count[e3] += 1
                conflict_count[e4] += 1
                
        if bad_cycles_count == 0:
            break
            
        max_count = -1
        edge_to_remove = -1
        for i in range(len(edges)):
            if active[i] and conflict_count[i] > max_count:
                max_count = conflict_count[i]
                edge_to_remove = i
                
        if edge_to_remove == -1:
            break
            
        active[edge_to_remove] = False
        
    result = []
    for i, (u, v, c) in enumerate(edges):
        if active[i]:
            result.append([u, v, c])
            
    return result

def construct():
    n = _N
    best_result = []
    max_edges = -1
    
    identity = list(range(n))
    res = _solve_with_permutation(identity)
    if len(res) > max_edges:
        max_edges = len(res)
        best_result = res
        
    random.seed(42)
    for _ in range(30): # Increase iterations slightly
        perm = list(range(n))
        random.shuffle(perm)
        res = _solve_with_permutation(perm)
        if len(res) > max_edges:
            max_edges = len(res)
            best_result = res
            
    return best_result
