import random
import time

_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_Q3_ADJ = [0] * 8
for _a, _b in _Q3_EDGES:
    _Q3_ADJ[_a] |= (1 << _b)
    _Q3_ADJ[_b] |= (1 << _a)

# Precompute all Q3 edge orientations with sorted unassigned vertices
_Q3_EDGE_ORIENTS = []
for qa, qb in _Q3_EDGES:
    for (gu_q, gv_q) in [(qa, qb), (qb, qa)]:
        unassigned = [i for i in range(8) if i != gu_q and i != gv_q]
        def _cc(x, a, b):
            c = 0
            if _Q3_ADJ[x] & (1 << a): c += 1
            if _Q3_ADJ[x] & (1 << b): c += 1
            return c
        unassigned.sort(key=lambda x: _cc(x, gu_q, gv_q), reverse=True)
        _Q3_EDGE_ORIENTS.append((gu_q, gv_q, tuple(unassigned)))

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 16
    full_mask = (1 << n) - 1

    # Start with K_{3,13}
    adj = [0] * n
    for u in range(3):
        for v in range(3, 16):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    def has_q3_after_add(u, v):
        """Check if adding edge (u,v) creates a Q3 subgraph using bitmask operations."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        result = False
        for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
            for (gu, gv) in [(u, v), (v, u)]:
                assignment = [None] * 8
                assignment[qa] = gu
                assignment[qb] = gv
                used_mask = (1 << gu) | (1 << gv)
                
                def backtrack(idx, used_mask):
                    if time.time() > deadline - 5:
                        return False
                    if idx == len(unassigned):
                        # All assigned, verify all Q3 edges
                        for a, b in _Q3_EDGES:
                            ga = assignment[a]
                            gb = assignment[b]
                            if not (adj[ga] & (1 << gb)):
                                return False
                        return True
                    
                    x = unassigned[idx]
                    candidates = full_mask & ~used_mask
                    
                    # Filter candidates: must be adjacent to all assigned Q3-neighbors of x
                    for y in range(8):
                        if assignment[y] is not None and (_Q3_ADJ[x] & (1 << y)):
                            hy = assignment[y]
                            candidates &= adj[hy]
                    
                    while candidates:
                        bit = candidates & (-candidates)
                        g = bit.bit_length() - 1
                        candidates ^= bit
                        
                        assignment[x] = g
                        new_used = used_mask | (1 << g)
                        if backtrack(idx + 1, new_used):
                            return True
                        assignment[x] = None
                    
                    return False
                
                if backtrack(0, used_mask):
                    result = True
                    break
            if result:
                break
        
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return result

    # Try to add missing edges
    missing = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                missing.append((i, j))
    
    random.shuffle(missing)
    
    for u, v in missing:
        if time.time() > deadline - 15:
            break
        if not has_q3_after_add(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges
