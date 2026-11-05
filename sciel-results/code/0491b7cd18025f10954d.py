import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def _build():
    edge_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_map[(i, j)] = idx
            idx += 1
    
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    for edges in [
                        [(a,b),(b,c),(c,d),(d,a)],
                        [(a,b),(b,d),(d,c),(c,a)],
                        [(a,c),(c,b),(b,d),(d,a)]
                    ]:
                        idxs = tuple(edge_map[(min(e[0],e[1]), max(e[0],e[1]))] for e in edges)
                        cycles.append(idxs)
    
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(cycles):
        for ei in cyc:
            edge_to_cycles[ei].append(ci)
    
    return edge_map, cycles, edge_to_cycles

EDGE_MAP, FOUR_CYCLES, EDGE_TO_CYCLES = _build()
NUM_CYCLES = len(FOUR_CYCLES)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    random.seed(777)
    
    # Precompute for fast cycle checking
    # For each cycle, the 4 edge indices
    cyc_edges = np.array(FOUR_CYCLES, dtype=np.int32)
    
    def distinct_count(c):
        return len(set(c.tolist()))
    
    def is_valid(c):
        # Vectorized check
        for ci in range(NUM_CYCLES):
            e = cyc_edges[ci]
            v0, v1, v2, v3 = c[e[0]], c[e[1]], c[e[2]], c[e[3]]
            if v0 != v1 and v0 != v2 and v0 != v3 and v1 != v2 and v1 != v3 and v2 != v3:
                return False
        return True
    
    # Initial: (i+j)%4
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i+1, N):
            colors[EDGE_MAP[(i,j)]] = (i + j) % 4
    
    best = colors.copy()
    best_dc = distinct_count(best)
    
    # Greedy: try to add new colors one at a time
    # For each edge, try to assign a new color
    for iteration in range(50):
        if time.time() >= deadline:
            break
        
        cur = best.copy()
        improved = False
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        for ei in edges:
            if time.time() >= deadline:
                break
            old = cur[ei]
            # Try colors not yet in cur
            used = set(cur.tolist())
            for nc in range(120):
                if nc in used:
                    continue
                cur[ei] = nc
                ok = True
                for ci in EDGE_TO_CYCLES[ei]:
                    e = cyc_edges[ci]
                    v0, v1, v2, v3 = cur[e[0]], cur[e[1]], cur[e[2]], cur[e[3]]
                    if v0 != v1 and v0 != v2 and v0 != v3 and v1 != v2 and v1 != v3 and v2 != v3:
                        ok = False
                        break
                if ok:
                    dc = distinct_count(cur)
                    if dc > best_dc:
                        best_dc = dc
                        best = cur.copy()
                    improved = True
                    break
                else:
                    cur[ei] = old
        
        if not improved:
            # Try swapping: change edge to existing color to enable future changes
            for ei in edges:
                if time.time() >= deadline:
                    break
                old = cur[ei]
                used = list(set(cur.tolist()))
                random.shuffle(used)
                for nc in used:
                    if nc == old:
                        continue
                    cur[ei] = nc
                    ok = True
                    for ci in EDGE_TO_CYCLES[ei]:
                        e = cyc_edges[ci]
                        v0, v1, v2, v3 = cur[e[0]], cur[e[1]], cur[e[2]], cur[e[3]]
                        if v0 != v1 and v0 != v2 and v0 != v3 and v1 != v2 and v1 != v3 and v2 != v3:
                            ok = False
                            break
                    if ok:
                        break
                    else:
                        cur[ei] = old
    
    if not is_valid(best):
        return [0] * NUM_EDGES
    
    return [int(x) for x in best.tolist()]
