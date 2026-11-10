import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def _edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_cycles():
    cycles = []
    for a in range(N - 3):
        for b in range(a + 1, N - 2):
            for c in range(b + 1, N - 1):
                for d in range(c + 1, N):
                    cycles.append((_edge_index(a,b), _edge_index(b,c), _edge_index(c,d), _edge_index(a,d)))
                    cycles.append((_edge_index(a,b), _edge_index(b,d), _edge_index(c,d), _edge_index(a,c)))
                    cycles.append((_edge_index(a,c), _edge_index(b,c), _edge_index(b,d), _edge_index(a,d)))
    return cycles

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

def _build_edge_to_cycles():
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(CYCLES):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

EDGE_TO_CYCLES = _build_edge_to_cycles()

def _count_violations(colors):
    violations = 0
    for cyc in CYCLES:
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            violations += 1
    return violations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Deterministic construction of 15 colors
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    
    # Partition vertices into 4 groups of 4
    # Group k: vertices 4k, 4k+1, 4k+2, 4k+3
    
    # Intra-group coloring
    # Use 3 colors per group: 3k, 3k+1, 3k+2
    for k in range(4):
        base = 3 * k
        v0, v1, v2, v3 = 4*k, 4*k+1, 4*k+2, 4*k+3
        
        # Edges in K4: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        # Pattern:
        # 01: base
        # 02: base+1
        # 03: base+2
        # 12: base
        # 13: base+1
        # 23: base+2
        
        colors[_edge_index(v0, v1)] = base
        colors[_edge_index(v0, v2)] = base + 1
        colors[_edge_index(v0, v3)] = base + 2
        colors[_edge_index(v1, v2)] = base
        colors[_edge_index(v1, v3)] = base + 1
        colors[_edge_index(v2, v3)] = base + 2
        
    # Inter-group coloring
    # Use 3 colors: 12, 13, 14
    # Pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # Assign:
    # (0,1) -> 12
    # (0,2) -> 13
    # (0,3) -> 14
    # (1,2) -> 12
    # (1,3) -> 13
    # (2,3) -> 14
    
    inter_colors = {
        (0, 1): 12,
        (0, 2): 13,
        (0, 3): 14,
        (1, 2): 12,
        (1, 3): 13,
        (2, 3): 14
    }
    
    for i in range(4):
        for j in range(i + 1, 4):
            c = inter_colors[(i, j)]
            for u in range(4*i, 4*i+4):
                for v in range(4*j, 4*j+4):
                    colors[_edge_index(u, v)] = c
                    
    best_colors = colors.copy()
    best_distinct = len(set(colors.tolist()))
    
    # Try to improve with local search
    # If we can find > 15 colors, great.
    
    cycle_is_violation = np.zeros(NUM_CYCLES, dtype=bool)
    total_violations = _count_violations(colors)
    
    if total_violations > 0:
        # Should not happen for this construction
        return [0] * NUM_EDGES

    def _try_change(e, new_color, colors_arr):
        old_color = colors_arr[e]
        if old_color == new_color:
            return False
        
        for ci in EDGE_TO_CYCLES[e]:
            cyc = CYCLES[ci]
            e0, e1, e2, e3 = cyc
            oc0, oc1, oc2, oc3 = colors_arr[e0], colors_arr[e1], colors_arr[e2], colors_arr[e3]
            
            nc0, nc1, nc2, nc3 = oc0, oc1, oc2, oc3
            if e0 == e:
                nc0 = new_color
            elif e1 == e:
                nc1 = new_color
            elif e2 == e:
                nc2 = new_color
            elif e3 == e:
                nc3 = new_color
            
            if not (len({oc0, oc1, oc2, oc3}) == 4) and (len({nc0, nc1, nc2, nc3}) == 4):
                return False
        
        colors_arr[e] = new_color
        return True

    # Local search to try to increase diversity
    # We are at 15. Can we get 16?
    # Try to introduce color 15, 16, ...
    
    used_colors = set(colors.tolist())
    
    for iter in range(5000):
        if time.time() > deadline:
            break
            
        # Try to add a new color
        new_c = random.randint(15, MAX_COLOR)
        if new_c in used_colors:
            continue
            
        # Try to assign new_c to a random edge
        e = random.randint(0, NUM_EDGES - 1)
        if _try_change(e, new_c, colors):
            used_colors.add(new_c)
            current_distinct = len(used_colors)
            if current_distinct > best_distinct:
                best_distinct = current_distinct
                best_colors = colors.copy()
                
            # If we improved, maybe we can continue
            # But we must ensure validity is maintained (it is, by _try_change)
            
            # Try to fix any issues? _try_change ensures no new violations.
            # But does it maintain global validity?
            # Yes, it only rejects moves that create violations.
            # Since we started valid, we remain valid.
            
    if best_distinct > 15:
        return best_colors.tolist()
    else:
        return colors.tolist()
