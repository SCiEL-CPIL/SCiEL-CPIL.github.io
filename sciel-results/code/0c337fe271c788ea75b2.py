"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2

# Precompute edge index for (i,j) with i<j
_edge_idx = {}
_ei = 0
for i in range(N):
    for j in range(i + 1, N):
        _edge_idx[(i, j)] = _ei
        _ei += 1

def _ei(i, j):
    if i > j:
        i, j = j, i
    return _edge_idx[(i, j)]

# Precompute all 4-cycles (each set of 4 vertices has 3 Hamiltonian 4-cycles)
# For vertices a<b<c<d, the 3 4-cycles (as edge sets) are:
# Cycle 1: a-b-c-d-a  -> edges (a,b),(b,c),(c,d),(d,a)
# Cycle 2: a-b-d-c-a  -> edges (a,b),(b,d),(d,c),(c,a)
# Cycle 3: a-c-b-d-a  -> edges (a,c),(c,b),(b,d),(d,a)
all_cycles = []
for a, b, c, d in combinations(range(N), 4):
    # Cycle 1
    all_cycles.append((_ei(a,b), _ei(b,c), _ei(c,d), _ei(d,a)))
    # Cycle 2
    all_cycles.append((_ei(a,b), _ei(b,d), _ei(d,c), _ei(c,a)))
    # Cycle 3
    all_cycles.append((_ei(a,c), _ei(c,b), _ei(b,d), _ei(d,a)))

NUM_CYCLES = len(all_cycles)

# For each edge, which cycles does it belong to?
edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for ci, cyc in enumerate(all_cycles):
    for e in cyc:
        edge_to_cycles[e].append(ci)

def _count_colors(colors):
    return len(set(colors))

def _violations_of_edge(colors, e):
    """Count how many 4-cycles containing edge e are rainbow (4 distinct colors)."""
    c = colors[e]
    v = 0
    for ci in edge_to_cycles[e]:
        a, b, d, f = all_cycles[ci]
        # The 4 edges in this cycle
        ca = colors[a]; cb = colors[b]; cd = colors[d]; cf = colors[f]
        if len({c, ca, cb, cd, cf}) > 4:  # can't happen but safety
            continue
        # Count distinct colors among the 4 edges
        # The 4 edges are a,b,d,f (where one of them is e)
        # Actually the 4 edges in the cycle are all_cycles[ci]
        e1, e2, e3, e4 = all_cycles[ci]
        vals = {colors[e1], colors[e2], colors[e3], colors[e4]}
        if len(vals) == 4:
            v += 1
    return v

def _total_violations(colors):
    v = 0
    for cyc in all_cycles:
        if len({colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]}) == 4:
            v += 1
    return v

def construct():
    DEADLINE = time.time() + 50.0  # safety margin (90s limit, 50s work)
    random.seed(42)
    
    best_colors = [0] * NUM_EDGES
    best_score = 0
    
    def _score(colors):
        # Higher is better: maximize colors used, minimize violations
        used = len(set(colors))
        viol = _total_violations(colors)
        return (used, -viol)
    
    def _is_valid(colors):
        for cyc in all_cycles:
            if len({colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]}) == 4:
                return False
        return True
    
    # === Phase 1: Greedy constructive with randomization ===
    for trial in range(50):
        if time.time() > DEADLINE:
            break
        random.shuffle(list(range(120)))
        colors = [-1] * NUM_EDGES
        used = set()
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        feasible = True
        for e in edge_order:
            # Try new colors first (unassigned), then old
            tried_new = False
            # Pick a random new color
            new_candidates = [c for c in range(120) if c not in used]
            random.shuffle(new_candidates)
            old_candidates = [c for c in range(120) if c in used]
            random.shuffle(old_candidates)
            
            placed = False
            for c in new_candidates + old_candidates:
                colors[e] = c
                # Quick check: only cycles involving this edge (and all other edges assigned)
                ok = True
                for ci in edge_to_cycles[e]:
                    cyc = all_cycles[ci]
                    # Check if all 4 edges are assigned
                    if all(colors[x] != -1 for x in cyc):
                        if len({colors[x] for x in cyc}) == 4:
                            ok = False
                            break
                if ok:
                    used.add(c)
                    placed = True
                    break
            if not placed:
                colors[e] = -1
                feasible = False
                break
        
        if feasible:
            s = _score(colors)
            if s > _score(best_colors):
                best_colors = colors[:]
                best_score = s
    
    # === Phase 2: Local search to improve (reduce violations, then increase colors) ===
    cur = best_colors[:]
    for trial in range(20):
        if time.time() > DEADLINE:
            break
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            # Try recoloring each edge
            for e in range(NUM_EDGES):
                if time.time() > DEADLINE:
                    break
                old_c = cur[e]
                current_score = _score(cur)
                # Try all 120 colors
                candidates = list(range(120))
                random.shuffle(candidates)
                best_local = None
                best_local_c = None
                for c in candidates:
                    if c == old_c:
                        continue
                    cur[e] = c
                    s = _score(cur)
                    if s > current_score:
                        best_local = s
                        best_local_c = c
                    elif best_local is None and s == current_score:
                        # Keep as tie-breaker
                        pass
                    cur[e] = old_c
                if best_local_c is not None:
                    cur[e] = best_local_c
                    improved = True
        
        s = _score(cur)
        if s > _score(best_colors):
            best_colors = cur[:]
            best_score = s
    
    # === Phase 3: Ensure validity - if not valid, fall back to single color ===
    if not _is_valid(best_colors):
        # Try to fix by reducing to valid
        # Simple approach: if too many violations, use a structured coloring
        best_colors = [0] * NUM_EDGES
    
    return best_colors

# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END
