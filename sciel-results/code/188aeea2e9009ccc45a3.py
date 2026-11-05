import time
import random
import sys

sys.setrecursionlimit(10000)

_N = 20
_COLORS = 20
_ALL_EDGES = [(i, j) for i in range(_N) for j in range(i + 1, _N)]
_EDGE_IDX = {(i, j): idx for idx, (i, j) in enumerate(_ALL_EDGES)}
_N_EDGES = len(_ALL_EDGES)

# Precompute all 4-cycles
_CYCLES = []
_seen_cycles = set()

for i in range(_N):
    for j in range(i + 1, _N):
        for k in range(j + 1, _N):
            for l in range(k + 1, _N):
                e1 = _EDGE_IDX[(i, j)]
                e2 = _EDGE_IDX[(j, k)]
                e3 = _EDGE_IDX[(k, l)]
                e4 = _EDGE_IDX[(l, i)]
                cycle1 = tuple(sorted([e1, e2, e3, e4]))
                if cycle1 not in _seen_cycles:
                    _seen_cycles.add(cycle1)
                    _CYCLES.append(cycle1)

                e1 = _EDGE_IDX[(i, j)]
                e2 = _EDGE_IDX[(j, l)]
                e3 = _EDGE_IDX[(l, k)]
                e4 = _EDGE_IDX[(k, i)]
                cycle2 = tuple(sorted([e1, e2, e3, e4]))
                if cycle2 not in _seen_cycles:
                    _seen_cycles.add(cycle2)
                    _CYCLES.append(cycle2)

                e1 = _EDGE_IDX[(i, k)]
                e2 = _EDGE_IDX[(k, j)]
                e3 = _EDGE_IDX[(j, l)]
                e4 = _EDGE_IDX[(l, i)]
                cycle3 = tuple(sorted([e1, e2, e3, e4]))
                if cycle3 not in _seen_cycles:
                    _seen_cycles.add(cycle3)
                    _CYCLES.append(cycle3)

# Build adjacency from cycles
_N_CYCLES = len(_CYCLES)
_EDGE_TO_CYCLES = [[] for _ in range(_N_EDGES)]
for c_idx, cycle in enumerate(_CYCLES):
    for e_idx in cycle:
        _EDGE_TO_CYCLES[e_idx].append(c_idx)

# Count occurrences of each edge in cycles (for tie-breaking)
_EDGE_CYCLE_COUNT = [len(c) for c in _EDGE_TO_CYCLES]

def _check_edge_color(e_idx, color, present_edges, colors):
    """Check if assigning 'color' to edge e_idx is valid given current graph."""
    for c_idx in _EDGE_TO_CYCLES[e_idx]:
        cycle = _CYCLES[c_idx]
        # Need all other 3 edges present
        found_all = True
        others = []
        for e in cycle:
            if e == e_idx:
                continue
            if e in present_edges:
                others.append(e)
            else:
                found_all = False
                break
        if found_all and len(others) == 3:
            if colors[others[0]] == color or colors[others[1]] == color or colors[others[2]] == color:
                return False
    return True

def _run_greedy(seed, deadline):
    """Greedy edge selection with color assignment, then local search."""
    rng = random.Random(seed)
    present = set()
    colors = {}
    
    # Order edges: prefer higher degree / less constrained
    order = list(range(_N_EDGES))
    rng.shuffle(order)
    
    for e_idx in order:
        if time.time() > deadline:
            break
        # Try colors in random order
        col_order = list(range(_COLORS))
        rng.shuffle(col_order)
        for c in col_order:
            if _check_edge_color(e_idx, c, present, colors):
                present.add(e_idx)
                colors[e_idx] = c
                break
    
    # Local search: try to add more edges
    # Build list of missing edges
    missing = [e for e in range(_N_EDGES) if e not in present]
    rng.shuffle(missing)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for e_idx in missing[:]:
            if time.time() > deadline:
                break
            # Try to add this edge
            for c in range(_COLORS):
                if _check_edge_color(e_idx, c, present, colors):
                    present.add(e_idx)
                    colors[e_idx] = c
                    missing.remove(e_idx)
                    improved = True
                    break
    
    # Recoloring local search
    # Try recoloring existing edges to unlock new additions
    edge_list = list(present)
    rng.shuffle(edge_list)
    
    for iter_ in range(10):
        if time.time() > deadline:
            break
        found_change = False
        for e_idx in edge_list:
            if time.time() > deadline:
                break
            old_color = colors[e_idx]
            # Try new color
            for new_c in range(_COLORS):
                if new_c == old_color:
                    continue
                # Check if recoloring is valid
                valid = True
                for c_idx in _EDGE_TO_CYCLES[e_idx]:
                    cycle = _CYCLES[c_idx]
                    others = [e for e in cycle if e != e_idx and e in present]
                    if len(others) == 3:
                        if new_c == colors[others[0]] or new_c == colors[others[1]] or new_c == colors[others[2]]:
                            valid = False
                            break
                if valid:
                    colors[e_idx] = new_c
                    found_change = True
                    break
        
        # After some recoloring, try adding more edges
        missing_now = [e for e in range(_N_EDGES) if e not in present]
        rng.shuffle(missing_now)
        for e_idx in missing_now[:20]:
            if time.time() > deadline:
                break
            for c in range(_COLORS):
                if _check_edge_color(e_idx, c, present, colors):
                    present.add(e_idx)
                    colors[e_idx] = c
                    break
    
    return len(present), [( _ALL_EDGES[e][0], _ALL_EDGES[e][1], colors[e]) for e in present]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_count = 0
    best_solution = []
    
    # Run multiple greedy restarts with different seeds
    seed = 42
    while time.time() < deadline:
        count, sol = _run_greedy(seed, deadline)
        if count > best_count:
            best_count = count
            best_solution = sol
        seed += 1
    
    return [[u, v, c] for u, v, c in best_solution]

