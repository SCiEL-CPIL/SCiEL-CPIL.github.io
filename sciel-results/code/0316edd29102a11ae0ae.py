"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 16
M = N * (N - 1) // 2  # 120 edges

# EVOLVE-BLOCK-START

def _edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def _build_cycles():
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        e1 = (_edge_index(a,b), _edge_index(b,c), _edge_index(c,d), _edge_index(d,a))
        e2 = (_edge_index(a,b), _edge_index(b,d), _edge_index(d,c), _edge_index(c,a))
        e3 = (_edge_index(a,c), _edge_index(c,b), _edge_index(b,d), _edge_index(d,a))
        cycles.append(e1)
        cycles.append(e2)
        cycles.append(e3)
    return cycles

_CYCLES = _build_cycles()
NUM_CYCLES = len(_CYCLES)

def _build_edge_to_cycles():
    edge_to_cycles = [[] for _ in range(M)]
    for ci, cyc in enumerate(_CYCLES):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

_EDGE_TO_CYCLES = _build_edge_to_cycles()

def _try_assign_color(colors, edge_idx, new_color, violations):
    affected = _EDGE_TO_CYCLES[edge_idx]
    old_color = colors[edge_idx]
    if old_color == new_color:
        return violations, []
    
    colors[edge_idx] = new_color
    
    new_violations = dict(violations)
    changed = []
    for ci in affected:
        cyc = _CYCLES[ci]
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        is_violation = (len({c0, c1, c2, c3}) == 4)
        if is_violation != new_violations.get(ci, False):
            new_violations[ci] = is_violation
            changed.append(ci)
    
    if any(new_violations.get(ci, False) for ci in affected):
        colors[edge_idx] = old_color
        return None, []
    
    return new_violations, changed

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    random.seed(42)
    colors = [0] * M
    violations = {i: False for i in range(NUM_CYCLES)}
    used_colors = {0}
    
    # Phase 1: Greedy assignment of new colors
    edge_order = sorted(range(M), key=lambda e: -len(_EDGE_TO_CYCLES[e]))
    color = 1
    for edge_idx in edge_order:
        if time.time() > deadline:
            break
        result = _try_assign_color(colors, edge_idx, color, violations)
        if result is not None:
            violations, _ = result
            used_colors.add(color)
            color += 1
        else:
            for alt_color in range(0, color):
                if alt_color == colors[edge_idx]:
                    continue
                result = _try_assign_color(colors, edge_idx, alt_color, violations)
                if result is not None:
                    violations, _ = result
                    break
    
    # Phase 2: Local search to introduce more colors
    for iteration in range(500):
        if time.time() > deadline:
            break
        progress = False
        for edge_idx in random.sample(range(M), min(120, M)):
            if time.time() > deadline:
                break
            current_color = colors[edge_idx]
            for new_color in range(len(used_colors) + 1, len(used_colors) + 15):
                result = _try_assign_color(colors, edge_idx, new_color, violations)
                if result is not None:
                    violations, _ = result
                    used_colors.add(new_color)
                    progress = True
                    break
            if progress:
                break
    
    return colors

# EVOLVE-BLOCK-END
