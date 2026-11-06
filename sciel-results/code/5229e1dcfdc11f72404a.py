"""Search for edge coloring of K_16 maximizing distinct colors with no rainbow 4-cycle."""
import numpy as np
import time
import itertools
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_cycles():
    """Return list of 4-tuples of edge indices, one per 4-cycle."""
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
        e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(c,d), edge_idx(a,c))
        e3 = (edge_idx(a,c), edge_idx(b,c), edge_idx(b,d), edge_idx(d,a))
        cycles.append(e1)
        cycles.append(e2)
        cycles.append(e3)
    return cycles

_CYCLES_LIST = build_cycles()
CYCLES = np.array(_CYCLES_LIST, dtype=np.int32)
NUM_CYCLES = len(CYCLES)

# For fast violation checking: for each edge, store which cycles it belongs to
_edge_cycles = [[] for _ in range(NUM_EDGES)]
for _ci in range(NUM_CYCLES):
    for _ei in CYCLES[_ci]:
        _edge_cycles[_ei].append(_ci)

def count_colors(colors):
    return len(set(colors))

def check_violations(colors):
    """Return number of violating cycles."""
    c = np.array(colors, dtype=np.int32)
    cyc = c[CYCLES]
    c0, c1, c2, c3 = cyc[:,0], cyc[:,1], cyc[:,2], cyc[:,3]
    bad = ((c0!=c1) & (c0!=c2) & (c0!=c3) & (c1!=c2) & (c1!=c3) & (c2!=c3))
    return int(bad.sum())

def is_valid(colors):
    return check_violations(colors) == 0

def initial_coloring(seed=0):
    """Generate structured initial coloring."""
    colors = [0] * NUM_EDGES
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            # Structured pattern
            colors[idx] = (i * 7 + j * 11) % 30
            idx += 1
    return colors

def fix_violations(colors, max_iters=20000, seed=0):
    """Fix violations by local search."""
    rng = random.Random(seed)
    colors = list(colors)
    violations = check_violations(colors)
    
    for _ in range(max_iters):
        if violations == 0:
            break
        c = np.array(colors, dtype=np.int32)
        cyc = c[CYCLES]
        c0, c1, c2, c3 = cyc[:,0], cyc[:,1], cyc[:,2], cyc[:,3]
        bad_mask = ((c0!=c1) & (c0!=c2) & (c0!=c3) & (c1!=c2) & (c1!=c3) & (c2!=c3))
        bad_indices = np.where(bad_mask)[0]
        if len(bad_indices) == 0:
            break
        ci = bad_indices[rng.randint(len(bad_indices))]
        edges = CYCLES[ci]
        ei = edges[rng.randint(4)]
        old_color = colors[ei]
        best_new = old_color
        best_v = violations
        for _ in range(15):
            new_color = rng.randint(0, MAX_COLOR)
            if new_color == old_color:
                continue
            colors[ei] = new_color
            v = check_violations(colors)
            if v < best_v:
                best_v = v
                best_new = new_color
            colors[ei] = old_color
        colors[ei] = best_new
        violations = best_v
    return colors

def sa_optimize(initial, time_budget, seed):
    """Simulated annealing to maximize distinct colors."""
    rng = random.Random(seed)
    colors = fix_violations(list(initial), max_iters=5000, seed=seed)
    if not is_valid(colors):
        # Force fix
        for _ in range(50):
            colors = fix_violations(colors, max_iters=5000, seed=seed+_+1)
            if is_valid(colors):
                break
    
    current = list(colors)
    current_score = count_colors(current)
    best = list(current)
    best_score = current_score
    
    start = time.time()
    deadline = start + time_budget
    temp = 3.0
    
    while time.time() < deadline - 2:
        ei = rng.randint(0, NUM_EDGES)
        old = current[ei]
        if rng.random() < 0.4:
            new = rng.randint(0, MAX_COLOR)
        else:
            new = (old + rng.randint(-5, 5)) % (MAX_COLOR + 1)
        if new == old:
            continue
        current[ei] = new
        if is_valid(current):
            sc = count_colors(current)
            delta = sc - current_score
            if delta >= 0 or rng.random() < np.exp(delta / max(temp, 0.1)):
                current_score = sc
                if sc > best_score:
                    best_score = sc
                    best = list(current)
            else:
                current[ei] = old
        else:
            current[ei] = old
        temp = max(temp * 0.9995, 0.1)
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_colors = None
    best_score = 0
    
    for seed in range(50):
        if time.time() > deadline - 15:
            break
        
        initial = initial_coloring(seed)
        fixed = fix_violations(initial, max_iters=10000, seed=seed)
        
        if is_valid(fixed):
            sc = count_colors(fixed)
            if sc > best_score:
                best_score = sc
                best_colors = list(fixed)
        
        if time.time() > deadline - 25:
            break
        
        remaining = deadline - time.time() - 10
        if remaining > 5:
            sa_colors, sa_score = sa_optimize(fixed, min(remaining, 8.0), seed)
            if is_valid(sa_colors) and sa_score > best_score:
                best_score = sa_score
                best_colors = list(sa_colors)
    
    if best_colors is None:
        # Fallback
        colors = [0] * NUM_EDGES
        return colors
    
    return best_colors
