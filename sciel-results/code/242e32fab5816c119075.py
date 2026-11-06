"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge indices
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2

# Precompute all 4-cycles: for each 4-vertex subset, the 3 distinct 4-cycles
# A 4-cycle on vertices {a,b,c,d} can be: a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
# Edges in each:
# Cycle 1: (a,b), (b,c), (c,d), (d,a)
# Cycle 2: (a,b), (b,d), (d,c), (c,a)
# Cycle 3: (a,c), (c,b), (b,d), (d,a)

def precompute_cycles():
    """Return list of 4-cycles, each as a tuple of 4 edge indices."""
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        c1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
        # Cycle 2: a-b-d-c-a
        c2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
        # Cycle 3: a-c-b-d-a
        c3 = (edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a))
        cycles.append(c1)
        cycles.append(c2)
        cycles.append(c3)
    return cycles

CYCLES = precompute_cycles()  # 1199 cycles
NUM_CYCLES = len(CYCLES)

# For each edge, precompute which cycles it participates in
EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cyc in enumerate(CYCLES):
    for e in cyc:
        EDGE_CYCLES[e].append(ci)

def count_colors(colors):
    return len(set(colors))

def is_rainbow(cyc, colors):
    """Check if a 4-cycle has 4 distinct colors."""
    c = [colors[e] for e in cyc]
    return len(set(c)) == 4

def check_valid(colors):
    """Check if coloring has no rainbow 4-cycle."""
    for cyc in CYCLES:
        if is_rainbow(cyc, colors):
            return False
    return True

def greedy_construct(seed=0):
    """Greedy: order edges by cycle count descending, assign unused color if possible."""
    random.seed(seed)
    colors = [-1] * NUM_EDGES
    used_colors = set()
    
    # Order edges by number of cycles they're in (more constrained first)
    edge_order = sorted(range(NUM_EDGES), key=lambda e: -len(EDGE_CYCLES[e]))
    
    for e in edge_order:
        # Try to assign an unused color first
        best_color = None
        # Try unused colors
        for c in range(120):
            if c in used_colors:
                continue
            colors[e] = c
            # Check all cycles containing this edge
            valid = True
            for ci in EDGE_CYCLES[e]:
                cyc = CYCLES[ci]
                # Check if this cycle would be rainbow
                col_set = set()
                all_assigned = True
                for ee in cyc:
                    if colors[ee] == -1:
                        all_assigned = False
                        break
                    col_set.add(colors[ee])
                if all_assigned and len(col_set) == 4:
                    valid = False
                    break
            if valid:
                best_color = c
                break
            colors[e] = -1
        
        if best_color is None:
            # Try used colors
            for c in sorted(used_colors):
                colors[e] = c
                valid = True
                for ci in EDGE_CYCLES[e]:
                    cyc = CYCLES[ci]
                    col_set = set()
                    all_assigned = True
                    for ee in cyc:
                        if colors[ee] == -1:
                            all_assigned = False
                            break
                        col_set.add(colors[ee])
                    if all_assigned and len(col_set) == 4:
                        valid = False
                        break
                if valid:
                    best_color = c
                    break
                colors[e] = -1
        
        if best_color is None:
            # Fallback: assign 0
            best_color = 0
            colors[e] = 0
        else:
            colors[e] = best_color
            used_colors.add(best_color)
    
    return colors

def local_search(colors, max_iter=5000, deadline=None):
    """Try to improve by reassigning edges to new colors."""
    colors = list(colors)
    best_colors = list(colors)
    best_count = count_colors(colors)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Try unused colors first
        used = set(colors)
        unused = [c for c in range(120) if c not in used]
        random.shuffle(unused)
        
        found = False
        for c in unused:
            colors[e] = c
            valid = True
            for ci in EDGE_CYCLES[e]:
                cyc = CYCLES[ci]
                col_set = set(colors[ee] for ee in cyc)
                if len(col_set) == 4:
                    valid = False
                    break
            if valid:
                found = True
                break
            colors[e] = -1
        
        if not found:
            # Try used colors (different from old)
            for c in used:
                if c == old_color:
                    continue
                colors[e] = c
                valid = True
                for ci in EDGE_CYCLES[e]:
                    cyc = CYCLES[ci]
                    col_set = set(colors[ee] for ee in cyc)
                    if len(col_set) == 4:
                        valid = False
                        break
                if valid:
                    found = True
                    break
            colors[e] = old_color
        
        if found:
            new_count = count_colors(colors)
            if new_count > best_count:
                best_count = new_count
                best_colors = list(colors)
            improved = True
        else:
            colors[e] = old_color
    
    return best_colors, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    # Phase 1: Greedy construction with multiple seeds
    best_colors = None
    best_count = 0
    
    for seed in range(50):
        if time.time() > deadline:
            break
        colors = greedy_construct(seed)
        if not check_valid(colors):
            # Fix invalid cycles
            for _ in range(100):
                if time.time() > deadline:
                    break
                fixed = False
                for cyc in CYCLES:
                    if is_rainbow(cyc, colors):
                        # Pick an edge in this cycle to recolor
                        for e in cyc:
                            old_c = colors[e]
                            for c in range(120):
                                if c == old_c:
                                    continue
                                colors[e] = c
                                # Check if this fixes the cycle without breaking others
                                valid_cycle = not is_rainbow(cyc, colors)
                                if valid_cycle:
                                    # Check no new rainbow cycles
                                    ok = True
                                    for ci in EDGE_CYCLES[e]:
                                        if is_rainbow(CYCLES[ci], colors):
                                            ok = False
                                            break
                                    if ok:
                                        fixed = True
                                        break
                            if fixed:
                                break
                        if fixed:
                            break
                if not fixed:
                    break
            if not check_valid(colors):
                continue
        c = count_colors(colors)
        if c > best_count:
            best_count = c
            best_colors = list(colors)
    
    if best_colors is None:
        return [0] * NUM_EDGES
    
    # Phase 2: Local search to improve
    best_colors, best_count = local_search(best_colors, max_iter=20000, deadline=deadline)
    
    # Phase 3: More greedy restarts if time permits
    for seed in range(50, 200):
        if time.time() > deadline:
            break
        colors = greedy_construct(seed)
        if not check_valid(colors):
            continue
        c = count_colors(colors)
        if c > best_count:
            best_count = c
            best_colors = list(colors)
            # Quick local search
            best_colors, best_count = local_search(best_colors, max_iter=5000, deadline=deadline)
    
    return best_colors
