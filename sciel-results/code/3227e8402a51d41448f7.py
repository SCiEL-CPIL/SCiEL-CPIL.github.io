import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

# Precompute edges in lexicographic order
EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
EDGE_IDX = {}
for idx, (i, j) in enumerate(EDGES):
    EDGE_IDX[(i, j)] = idx
    EDGE_IDX[(j, i)] = idx

# Precompute all 4-cycles and their edge indices
CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d-a
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, c)]
                e3 = EDGE_IDX[(c, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))
                # Cycle 2: a-b-d-c-a
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, d)]
                e3 = EDGE_IDX[(d, c)]
                e4 = EDGE_IDX[(c, a)]
                CYCLES.append((e1, e2, e3, e4))
                # Cycle 3: a-c-b-d-a
                e1 = EDGE_IDX[(a, c)]
                e2 = EDGE_IDX[(c, b)]
                e3 = EDGE_IDX[(b, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))

NUM_CYCLES = len(CYCLES)  # 3 * C(16,4) = 3 * 1820 = 5460

# For each edge, which cycles does it belong to?
EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cyc in enumerate(CYCLES):
    for ei in cyc:
        EDGE_TO_CYCLES[ei].append(ci)

# Precompute for each edge the other edges in each cycle it belongs to
# OTHER_EDGES[ei] = list of tuples of 3 edge indices (the other edges in each cycle containing ei)
OTHER_EDGES = [None] * NUM_EDGES
for e in range(NUM_EDGES):
    OTHER_EDGES[e] = []
    for cyc_idx in EDGE_TO_CYCLES[e]:
        cyc = CYCLES[cyc_idx]
        others = tuple(x for x in cyc if x != e)
        OTHER_EDGES[e].append(others)


def count_violations(colors):
    """Count number of 4-cycles with 4 distinct edge colors."""
    violations = 0
    for e1, e2, e3, e4 in CYCLES:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if len(set((c1, c2, c3, c4))) == 4:
            violations += 1
    return violations


def count_distinct_colors(colors):
    return len(set(colors))


def make_initial_coloring():
    """Create a valid coloring using sum-based coloring.
    Color of edge (i,j) = i+j (range 1..31).
    This guarantees the property because in any 4-cycle, diagonals share a color.
    """
    colors = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(EDGES):
        colors[idx] = i + j
    return colors


def check_edge_change(coloring, edge_idx, new_color):
    """
    Check if changing coloring[edge_idx] to new_color creates a rainbow cycle.
    Returns True if valid, False otherwise.
    """
    if coloring[edge_idx] == new_color:
        return True
    
    for others in OTHER_EDGES[edge_idx]:
        c1 = coloring[others[0]]
        c2 = coloring[others[1]]
        c3 = coloring[others[2]]
        
        # If any two of the other colors are the same, it can't be a rainbow cycle
        if c1 == c2 or c2 == c3 or c1 == c3:
            continue
            
        # If the new color matches any of the other colors, it's not a rainbow cycle
        if new_color == c1 or new_color == c2 or new_color == c3:
            continue
            
        # If we get here, all 4 colors are distinct
        return False
        
    return True


def greedy_construct(rng):
    """
    Greedily assign colors to edges to maximize distinct colors while maintaining validity.
    Start with all zeros, then try to assign new unused colors to random edges.
    """
    coloring = [0] * NUM_EDGES
    used_colors = {0}
    
    # Shuffle edges to break symmetry
    edges = list(range(NUM_EDGES))
    rng.shuffle(edges)
    
    for e in edges:
        # Try to assign a new unused color
        unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
        if not unused:
            break
        
        # Try a few random candidates
        candidates = rng.sample(unused, min(15, len(unused)))
        for c in candidates:
            if check_edge_change(coloring, e, c):
                coloring[e] = c
                used_colors.add(c)
                break
    
    return coloring, len(used_colors)


def local_search(coloring, rng, deadline, max_stuck=300):
    """
    Local search to increase the number of distinct colors while maintaining validity.
    """
    current_coloring = list(coloring)
    used_colors = set(current_coloring)
    current_score = len(used_colors)
    
    stuck_counter = 0
    max_stuck_val = max_stuck
    
    while time.time() < deadline - 1.0:
        if current_score == MAX_COLOR + 1:
            break
        
        # Pick a random edge
        idx = rng.randint(0, NUM_EDGES - 1)
        
        # Try to add a new color
        unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
        if unused:
            new_color = rng.choice(unused)
            if check_edge_change(current_coloring, idx, new_color):
                current_coloring[idx] = new_color
                used_colors.add(new_color)
                current_score += 1
                stuck_counter = 0
                continue
        
        # If we didn't add a color, try a random perturbation with existing colors
        stuck_counter += 1
        if stuck_counter > max_stuck_val:
            # Perform a random valid recoloring to escape local optima
            p_idx = rng.randint(0, NUM_EDGES - 1)
            p_old = current_coloring[p_idx]
            # Pick a random used color different from current
            other_colors = [c for c in used_colors if c != p_old]
            if other_colors:
                p_new = rng.choice(other_colors)
                if check_edge_change(current_coloring, p_idx, p_new):
                    current_coloring[p_idx] = p_new
                    # Update used_colors if needed
                    # (p_new is already in used_colors, p_old might still be used elsewhere)
                    stuck_counter = 0
                    max_stuck_val = int(max_stuck_val * 1.1)
    
    return current_coloring, current_score


def construct():
    """Return 120 integer colors in 0..119 for lexicographically ordered pairs (i,j), 0<=i<j<16.
    No four-cycle may have four distinct edge colors.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    rng = random.Random(42)
    
    # Start with the known valid sum-based coloring
    best_colors = make_initial_coloring()
    best_score = count_distinct_colors(best_colors)
    
    # Phase 1: Greedy construction + local search
    while time.time() < deadline - 3.0:
        coloring, score = greedy_construct(rng)
        if score > best_score:
            best_score = score
            best_colors = list(coloring)
        
        # Refine with local search
        refined, ref_score = local_search(coloring, rng, deadline)
        if ref_score > best_score:
            best_score = ref_score
            best_colors = list(refined)
        
        # If we've used all colors, we're done
        if best_score == MAX_COLOR + 1:
            break
        
        # Check if we should continue
        if time.time() > deadline - 5.0:
            break
    
    # Final safety check
    if count_violations(best_colors) != 0:
        best_colors = make_initial_coloring()
    
    return list(best_colors)
