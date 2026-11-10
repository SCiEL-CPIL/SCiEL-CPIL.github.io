"""Parametric algebraic colorings to avoid rainbow 4-cycles."""
import time
import itertools
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge index: edge_idx[i][j] for i < j
def _edge_idx():
    idx = [[-1] * N for _ in range(N)]
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx[i][j] = k
            idx[j][i] = k
            k += 1
    return idx

_EDGE = _edge_idx()

# Precompute all 4-cycles: each is a tuple of 4 edge indices (a-b, b-c, c-d, d-a)
def _all_4cycles():
    cycles = []
    # A 4-cycle is a set of 4 vertices with 3 possible cycle structures.
    # For each 4-vertex subset {a,b,c,d} with a<b<c<d, the three 4-cycles are:
    # 1. a-b-c-d-a  (edges: ab, bc, cd, da)
    # 2. a-b-d-c-a  (edges: ab, bd, dc, ca)
    # 3. a-c-b-d-a  (edges: ac, cb, bd, da)
    for a, b, c, d in itertools.combinations(range(N), 4):
        e1 = (_EDGE[a][b], _EDGE[b][c], _EDGE[c][d], _EDGE[d][a])
        e2 = (_EDGE[a][b], _EDGE[b][d], _EDGE[d][c], _EDGE[c][a])
        e3 = (_EDGE[a][c], _EDGE[c][b], _EDGE[b][d], _EDGE[d][a])
        cycles.append(e1)
        cycles.append(e2)
        cycles.append(e3)
    return cycles

_CYCLES = _all_4cycles()

def _is_valid(colors):
    """Check that no 4-cycle has 4 distinct edge colors."""
    for e in _CYCLES:
        c0, c1, c2, c3 = colors[e[0]], colors[e[1]], colors[e[2]], colors[e[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def _count_rainbows(colors):
    """Count the number of rainbow 4-cycles."""
    count = 0
    for e in _CYCLES:
        c0, c1, c2, c3 = colors[e[0]], colors[e[1]], colors[e[2]], colors[e[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            count += 1
    return count

def _random_repair(deadline):
    """Use random initialization and local repair to find valid colorings."""
    best_colors = [0] * NUM_EDGES
    best_distinct = 1
    
    # Initialize with a random assignment
    current_colors = [random.randint(0, 119) for _ in range(NUM_EDGES)]
    current_distinct = len(set(current_colors))
    
    # If random is already valid, keep it
    if _is_valid(current_colors):
        best_colors = current_colors
        best_distinct = current_distinct
    
    # Repair loop
    while time.time() < deadline:
        # Find a rainbow cycle
        rainbow_found = False
        for e in _CYCLES:
            c0, c1, c2, c3 = current_colors[e[0]], current_colors[e[1]], current_colors[e[2]], current_colors[e[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                # Change color of edge e[0] to match e[1]
                current_colors[e[0]] = current_colors[e[1]]
                rainbow_found = True
                break
        
        if not rainbow_found:
            # No more rainbows, check validity (should be valid)
            if _is_valid(current_colors):
                cd = len(set(current_colors))
                if cd > best_distinct:
                    best_colors = current_colors[:]
                    best_distinct = cd
            # Restart with a new random assignment to explore other local optima
            current_colors = [random.randint(0, 119) for _ in range(NUM_EDGES)]
            # Occasionally try a biased random to keep colors diverse?
            # For now, pure random restart is safe.
            
        # Check time
        if time.time() > deadline:
            break
            
    # Final check
    if _is_valid(best_colors):
        return best_colors
    return [0] * NUM_EDGES # Fallback: all 0s is always valid

def construct():
    start = time.time()
    deadline = start + 50.0  # 50s deadline, 10s safety margin
    
    # Use the random repair strategy
    result = _random_repair(deadline)
    
    # Ensure all colors are in range 0..119
    # The random generator uses 0..119, so this is guaranteed.
    
    return result
