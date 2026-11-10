import time
import random
import sys
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

NUM_CYCLES = len(CYCLES)  # 5460

# For each edge, which cycles does it belong to?
OTHER_EDGES = [[] for _ in range(NUM_EDGES)]
for cyc in CYCLES:
    e1, e2, e3, e4 = cyc
    OTHER_EDGES[e1].append((e2, e3, e4))
    OTHER_EDGES[e2].append((e1, e3, e4))
    OTHER_EDGES[e3].append((e1, e2, e4))
    OTHER_EDGES[e4].append((e1, e2, e3))


def count_violations(colors):
    """Count number of 4-cycles with 4 distinct edge colors."""
    violations = 0
    for e1, e2, e3, e4 in CYCLES:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            violations += 1
    return violations


def check_edge_change(colors, edge_idx, new_color):
    """
    Check if changing colors[edge_idx] to new_color creates a rainbow cycle.
    Returns True if valid, False otherwise.
    """
    if colors[edge_idx] == new_color:
        return True
    
    for o1, o2, o3 in OTHER_EDGES[edge_idx]:
        c1 = colors[o1]
        c2 = colors[o2]
        c3 = colors[o3]
        
        # If any two of the other colors are the same, it can't be a rainbow cycle
        if c1 == c2 or c2 == c3 or c1 == c3:
            continue
            
        # If the new color matches any of the other colors, it's not a rainbow cycle
        if new_color == c1 or new_color == c2 or new_color == c3:
            continue
            
        # If we get here, all 4 colors are distinct
        return False
        
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin
    
    # Initial valid coloring: all 0
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    used_colors = {0}
    current_score = 1
    
    # Best found
    best_colors = colors.copy()
    best_score = 1
    
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    all_colors = list(range(MAX_COLOR + 1))
    
    # Simulated annealing parameters
    T_initial = 10.0
    T_final = 0.1
    cooling_rate = 0.9995
    T = T_initial
    
    iterations = 0
    
    while time.time() < deadline and T > T_final:
        iterations += 1
        
        # 1. Try to add a new unused color (greedy)
        if len(used_colors) < MAX_COLOR + 1:
            # Pick a random edge
            e = rng.randint(0, NUM_EDGES - 1)
            
            # Pick a random unused color
            unused = [c for c in all_colors if c not in used_colors]
            if unused:
                new_c = rng.choice(unused)
                if check_edge_change(colors, e, new_c):
                    colors[e] = new_c
                    used_colors.add(new_c)
                    current_score = len(used_colors)
                    if current_score > best_score:
                        best_score = current_score
                        best_colors = colors.copy()
                    continue
        
        # 2. Simulated annealing move: recolor an edge
        e = rng.randint(0, NUM_EDGES - 1)
        old_c = int(colors[e])
        
        # Pick a random color (biased toward existing colors for stability)
        if len(used_colors) > 1 and rng.random() < 0.7:
            candidates = [c for c in used_colors if c != old_c]
            if candidates:
                new_c = rng.choice(candidates)
            else:
                new_c = rng.randint(0, MAX_COLOR)
        else:
            new_c = rng.randint(0, MAX_COLOR)
        
        if new_c == old_c:
            continue
        
        # Check validity
        if check_edge_change(colors, e, new_c):
            colors[e] = new_c
            used_colors.add(new_c)
            current_score = len(used_colors)
            
            if current_score > best_score:
                best_score = current_score
                best_colors = colors.copy()
        else:
            # Invalid move: accept with probability exp(-1/T) to escape local optima
            if rng.random() < np.exp(-1.0 / T):
                # Accept invalid move temporarily (we'll fix it later or revert)
                # For simplicity, just revert
                pass
        
        # Cool down
        T *= cooling_rate
        
        # Periodic restart: if stuck for a while, try a different strategy
        if iterations % 500 == 0 and len(used_colors) < best_score:
            # Revert to best and try again
            colors = best_colors.copy()
            # Recompute used_colors from best_colors
            used_colors = set(int(c) for c in best_colors)
            current_score = len(used_colors)
            T = max(T, T_initial * 0.5)  # Partial restart of temperature
    
    # Final safety check
    if count_violations(best_colors.tolist()) != 0:
        # Fallback to a known valid coloring
        best_colors = np.zeros(NUM_EDGES, dtype=np.int32)
        best_score = 1

    return list(best_colors.astype(int))

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# We need to make sure the interface matches.
# The task requires: construct() -> 120 integer colors
# The code above provides that.
# EVOLVE-BLOCK-END
