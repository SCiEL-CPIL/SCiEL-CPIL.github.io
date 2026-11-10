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

NUM_CYCLES = len(CYCLES)  # 5460

# For each edge, which cycles does it belong to?
# OTHER_EDGES[e] = list of tuples (o1, o2, o3)
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
        
        if c1 == c2 or c2 == c3 or c1 == c3:
            continue
            
        if new_color == c1 or new_color == c2 or new_color == c3:
            continue
            
        return False
        
    return True

def check_swap(colors, e1, e2):
    """
    Check if swapping colors[e1] and colors[e2] creates a rainbow cycle.
    Returns True if valid, False otherwise.
    """
    c1_old = colors[e1]
    c2_old = colors[e2]
    
    if c1_old == c2_old:
        return True

    # New colors
    c1_new = c2_old
    c2_new = c1_old

    # Check cycles involving e1
    for o1, o2, o3 in OTHER_EDGES[e1]:
        # Determine new colors for o1, o2, o3
        # If o_i is e2, its new color is c2_new (which is c1_old)
        # Otherwise, its new color is colors[o_i]
        
        nc1 = c1_new
        # Get new colors for neighbors
        if o1 == e2:
            nc_o1 = c2_new
        else:
            nc_o1 = colors[o1]
            
        if o2 == e2:
            nc_o2 = c2_new
        else:
            nc_o2 = colors[o2]
            
        if o3 == e2:
            nc_o3 = c2_new
        else:
            nc_o3 = colors[o3]
            
        # Check if {nc1, nc_o1, nc_o2, nc_o3} are distinct
        if nc1 != nc_o1 and nc1 != nc_o2 and nc1 != nc_o3 and nc_o1 != nc_o2 and nc_o1 != nc_o3 and nc_o2 != nc_o3:
            return False

    # Check cycles involving e2 (and not e1, to avoid redundant work? 
    # Actually, if a cycle has both e1 and e2, it was checked in the first loop.
    # If it has e2 but not e1, we need to check it here.
    # The list OTHER_EDGES[e2] contains all cycles with e2.
    # We can just iterate all of them. The ones with e1 were already checked.
    # Checking them again is safe (just redundant).
    for o1, o2, o3 in OTHER_EDGES[e2]:
        nc1 = c2_new
        if o1 == e1:
            nc_o1 = c1_new
        else:
            nc_o1 = colors[o1]
            
        if o2 == e1:
            nc_o2 = c1_new
        else:
            nc_o2 = colors[o2]
            
        if o3 == e1:
            nc_o3 = c1_new
        else:
            nc_o3 = colors[o3]
            
        if nc1 != nc_o1 and nc1 != nc_o2 and nc1 != nc_o3 and nc_o1 != nc_o2 and nc_o1 != nc_o3 and nc_o2 != nc_o3:
            return False

    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_colors = [0] * NUM_EDGES
    best_score = 1
    
    # Simulated Annealing
    # State: colors
    # Score: distinct colors count
    # Constraint: No rainbow cycles
    
    rng = random.Random(42)
    
    # Initialize with all 0s
    colors = [0] * NUM_EDGES
    current_score = 1
    
    T = 1.0
    T_min = 1e-5
    cooling = 0.999
    
    steps = 0
    
    while time.time() < deadline:
        steps += 1
        if steps % 10000 == 0:
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Generate move
        move_type = rng.random()
        
        if move_type < 0.5:
            # Recolor
            e = rng.randint(0, NUM_EDGES - 1)
            # Bias towards unused or less used colors?
            # Just random for now
            c = rng.randint(0, MAX_COLOR)
            
            if check_edge_change(colors, e, c):
                old_c = colors[e]
                colors[e] = c
                
                # Update score
                # We need to know if old_c is still used and if c was already used.
                # For efficiency in SA, we might just calculate score from scratch occasionally?
                # Or maintain counts.
                # Let's maintain counts.
                # But SA needs to rollback.
                # Easier: Calculate score after move.
                # len(set(colors)) is O(120). 120 ops is very fast.
                new_score = len(set(colors))
                
                delta = new_score - current_score
                
                if delta > 0 or rng.random() < np.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_colors = list(colors)
                else:
                    # Reject
                    colors[e] = old_c
            else:
                # Invalid move, reject
                pass
        else:
            # Swap
            e1 = rng.randint(0, NUM_EDGES - 1)
            e2 = rng.randint(0, NUM_EDGES - 1)
            if e1 == e2:
                continue
                
            if check_swap(colors, e1, e2):
                # Apply swap
                temp = colors[e1]
                colors[e1] = colors[e2]
                colors[e2] = temp
                
                new_score = len(set(colors))
                delta = new_score - current_score
                
                if delta > 0 or rng.random() < np.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_colors = list(colors)
                else:
                    # Reject
                    colors[e2] = colors[e1]
                    colors[e1] = temp
            else:
                pass

    # Final safety check
    if count_violations(best_colors) != 0:
        best_colors = [0] * NUM_EDGES
        best_score = 1

    return list(best_colors)
