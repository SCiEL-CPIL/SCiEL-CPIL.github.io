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
# A 4-cycle is defined by 4 vertices a<b<c<d.
# The 3 cycles are:
# 1. a-b-c-d-a  (edges: ab, bc, cd, da)
# 2. a-b-d-c-a  (edges: ab, bd, dc, ca)
# 3. a-c-b-d-a  (edges: ac, cb, bd, da)
CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, c)]
                e3 = EDGE_IDX[(c, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))
                
                # Cycle 2
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, d)]
                e3 = EDGE_IDX[(d, c)]
                e4 = EDGE_IDX[(c, a)]
                CYCLES.append((e1, e2, e3, e4))
                
                # Cycle 3
                e1 = EDGE_IDX[(a, c)]
                e2 = EDGE_IDX[(c, b)]
                e3 = EDGE_IDX[(b, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))

NUM_CYCLES = len(CYCLES)

# For each edge, list of tuples of the other 3 edges in the cycles it belongs to
OTHER_EDGES = [[] for _ in range(NUM_EDGES)]
for cyc in CYCLES:
    e1, e2, e3, e4 = cyc
    OTHER_EDGES[e1].append((e2, e3, e4))
    OTHER_EDGES[e2].append((e1, e3, e4))
    OTHER_EDGES[e3].append((e1, e2, e4))
    OTHER_EDGES[e4].append((e1, e2, e3))

def is_valid_change(colors, edge_idx, new_color):
    """
    Check if setting colors[edge_idx] to new_color results in no rainbow 4-cycles.
    Assumes the current coloring is valid.
    """
    if colors[edge_idx] == new_color:
        return True
        
    for o1, o2, o3 in OTHER_EDGES[edge_idx]:
        c1 = colors[o1]
        c2 = colors[o2]
        c3 = colors[o3]
        
        # If any of the other 3 edges share a color, they can't form a rainbow with edge_idx
        # even if new_color is distinct from them.
        # Wait, the condition for a rainbow is that ALL 4 are distinct.
        # So if c1, c2, c3 are not all distinct, the cycle cannot be a rainbow regardless of new_color.
        if c1 == c2 or c1 == c3 or c2 == c3:
            continue
            
        # If new_color matches any of c1, c2, c3, the cycle is not a rainbow.
        if new_color == c1 or new_color == c2 or new_color == c3:
            continue
            
        # If we reach here, c1, c2, c3 are distinct, and new_color is distinct from all three.
        # Thus, the 4 colors are distinct -> Rainbow!
        return False
        
    return True

def is_valid_swap(colors, e1, e2):
    """
    Check if swapping colors[e1] and colors[e2] results in no rainbow 4-cycles.
    """
    c1_old = colors[e1]
    c2_old = colors[e2]
    
    if c1_old == c2_old:
        return True

    c1_new = c2_old
    c2_new = c1_old

    # Check cycles involving e1
    for o1, o2, o3 in OTHER_EDGES[e1]:
        # Determine colors of o1, o2, o3 in the new state
        # If o_i is e2, its color becomes c2_new
        # Otherwise, its color remains colors[o_i]
        
        nc1 = c1_new
        
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
            
        if nc1 != nc_o1 and nc1 != nc_o2 and nc1 != nc_o3 and nc_o1 != nc_o2 and nc_o1 != nc_o3 and nc_o2 != nc_o3:
            return False

    # Check cycles involving e2 (and not e1)
    # Note: Cycles containing both e1 and e2 were checked in the first loop.
    # We must check cycles containing e2 but NOT e1.
    # To avoid double work and errors, we can just iterate all OTHER_EDGES[e2]
    # and skip the ones that contain e1? 
    # Actually, if a cycle contains both e1 and e2, it was already validated in the first loop.
    # Re-validating it in the second loop is safe but redundant.
    # However, we must be careful: in the second loop, for a cycle containing both,
    # the logic "if o_i == e1" will correctly set the color.
    # So just iterating all is fine and safer.
    
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
    deadline = start_time + 55.0  # Keep margin
    
    # Initial Coloring
    # Strategy: Greedy introduction of colors
    colors = [0] * NUM_EDGES
    rng = random.Random(42)
    
    # Greedy Phase: Try to maximize distinct colors
    # We maintain a set of used colors.
    # We pick random edges and try to assign them a color from a pool.
    # To encourage new colors, we try colors in the range [1, 20] first, then random.
    
    # A better greedy:
    # 1. All 0.
    # 2. For each edge, try to change to a new color if valid.
    # This is O(120 * 120 * 273) which is too slow for a simple loop?
    # 120 edges * 120 colors * 273 cycles = ~4 million checks. Each check is fast.
    # It should run in < 1s.
    
    edges_to_try = list(range(NUM_EDGES))
    rng.shuffle(edges_to_try)
    
    used_colors = {0}
    
    for e in edges_to_try:
        # Try to find a valid color for e that is not in used_colors, or any valid color
        # We want to maximize distinct colors, so prefer new colors.
        
        # Try a few random new colors
        found_new = False
        for _ in range(50):
            c = rng.randint(1, MAX_COLOR)
            if c in used_colors:
                continue
            if is_valid_change(colors, e, c):
                colors[e] = c
                used_colors.add(c)
                found_new = True
                break
        
        if not found_new:
            # Try random existing colors (might allow other edges to take new colors later?)
            # Or just leave it.
            pass
            
    current_score = len(used_colors)
    best_colors = list(colors)
    best_score = current_score
    
    # Simulated Annealing
    T = 5.0
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() < deadline:
        # Move selection
        r = rng.random()
        
        if r < 0.6:
            # Recolor
            e = rng.randint(0, NUM_EDGES - 1)
            # Bias towards unused colors?
            # 50% chance pick from used, 50% pick from unused
            if rng.random() < 0.5:
                c = rng.choice(list(used_colors))
            else:
                c = rng.randint(1, MAX_COLOR)
                
            if c == colors[e]:
                continue
                
            if is_valid_change(colors, e, c):
                old_c = colors[e]
                colors[e] = c
                
                # Update score
                # If old_c was the only instance, it disappears.
                # If c was new, it appears.
                # Simple: len(set(colors))
                new_score = len(set(colors))
                
                delta = new_score - current_score
                
                if delta > 0 or rng.random() < np.exp(delta / T):
                    current_score = new_score
                    # Update used_colors set? 
                    # For efficiency in move selection, we could maintain it, 
                    # but len(set()) is fast enough for 120 items.
                    if current_score > best_score:
                        best_score = current_score
                        best_colors = list(colors)
                else:
                    colors[e] = old_c
        else:
            # Swap
            e1 = rng.randint(0, NUM_EDGES - 1)
            e2 = rng.randint(0, NUM_EDGES - 1)
            if e1 == e2:
                continue
                
            if is_valid_swap(colors, e1, e2):
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
                    colors[e2] = colors[e1]
                    colors[e1] = temp

        # Cooling
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_colors
