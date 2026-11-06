"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge indices
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

# Precompute all 4-cycles: for each 4-cycle, the 6 edges involved
# A 4-cycle on vertices (a,b,c,d) in order has edges:
# (a,b), (b,c), (c,d), (d,a) - the cycle edges
# (a,c), (b,d) - the diagonals
# But the constraint is: no 4-cycle (4 vertices in a cycle) has 4 distinct colors on its 4 cycle edges
# Actually, re-reading: "No four-cycle may have four distinct edge colors"
# A four-cycle is a cycle of length 4, i.e., 4 vertices v1,v2,v3,v4 with edges (v1,v2),(v2,v3),(v3,v4),(v4,v1)
# The constraint is that these 4 edges don't all have distinct colors.
# "all three cycles on every four-vertex subset are checked"
# For 4 vertices {a,b,c,d}, the three 4-cycles are:
# a-b-c-d-a (edges: ab, bc, cd, da)
# a-b-d-c-a (edges: ab, bd, dc, ca)  
# a-c-b-d-a (edges: ac, cb, bd, da)
# Wait, let me think more carefully. A 4-cycle on 4 vertices uses 4 of the 6 edges.
# The three ways to choose a 4-cycle from 4 vertices:
# The 6 edges form a K4. A 4-cycle in K4 omits 2 opposite edges.
# So the three 4-cycles correspond to omitting: {ac, bd}, {ab, cd}, {ad, bc}
# Cycle 1 (omit ac, bd): a-b, b-c, c-d, d-a
# Cycle 2 (omit ab, cd): a-c, c-b, b-d, d-a  -> edges: ac, bc, bd, da
# Cycle 3 (omit ad, bc): a-b, b-d, d-c, c-a  -> edges: ab, bd, cd, ca

def precompute_cycles():
    """Precompute all 4-cycles as lists of 4 edge indices."""
    cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    # Cycle 1: a-b-c-d-a, edges: ab, bc, cd, da
                    e1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)]
                    # Cycle 2: a-c-b-d-a, edges: ac, cb, bd, da
                    e2 = [edge_idx(a,c), edge_idx(b,c), edge_idx(b,d), edge_idx(d,a)]
                    # Cycle 3: a-b-d-c-a, edges: ab, bd, dc, ca
                    e3 = [edge_idx(a,b), edge_idx(b,d), edge_idx(c,d), edge_idx(a,c)]
                    cycles.append(e1)
                    cycles.append(e2)
                    cycles.append(e3)
    return cycles

# Precompute for each edge, which cycles it belongs to (for fast violation tracking)
def precompute_edge_cycle_map(cycles):
    """For each edge, list of (cycle_idx, position_in_cycle)."""
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for pos, e in enumerate(cycle):
            edge_to_cycles[e].append((ci, pos))
    return edge_to_cycles

# Global precomputation
CYCLES = precompute_cycles()
NUM_CYCLES = len(CYCLES)
EDGE_TO_CYCLES = precompute_edge_cycle_map(CYCLES)

def count_colors(colors):
    return len(set(colors))

def count_violations(colors):
    """Count 4-cycles with 4 distinct colors."""
    violations = 0
    for cycle in CYCLES:
        c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        if len({c0, c1, c2, c3}) == 4:
            violations += 1
    return violations

def is_valid(colors):
    return count_violations(colors) == 0

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # Leave margin before 90s limit
    rng = random.Random(42)
    
    best_colors = None
    best_num_colors = 0
    
    # Initial solution: try to build a valid coloring with many colors
    # Strategy: local search with random restarts
    
    # Precompute cycle edge arrays as numpy for speed
    cycles_arr = np.array(CYCLES, dtype=np.int32)  # shape (NUM_CYCLES, 4)
    
    def fast_violation_count(colors_np):
        """Count violations using numpy."""
        # colors_np[cycles_arr] gives shape (NUM_CYCLES, 4)
        cycle_colors = colors_np[cycles_arr]
        # Count distinct colors in each row
        # A row has 4 distinct colors iff all pairwise comparisons show differences
        # Faster: check if sorted row has all distinct
        # Or: use the fact that 4 values are distinct iff (v0!=v1)&(v0!=v2)&(v0!=v3)&(v1!=v2)&(v1!=v3)&(v2!=v3)
        v0, v1, v2, v3 = cycle_colors[:,0], cycle_colors[:,1], cycle_colors[:,2], cycle_colors[:,3]
        distinct = (v0!=v1) & (v0!=v2) & (v0!=v3) & (v1!=v2) & (v1!=v3) & (v2!=v3)
        return int(np.sum(distinct))
    
    def fast_num_colors(colors_np):
        return len(set(colors_np.tolist()))
    
    # Local search function
    def local_search(initial, max_steps, deadline):
        colors = np.array(initial, dtype=np.int32)
        cur_violations = fast_violation_count(colors)
        cur_num_colors = fast_num_colors(colors)
        
        best_local = colors.copy()
        best_local_nc = cur_num_colors
        
        if cur_violations == 0:
            best_local_nc = cur_num_colors
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Choose a move: change one edge's color
            e = rng.randint(0, NUM_EDGES - 1)
            old_color = colors[e]
            new_color = rng.randint(0, 119)
            if new_color == old_color:
                continue
            
            colors[e] = new_color
            
            # Compute new violations
            new_violations = fast_violation_count(colors)
            new_num_colors = fast_num_colors(colors)
            
            # Acceptance: prefer valid solutions with more colors
            # If current is valid, only accept if new is valid and has more colors
            # If current is invalid, accept if new has fewer violations, or same violations and more colors
            accept = False
            if cur_violations == 0:
                if new_violations == 0 and new_num_colors >= cur_num_colors:
                    accept = True
            else:
                if new_violations < cur_violations:
                    accept = True
                elif new_violations == cur_violations and new_num_colors > cur_num_colors:
                    accept = True
                elif new_violations == cur_violations and new_num_colors == cur_num_colors:
                    # Neutral move: accept with some probability to escape local optima
                    if rng.random() < 0.1:
                        accept = True
            
            if accept:
                cur_violations = new_violations
                cur_num_colors = new_num_colors
                if cur_violations == 0 and cur_num_colors > best_local_nc:
                    best_local_nc = cur_num_colors
                    best_local = colors.copy()
            else:
                colors[e] = old_color
        
        return best_local, best_local_nc
    
    # Multiple restarts with different initializations
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > DEADLINE:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Start with a simple valid coloring
            colors_init = [0] * NUM_EDGES
        elif restart == 1:
            # Start with 2 colors
            colors_init = [rng.randint(0, 1) for _ in range(NUM_EDGES)]
        elif restart == 2:
            # Start with 3 colors
            colors_init = [rng.randint(0, 2) for _ in range(NUM_EDGES)]
        elif restart % 3 == 0:
            # Random with limited colors
            k = rng.randint(2, 6)
            colors_init = [rng.randint(0, k-1) for _ in range(NUM_EDGES)]
        else:
            # Fully random
            colors_init = [rng.randint(0, 119) for _ in range(NUM_EDGES)]
        
        # Run local search
        max_steps = 5000
        colors, nc = local_search(colors_init, max_steps, DEADLINE)
        
        if nc > best_num_colors:
            best_num_colors = nc
            best_colors = colors
        
        # If we found a good solution, try to improve it further
        if best_num_colors >= 10:
            # Polish: more iterations on the best solution
            if time.time() < DEADLINE - 5:
                colors2, nc2 = local_search(best_colors.tolist(), 10000, DEADLINE)
                if nc2 > best_num_colors:
                    best_num_colors = nc2
                    best_colors = colors2
    
    if best_colors is None:
        return [0] * NUM_EDGES
    
    return best_colors.tolist()

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
