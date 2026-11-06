import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

def get_edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute edge list
edges = []
edge_to_vertices = {}
for i in range(N):
    for j in range(i + 1, N):
        idx = get_edge_index(i, j)
        edges.append((i, j))
        edge_to_vertices[idx] = (i, j)

# Precompute all 4-cycles
# A 4-cycle is defined by 4 vertices {a,b,c,d}.
# There are 3 distinct 4-cycles on these 4 vertices:
# 1. a-b-c-d-a (edges: ab, bc, cd, da)
# 2. a-b-d-c-a (edges: ab, bd, dc, ca)
# 3. a-c-b-d-a (edges: ac, cb, bd, da)
# Note: The edges involved in any 4-cycle on {a,b,c,d} are 4 out of the 6 possible edges.
# The "missing" edges form a matching.
# For each set of 4 vertices, the 3 cycles correspond to the 3 ways to partition 4 vertices into 2 pairs (matchings) that are NOT in the cycle? 
# Actually, let's just enumerate the 3 permutations of the 4th vertex relative to the first 3?
# Simpler: For vertices a<b<c<d, the 3 cycles are:
# C1: (a,b), (b,c), (c,d), (d,a)
# C2: (a,b), (b,d), (d,c), (c,a)
# C3: (a,c), (c,b), (b,d), (d,a)

cycles = []
edge_to_cycles = [[] for _ in range(NUM_EDGES)]

for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d-a
                e1 = get_edge_index(a, b)
                e2 = get_edge_index(b, c)
                e3 = get_edge_index(c, d)
                e4 = get_edge_index(d, a)
                cyc1 = (e1, e2, e3, e4)
                cycles.append(cyc1)
                for e in cyc1:
                    edge_to_cycles[e].append(len(cycles) - 1)
                
                # Cycle 2: a-b-d-c-a
                e1 = get_edge_index(a, b)
                e2 = get_edge_index(b, d)
                e3 = get_edge_index(d, c)
                e4 = get_edge_index(c, a)
                cyc2 = (e1, e2, e3, e4)
                cycles.append(cyc2)
                for e in cyc2:
                    edge_to_cycles[e].append(len(cycles) - 1)
                
                # Cycle 3: a-c-b-d-a
                e1 = get_edge_index(a, c)
                e2 = get_edge_index(c, b)
                e3 = get_edge_index(b, d)
                e4 = get_edge_index(d, a)
                cyc3 = (e1, e2, e3, e4)
                cycles.append(cyc3)
                for e in cyc3:
                    edge_to_cycles[e].append(len(cycles) - 1)

# Convert cycles to numpy array for faster access if needed, but list of tuples is fine for 5460 items
# Actually, we will store colors in a numpy array for fast indexing

def is_valid_coloring(colors):
    """Check if the entire coloring is valid. Used for initial checks or debugging."""
    for cyc in cycles:
        c0 = colors[cyc[0]]
        c1 = colors[cyc[1]]
        c2 = colors[cyc[2]]
        c3 = colors[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initialize with a valid coloring
    # Start with all 0s
    colors = np.zeros(NUM_EDGES, dtype=int)
    current_diversity = 1
    best_colors = colors.copy()
    best_diversity = 1
    
    # Precompute cycle edge indices as numpy arrays for potential vectorization? 
    # No, the check is "are 4 values distinct". 
    # We can precompute the list of cycles for each edge.
    
    # To speed up the local search, we can maintain a count of how many times each color is used?
    # No, the constraint is local. We just need to check the 182 cycles per move.
    
    # Optimization: 
    # When we change color of edge e from c_old to c_new, we only need to check cycles containing e.
    # For each such cycle, the set of colors changes.
    # Old set: {c_old, c_b, c_c, c_d}
    # New set: {c_new, c_b, c_c, c_d}
    # We can precompute the "other 3 edges" for each edge in each cycle?
    # Or just fetch the 4 colors from the array.
    
    # Let's try a simple hill climbing with random restarts.
    
    num_restarts = 0
    max_restarts = 200
    
    while time.time() < deadline:
        # If we hit a local optimum, restart
        if num_restarts > max_restarts:
            break
            
        # Local search phase
        stagnation = 0
        max_stagnation = 5000
        
        while time.time() < deadline and stagnation < max_stagnation:
            # Pick a random edge
            e = random.randint(0, NUM_EDGES - 1)
            c_old = colors[e]
            
            # Pick a new color. 
            # Strategy: Pick a random color from 0..119.
            # Bias towards colors not yet used to increase diversity?
            # Or just random.
            c_new = random.randint(0, MAX_COLOR - 1)
            
            if c_new == c_old:
                continue
                
            # Check validity
            valid = True
            for cyc_idx in edge_to_cycles[e]:
                cyc = cycles[cyc_idx]
                # Get the other 3 edges
                # We need to be careful: e is in cyc.
                # Find the position of e in cyc? Or just get the 4 values.
                c0 = colors[cyc[0]]
                c1 = colors[cyc[1]]
                c2 = colors[cyc[2]]
                c3 = colors[cyc[3]]
                
                # Update the one that corresponds to e
                # This is inefficient if we don't know the index.
                # Let's precompute for each edge e, and for each cycle it belongs to, the indices of the other 3 edges.
                
                if len({c0, c1, c2, c3}) == 4:
                    # Wait, this checks the CURRENT state.
                    # The current state is valid.
                    # We are proposing a change.
                    # We need to check if the NEW state is valid.
                    # The current state is valid, so no 4-cycle has 4 distinct colors.
                    # Changing one edge color might break this.
                    # Specifically, if the cycle had 3 distinct colors, and we change one to a 4th distinct color, it becomes invalid.
                    # If the cycle had 4 distinct colors, it was already invalid (impossible in our valid state).
                    # So, a valid state -> change edge e -> invalid state happens ONLY IF:
                    # 1. The cycle currently has 3 distinct colors.
                    # 2. The color we are removing (c_old) is the "redundant" one? No.
                    #    If colors are {A, A, B, C}, and we change one A to D (new), we get {A, D, B, C} -> 4 distinct. Invalid.
                    #    If colors are {A, A, B, C}, and we change B to A, we get {A, A, A, C} -> 2 distinct. Valid.
                    #    If colors are {A, B, B, C}, and we change B to D, we get {A, D, B, C} -> 4 distinct. Invalid.
                    
                    # So we must check:
                    # For each cycle containing e:
                    #   Let other_colors be the colors of the other 3 edges.
                    #   Let new_set = {c_new} | other_colors
                    #   If len(new_set) == 4, then invalid.
                    
                    # We need to efficiently get the other 3 colors.
                    pass
            
            # The above loop is too slow if we don't precompute the "other edges".
            # Let's precompute `cycle_neighbors[e]` as a list of tuples (cyc_idx, idx_of_e_in_cyc, other_3_edges).
            # Actually, simpler: Precompute for each edge e, a list of lists of 3 edge indices.
            # `other_edges_for_cycle[e][k]` = the 3 edges in the k-th cycle containing e, excluding e itself.
            
            # Since we are in the "thinking" phase, let's implement this precomputation now.
            break # Break inner loop to implement precomputation properly in the final code
            
        # Restart
        colors = np.zeros(NUM_EDGES, dtype=int)
        # Maybe randomize initial colors slightly? 
        # All 0 is valid. 
        # Random valid coloring:
        # We can try to assign random colors and fix conflicts, but that's hard.
        # Start with all 0, then randomize.
        num_restarts += 1

    # The above logic was flawed in the draft. I will write the clean version below.
    return best_colors.tolist()

# RE-IMPLEMENTATION WITH PRECOMPUTATION

# Precompute for each edge e, the list of (cycle_index, position_of_e, other_3_edge_indices)
# Actually, we just need the other 3 edge indices.
# Let's create `other_edges[e]` as a list of 3-tuples.
# There are 182 cycles per edge. So 182 tuples per edge.
# Total memory: 120 * 182 * 3 integers. Small.

# We will do this in the module scope to avoid re-computation on each call if the module is reloaded?
# The task says "Keep the function names...". It doesn't say we can't have module-level precomputation.
# However, if the evaluator imports the module multiple times, it might take time.
# 5460 cycles is small. Precomputation is fast.

def _precompute():
    global cycles, edge_to_cycles, other_edges
    cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = get_edge_index(a, b)
                    e2 = get_edge_index(b, c)
                    e3 = get_edge_index(c, d)
                    e4 = get_edge_index(d, a)
                    cyc1 = (e1, e2, e3, e4)
                    idx = len(cycles)
                    cycles.append(cyc1)
                    for i, e in enumerate(cyc1):
                        edge_to_cycles[e].append(idx)
                        
                    # Cycle 2: a-b-d-c-a
                    e1 = get_edge_index(a, b)
                    e2 = get_edge_index(b, d)
                    e3 = get_edge_index(d, c)
                    e4 = get_edge_index(c, a)
                    cyc2 = (e1, e2, e3, e4)
                    idx = len(cycles)
                    cycles.append(cyc2)
                    for i, e in enumerate(cyc2):
                        edge_to_cycles[e].append(idx)
                        
                    # Cycle 3: a-c-b-d-a
                    e1 = get_edge_index(a, c)
                    e2 = get_edge_index(c, b)
                    e3 = get_edge_index(b, d)
                    e4 = get_edge_index(d, a)
                    cyc3 = (e1, e2, e3, e4)
                    idx = len(cycles)
                    cycles.append(cyc3)
                    for i, e in enumerate(cyc3):
                        edge_to_cycles[e].append(idx)
    
    # Precompute other edges
    other_edges = [[] for _ in range(NUM_EDGES)]
    for e in range(NUM_EDGES):
        for cyc_idx in edge_to_cycles[e]:
            cyc = cycles[cyc_idx]
            # Find index of e in cyc
            # cyc is a tuple of 4.
            # Since e is in cyc, we can just remove it.
            others = []
            for ce in cyc:
                if ce != e:
                    others.append(ce)
            other_edges[e].append(tuple(others))

_precompute()

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initialize
    colors = np.zeros(NUM_EDGES, dtype=int)
    current_diversity = 1
    best_colors = colors.copy()
    best_diversity = 1
    
    # Local Search
    # We will use a simple hill climbing.
    # If stuck, restart.
    
    while time.time() < deadline:
        stagnation = 0
        max_stagnation = 2000 # Lower for more restarts
        
        while time.time() < deadline and stagnation < max_stagnation:
            e = random.randint(0, NUM_EDGES - 1)
            c_old = colors[e]
            c_new = random.randint(0, MAX_COLOR - 1)
            
            if c_new == c_old:
                continue
            
            # Check validity
            valid = True
            for others in other_edges[e]:
                # Get colors of other 3 edges
                c0 = colors[others[0]]
                c1 = colors[others[1]]
                c2 = colors[others[2]]
                
                # New set: {c_new, c0, c1, c2}
                if len({c_new, c0, c1, c2}) == 4:
                    valid = False
                    break
            
            if valid:
                colors[e] = c_new
                # Update diversity
                # We can maintain a count array? 
                # Or just compute set size. 120 items is fast.
                new_div = len(set(colors))
                if new_div > current_diversity:
                    current_diversity = new_div
                    if current_diversity > best_diversity:
                        best_diversity = current_diversity
                        best_colors = colors.copy()
                    stagnation = 0
                else:
                    # Accepting a move that doesn't increase diversity might help escape local minima?
                    # In hill climbing for max diversity, we usually only accept if it improves or is neutral?
                    # If we accept neutral, we might wander. 
                    # Let's accept neutral moves with some probability? Or just accept all valid moves?
                    # If we accept all valid moves, we might cycle.
                    # Let's accept valid moves. 
                    stagnation += 1
            else:
                stagnation += 1
        
        # Restart
        # Randomize initial state?
        # Start with all 0 is safe.
        # Try a random valid coloring?
        # We can generate a random coloring and fix it? No, fixing is hard.
        # Just restart with all 0 and let it explore.
        # Or, restart with the best_colors found so far? That's not a restart, that's continuing.
        # True restart: All 0.
        colors = np.zeros(NUM_EDGES, dtype=int)
        current_diversity = 1

    return best_colors.tolist()
