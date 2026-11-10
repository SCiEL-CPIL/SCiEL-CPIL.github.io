import itertools
import random
import time

# Precompute K4 edge indices
# Vertices 0..15
# Edges are pairs (i,j) with i<j
# Total edges 120

def get_k4_edges():
    k4s = []
    vertices = list(range(16))
    for quad in itertools.combinations(vertices, 4):
        edges = []
        for i in range(4):
            for j in range(i+1, 4):
                u = quad[i]
                v = quad[j]
                # Map (u,v) to linear edge index
                # Edge (i,j) index:
                # For fixed i, j goes from i+1 to 15.
                # Number of edges starting with k < i is sum_{k=0}^{i-1} (15-k)
                # Edges starting with i are j-i
                # Let's just use a dictionary or formula
                # Formula: idx(i, j) = i*(15-i+1)/2 + (j-i-1) ... no
                # Simpler: just generate a list of edge tuples and use a dict
        k4s.append(edges)
    return k4s

# Let's create a static mapping
edge_to_idx = {}
edge_list = []
idx = 0
for i in range(16):
    for j in range(i+1, 16):
        edge_to_idx[(i,j)] = idx
        edge_list.append((i,j))
        idx += 1

# Precompute K4 edge indices
K4_EDGE_INDICES = []
for quad in itertools.combinations(range(16), 4):
    edges = []
    for i in range(4):
        for j in range(i+1, 4):
            u, v = quad[i], quad[j]
            if u > v: u, v = v, u
            edges.append(edge_to_idx[(u,v)])
    K4_EDGE_INDICES.append(edges)

def construct():
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    best_colors = 17
    best_coloring = None
    
    # Initial best from seed
    # Seed: (i+j) % 17
    seed_coloring = [((i+j) % 17) for i in range(16) for j in range(i+1, 16)]
    
    def count_colors(coloring):
        return len(set(coloring))
    
    def is_valid_merge(coloring, c1, c2):
        # Check if merging c2 into c1 keeps all K4s valid
        # A K4 is invalid if it has < 5 distinct colors.
        # If a K4 has exactly 5 distinct colors and contains both c1 and c2, 
        # merging them reduces distinct count to 4.
        # If a K4 has < 5 distinct colors, it's already invalid (shouldn't happen in valid state).
        # So we just need to ensure no K4 has exactly 5 distinct colors containing both c1 and c2.
        
        # Optimization: Precompute color sets for all K4s for this coloring?
        # That takes O(N_K4 * 6) time.
        # Then checking pairs takes O(N_K4) per pair.
        # Total O(M^2 * N_K4).
        
        # Let's compute color sets for all K4s once per coloring state
        k4_color_sets = []
        for edges in K4_EDGE_INDICES:
            colors = set(coloring[e] for e in edges)
            k4_color_sets.append(colors)
        
        for colors in k4_color_sets:
            if len(colors) == 5 and c1 in colors and c2 in colors:
                return False
            # If len < 5, it's already invalid, but we assume input is valid.
            # If len > 5, merging two colors reduces count by at most 1, so >= 5.
        
        return True

    def greedy_merge(coloring):
        colors = list(set(coloring))
        # Map color to list of edges
        color_to_edges = {c: [] for c in colors}
        for i, c in enumerate(coloring):
            color_to_edges[c].append(i)
            
        improved = True
        while improved:
            improved = False
            # Try all pairs
            current_colors = list(color_to_edges.keys())
            for i in range(len(current_colors)):
                for j in range(i+1, len(current_colors)):
                    c1 = current_colors[i]
                    c2 = current_colors[j]
                    if is_valid_merge(coloring, c1, c2):
                        # Merge c2 into c1
                        for e in color_to_edges[c2]:
                            coloring[e] = c1
                        del color_to_edges[c2]
                        improved = True
                        # Restart pair search as colors changed
                        break
                if improved:
                    break
                    
        return len(set(coloring)), coloring

    # Run initial seed
    num_c, final_coloring = greedy_merge(seed_coloring.copy())
    best_colors = num_c
    best_coloring = final_coloring

    # If we already have 15 or less, we can stop or try to improve further
    # If best_colors > 15, try random permutations
    
    while time.time() - start_time < time_limit:
        # Random permutation
        perm = list(range(16))
        random.shuffle(perm)
        
        # Apply seed formula to permuted vertices
        # Original edge (i,j) in output corresponds to vertices i,j.
        # We want to color edge (i,j) based on perm[i] and perm[j].
        coloring = []
        for i in range(16):
            for j in range(i+1, 16):
                u = perm[i]
                v = perm[j]
                coloring.append((u + v) % 17)
                
        num_c, final_coloring = greedy_merge(coloring)
        
        if num_c < best_colors:
            best_colors = num_c
            best_coloring = final_coloring
            
            if best_colors <= 14:
                break # Lower bound is 13, 14 is great, 15 is target.
                
    # Return the best coloring found
    # The return value must be a list of 120 integers.
    # Ensure it's a list, not a numpy array or similar.
    return list(best_coloring)
