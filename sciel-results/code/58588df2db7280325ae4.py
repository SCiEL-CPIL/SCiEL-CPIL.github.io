import random
import time
import itertools

def count_distinct_colors(colors):
    return len(set(colors))

def is_valid(coloring):
    # coloring is a list of 120 ints
    # vertices 0..15
    # edges (i,j)
    # map (i,j) to index
    # idx = i * 16 - i * (i+1) / 2 + (j - i - 1) ? No, simpler:
    # Let's create a 16x16 matrix for fast access
    
    # Pre-compute K4s
    # This is done once outside or cached
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    n = 16
    num_edges = n * (n - 1) // 2
    num_k4 = 1820
    
    # Precompute K4s
    k4s = []
    for a, b, c, d in itertools.combinations(range(n), 4):
        # Edges: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
        # Indices in the flattened list
        # We need a fast way to get edge index from (u,v)
        # idx(u,v) where u < v
        # Sum of lengths of previous rows: u*(n-1) - u*(u-1)/2 + (v-u-1)
        # Let's just use a dict or a 2D array
        pass
        
    # Better: create a 2D list edge_color[u][v]
    
    # Initial random coloring with 15 colors
    max_colors = 15
    best_coloring = None
    best_max_color = max_colors
    
    # Random initialization
    coloring = [random.randint(0, max_colors - 1) for _ in range(num_edges)]
    
    # Helper to get edge index
    def get_idx(u, v):
        if u > v: u, v = v, u
        # Number of edges before row u
        # Edges in rows 0..u-1: sum_{k=0}^{u-1} (n-1-k)
        # = u*(n-1) - u*(u-1)//2
        # Edges before v in row u: v - u - 1
        return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)
    
    # Precompute edge indices for each K4
    k4_edges = []
    for a, b, c, d in itertools.combinations(range(n), 4):
        edges = [
            get_idx(a, b), get_idx(a, c), get_idx(a, d),
            get_idx(b, c), get_idx(b, d), get_idx(c, d)
        ]
        k4_edges.append(edges)
        
    def check_valid(coloring):
        for edges in k4_edges:
            c0 = coloring[edges[0]]
            c1 = coloring[edges[1]]
            c2 = coloring[edges[2]]
            c3 = coloring[edges[3]]
            c4 = coloring[edges[4]]
            c5 = coloring[edges[5]]
            # Check if at least 5 distinct
            # Simple check: if all same, 1. If 2 same, etc.
            # Fast check:
            # If c0==c1 and c0==c2 and c0==c3 and c0==c4 and c0==c5: return False (1 color)
            # This is 1820 * 6 ops.
            distinct = len({c0, c1, c2, c3, c4, c5})
            if distinct < 5:
                return False
        return True

    # Local Search
    # We will try to find a valid coloring with 15 colors, then try to reduce to 14, 13.
    
    # First, find a valid 15-coloring
    valid_found = False
    while time.time() < deadline - 5.0:
        if check_valid(coloring):
            valid_found = True
            best_coloring = coloring.copy()
            best_max_color = max_colors
            break
        # Pick a random K4, check if invalid
        # Actually, check_valid scans all. Let's find an invalid one.
        invalid_k4_idx = -1
        for i, edges in enumerate(k4_edges):
            c0 = coloring[edges[0]]
            c1 = coloring[edges[1]]
            c2 = coloring[edges[2]]
            c3 = coloring[edges[3]]
            c4 = coloring[edges[4]]
            c5 = coloring[edges[5]]
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                invalid_k4_idx = i
                break
        
        if invalid_k4_idx != -1:
            edges = k4_edges[invalid_k4_idx]
            # Recolor one of these edges
            # Try to change to a color that helps
            # Random choice
            e_idx = random.choice(edges)
            coloring[e_idx] = random.randint(0, max_colors - 1)
        else:
            # Should not happen if check_valid is consistent
            break
            
    if not valid_found:
        # Return the seed if we couldn't find a valid one quickly?
        # But seed is 17. We must return something valid.
        # The seed is always valid.
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

    # Now try to reduce colors
    # Current max color is best_max_color (15)
    # Try to eliminate color best_max_color - 1, then best_max_color - 2, etc.
    
    current_coloring = best_coloring.copy()
    current_max = best_max_color
    
    while time.time() < deadline - 2.0:
        # Try to eliminate the highest color
        color_to_eliminate = current_max - 1
        if color_to_eliminate < 0:
            break
            
        # Find all edges with this color
        edges_to_change = [i for i, c in enumerate(current_coloring) if c == color_to_eliminate]
        
        if not edges_to_change:
            current_max -= 1
            continue
            
        # Try to recolor these edges to colors 0..current_max-2
        # We can do this randomly or greedily.
        # Let's try a few random attempts
        success = False
        for _ in range(100):
            trial = current_coloring.copy()
            for e in edges_to_change:
                trial[e] = random.randint(0, current_max - 2)
            if check_valid(trial):
                success = True
                current_coloring = trial
                current_max -= 1
                break
                
        if not success:
            # Maybe try a different strategy or break
            # If we can't eliminate this color, we might be stuck at this max.
            # But we could try to swap colors or something more complex.
            # For now, if we fail, we stop reducing.
            break
            
    # Return the best found
    return current_coloring
