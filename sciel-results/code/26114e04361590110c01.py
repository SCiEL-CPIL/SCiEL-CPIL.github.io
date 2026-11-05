import time
import math

def construct():
    start_time = time.time()
    limit_time = 55.0  # Safety margin for 60s limit
    
    # 1. Preprocess: Identify valid vertices (2*v is not a square)
    # and build conflict graph.
    N_orig = 256
    valid_indices = []
    for v in range(1, N_orig + 1):
        s = 2 * v
        r = int(math.isqrt(s))
        if r * r != s:
            valid_indices.append(v)
    
    N = len(valid_indices)
    # Map original value to index in valid_indices
    val_to_idx = {v: i for i, v in enumerate(valid_indices)}
    
    # Build adjacency bitmasks
    # adj[i] is a bitmask where bit j is 1 if valid_indices[i] + valid_indices[j] is a square
    adj = [0] * N
    
    # Precompute squares up to 256+256=512
    squares = set()
    for k in range(1, int(math.isqrt(512)) + 1):
        squares.add(k * k)
        
    for i in range(N):
        vi = valid_indices[i]
        mask = 0
        for j in range(N):
            if i == j:
                continue
            if vi + valid_indices[j] in squares:
                mask |= (1 << j)
        adj[i] = mask
        
    # Initial candidates: all valid vertices
    # We want to find MIS.
    
    best_set = []
    best_size = 0
    
    # Heuristic: Sort vertices by degree descending for branching?
    # We'll pick vertices dynamically.
    
    def bit_count(x):
        return x.bit_count()
        
    def greedy_color_bound(candidates_mask):
        # Returns an upper bound on the size of the MIS in the subgraph induced by candidates_mask
        # using greedy coloring.
        # We need to iterate through vertices in candidates_mask.
        # To make it faster, we can just count colors.
        # Colors are integers.
        # color[v] = color assigned to v
        # We iterate through bits in candidates_mask.
        
        # Optimization: If candidates_mask is 0, return 0.
        if candidates_mask == 0:
            return 0
            
        # Simple greedy coloring
        # We assign colors 0, 1, 2...
        # A vertex can take a color if no neighbor in candidates_mask has that color.
        
        # To speed up, we can maintain a list of available colors?
        # Or just track used colors for neighbors.
        
        # Since N is small (~200), O(N^2) is fine.
        
        colors = {}
        max_color = 0
        c = candidates_mask
        while c:
            # Pick a vertex
            # Let's pick the one with lowest index for determinism, or highest degree?
            # For bound calculation, order doesn't change the *validity* of the bound,
            # but affects the *tightness*.
            # Standard greedy: pick vertex, assign smallest color not used by neighbors.
            
            # Get index of lowest set bit
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            
            # Find neighbors of v in the remaining candidates (including those already processed? No, just check all neighbors in candidates_mask)
            # Actually, we just need to know which colors are used by neighbors of v that are IN candidates_mask.
            # It is sufficient to check neighbors of v.
            
            neighbor_colors = set()
            n_mask = adj[v] & candidates_mask
            while n_mask:
                n_lsb = n_mask & -n_mask
                n_v = n_lsb.bit_length() - 1
                n_mask ^= n_lsb
                if n_v in colors:
                    neighbor_colors.add(colors[n_v])
            
            # Assign smallest available color
            col = 0
            while col in neighbor_colors:
                col += 1
            
            colors[v] = col
            if col > max_color:
                max_color = col
            
        return max_color + 1 # Number of colors
        
    # Branch and Bound
    # State: current_mask (candidates), current_size, current_set (list of indices)
    
    # We use a stack for DFS
    # (candidates_mask, current_size, current_set_indices)
    
    # To improve pruning, we should order branching vertices by degree.
    # Let's pre-compute degrees.
    degrees = [bit_count(adj[i]) for i in range(N)]
    
    # Sort indices by degree descending?
    # We can just pick the vertex with highest degree in the current candidate set at each step.
    
    stack = [( (1 << N) - 1, 0, [] )]
    
    while stack:
        if time.time() - start_time > limit_time:
            break
            
        c_mask, c_size, c_set = stack.pop()
        
        # Pruning: Simple bound
        if c_size + bit_count(c_mask) <= best_size:
            continue
            
        # Pruning: Coloring bound
        # This is expensive. Only do it if simple bound doesn't prune?
        # Or do it always?
        # Let's try simple bound first.
        
        if c_mask == 0:
            if c_size > best_size:
                best_size = c_size
                best_set = c_set[:]
            continue
            
        # Heuristic: Pick vertex with highest degree in c_mask
        # To find max degree vertex in c_mask:
        # Iterate bits?
        
        # Optimization: If we just pick any vertex, it's slower.
        # Let's pick the vertex with the most neighbors in c_mask.
        
        # Since N is small, we can find it quickly.
        # But doing this for every node might be slow.
        # Let's just pick the lowest index vertex in c_mask for now, or pre-sorted order?
        # Pre-sorted order by static degree is a good heuristic.
        
        # Let's use static degree order.
        # We need to find the first vertex in c_mask that is in the sorted order.
        # This is tricky with bitmasks.
        
        # Alternative: Just pick the lowest set bit.
        # This is fast.
        
        lsb = c_mask & -c_mask
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        # New candidates: c_mask & ~(adj[v] | (1 << v))
        new_mask_include = c_mask & ~(adj[v] | (1 << v))
        new_size_include = c_size + 1
        new_set_include = c_set + [v]
        
        # Branch 2: Exclude v
        new_mask_exclude = c_mask & ~(1 << v)
        new_size_exclude = c_size
        new_set_exclude = c_set
        
        # Push to stack.
        # We want to explore the "Include" branch first if it looks promising?
        # Usually, including high-degree vertices prunes more.
        
        # Push Exclude first, then Include, so Include is popped first (LIFO).
        stack.append((new_mask_exclude, new_size_exclude, new_set_exclude))
        stack.append((new_mask_include, new_size_include, new_set_include))
        
    # Convert best_set indices back to values
    result = [valid_indices[i] for i in best_set]
    
    # Ensure non-empty (task requires non-empty)
    # If best_set is empty, we might need to handle that, but there should always be a solution.
    # The smallest square is 1, 4, 9...
    # 1+1=2 (not square), 1+2=3 (not square)...
    # 1 is valid? 2*1=2 not square.
    # 1+1=2 not square.
    # So {1} is a valid set.
    
    return result
