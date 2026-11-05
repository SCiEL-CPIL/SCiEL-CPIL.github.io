import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 256
    # 1. Filter out vertices where 2*u is a perfect square
    forbidden = set()
    for m in range(1, 20):
        val = 2 * m * m
        if val <= N:
            forbidden.add(val)
    
    candidates = [u for u in range(1, N + 1) if u not in forbidden]
    n = len(candidates)
    
    # 2. Build Compatibility Graph H (bitsets)
    # H[i][j] = 1 if candidates[i] + candidates[j] is NOT a perfect square
    max_sum = 2 * N
    squares = set()
    for k in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(k * k)
    
    adj_bits = [0] * n
    for i in range(n):
        ui = candidates[i]
        for j in range(i + 1, n):
            uj = candidates[j]
            if (ui + uj) not in squares:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    best_clique_size = 0
    best_clique = []
    
    # Greedy Coloring Bound
    def get_color_bound(cand_list, cand_mask):
        if not cand_list:
            return 0, []
            
        # Calculate degrees in the induced subgraph
        # To speed up, we can just use the order of cand_list or sort by degree.
        # Sorting by degree is better for the bound.
        
        # Compute degrees
        degs = {}
        for v in cand_list:
            degs[v] = (adj_bits[v] & cand_mask).bit_count()
            
        # Sort by degree descending
        sorted_verts = sorted(cand_list, key=lambda x: degs[x], reverse=True)
        
        colors = {}
        max_color = 0
        
        # Greedy Coloring
        for v in sorted_verts:
            used_colors = set()
            # Check neighbors in cand_list that are already colored
            # To optimize, we can iterate through neighbors of v that are in cand_list
            # But checking all cand_list is O(k).
            # Since k is small, it's fine.
            for u in cand_list:
                if u == v:
                    continue
                if u in colors:
                    if (adj_bits[v] >> u) & 1:
                        used_colors.add(colors[u])
                        
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
                
        # Order vertices by color
        # Vertices with lower colors are processed first?
        # In the search, we will pick vertices from this ordered list.
        ordered = sorted(cand_list, key=lambda x: colors[x])
        
        return max_color, ordered

    def search(cand_list, cand_mask, current_size, current_clique):
        nonlocal best_clique_size, best_clique
        nonlocal deadline
        
        if time.time() > deadline:
            return
            
        if not cand_list:
            if current_size > best_clique_size:
                best_clique_size = current_size
                best_clique = current_clique.copy()
            return
            
        # Pruning with simple bound
        if current_size + len(cand_list) <= best_clique_size:
            return
            
        # Pruning with coloring bound
        bound, ordered_list = get_color_bound(cand_list, cand_mask)
        if current_size + bound <= best_clique_size:
            return
            
        # Branch
        # Pick the first vertex in the ordered list
        v = ordered_list[0]
        
        # Branch 1: Include v
        # New candidates are neighbors of v in the current candidate set
        new_mask = cand_mask & adj_bits[v]
        new_mask &= ~(1 << v)
        
        # Filter list
        new_list = [u for u in cand_list if (new_mask >> u) & 1]
        
        search(new_list, new_mask, current_size + 1, current_clique + [v])
        
        # Branch 2: Exclude v
        new_mask_ex = cand_mask & ~(1 << v)
        new_list_ex = [u for u in cand_list if u != v]
        
        search(new_list_ex, new_mask_ex, current_size, current_clique)
        
    # Initial Call
    initial_list = list(range(n))
    initial_mask = (1 << n) - 1
    
    search(initial_list, initial_mask, 0, [])
    
    return best_clique_size
