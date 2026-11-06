"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    """
    Find a maximum cardinality subset of [1, 256] such that a+b is never a perfect square
    for any a <= b in the set (including a=b).
    
    Optimized Branch-and-Bound for Maximum Independent Set.
    """
    N = 256
    
    # Precompute perfect squares up to 2*256 = 512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    
    # Filter out self-conflicting nodes (where 2*x is a square)
    self_valid = []
    for x in range(1, N + 1):
        if 2 * x not in squares:
            self_valid.append(x)
    
    m = len(self_valid)
    
    # Build adjacency bitsets
    adj = [0] * m
    for i in range(m):
        a = self_valid[i]
        mask = 0
        for j in range(m):
            if i != j and (a + self_valid[j]) in squares:
                mask |= (1 << j)
        adj[i] = mask
        
    # Precompute global degrees and sort vertices by degree (descending)
    # This ordering helps the greedy coloring bound be tighter and faster
    global_deg = [bin(adj[i]).count('1') for i in range(m)]
    order = list(range(m))
    order.sort(key=lambda x: -global_deg[x])
    
    # Precompute bit masks for quick access? No, adj is already bitmask.
    
    full_mask = (1 << m) - 1
    
    # Greedy initial solution
    def greedy_solution(candidates_mask):
        sol = []
        remaining = candidates_mask
        # Use the global order to pick vertices, but check if they are in remaining
        # Actually, for greedy, picking highest degree in *remaining* is better.
        # But to keep it fast, we can just iterate through global order and pick if available?
        # No, greedy MIS usually picks max degree in current graph.
        # Let's stick to a simple greedy: pick first available in high-degree order?
        # That's not optimal. Let's do a proper greedy max-degree.
        while remaining:
            best_v = -1
            best_deg = -1
            # Iterate through remaining vertices
            # Optimization: iterate through 'order' and check if in remaining
            for v in order:
                if (remaining >> v) & 1:
                    deg = bin(remaining & adj[v]).count('1')
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
            if best_v == -1:
                break
            sol.append(best_v)
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
        return sol
    
    initial_sol = greedy_solution(full_mask)
    best_size = len(initial_sol)
    best_set = list(initial_sol)
    
    # Optimized color bound using global order
    def color_bound(mask):
        if mask == 0:
            return 0
        
        colors = {}
        max_color = 0
        
        # Iterate in global degree order
        for v in order:
            if not (mask >> v) & 1:
                continue
            
            # Find used colors among neighbors
            # We need to check neighbors u that are in mask and have been assigned a color
            # Since we process in 'order', any neighbor u processed earlier is in 'colors'
            used_colors = set()
            # Check neighbors of v
            # To speed this up, we can iterate through the bits of adj[v] & mask
            # But we only care about those in 'colors'
            # adj[v] is a bitmask.
            # Let's iterate through the set bits of (adj[v] & mask)
            # However, we only need to know which colors are used.
            
            # Efficient way:
            # Get the bitmask of neighbors in mask
            neigh_mask = adj[v] & mask
            # Iterate through neighbors
            # We can just check all u in 'colors' if they are neighbors?
            # 'colors' size is small (max ~100). 'order' size is ~200.
            # Checking all u in colors for each v is O(|colors|^2) worst case per v? No.
            
            # Better:
            for u, c in colors.items():
                if (adj[v] >> u) & 1:
                    used_colors.add(c)
            
            # Find smallest available color
            c = 0
            while c in used_colors:
                c += 1
            
            colors[v] = c
            if c + 1 > max_color:
                max_color = c + 1
                
        return max_color
    
    # Branch and bound
    def branch_and_bound(candidates_mask, current_size, current_set):
        nonlocal best_set, best_size
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        # Simple bound
        if current_size + bin(candidates_mask).count('1') <= best_size:
            return
        
        # Coloring bound
        cb = color_bound(candidates_mask)
        if current_size + cb <= best_size:
            return
        
        # Select vertex to branch on
        # Heuristic: Pick vertex with highest degree in candidates_mask
        # We can use the global 'order' to find the first valid vertex?
        # Or compute degree?
        # Computing degree is O(n). Finding max is O(n).
        # Let's pick the first vertex in 'order' that is in candidates_mask.
        # This is fast and often effective because high-degree vertices are earlier.
        best_v = -1
        for v in order:
            if (candidates_mask >> v) & 1:
                best_v = v
                break
        
        if best_v == -1:
            return
        
        # Branch 1: Include best_v
        new_mask_incl = candidates_mask & ~(1 << best_v) & ~adj[best_v]
        current_set.append(best_v)
        branch_and_bound(new_mask_incl, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude best_v
        new_mask_excl = candidates_mask & ~(1 << best_v)
        branch_and_bound(new_mask_excl, current_size, current_set)
    
    branch_and_bound(full_mask, 0, [])
    
    result = [self_valid[i] for i in best_set]
    
    if not result:
        for x in range(1, N + 1):
            if 2 * x not in squares:
                return [x]
        return [1]
        
    return result

# EVOLVE-BLOCK-END
