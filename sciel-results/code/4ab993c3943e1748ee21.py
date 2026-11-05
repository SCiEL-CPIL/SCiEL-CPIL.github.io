import time
import sys

def _creates_c6(adj, c, u, v):
    """
    Check if adding edge (u, v) with color c creates a monochromatic C6.
    adj: list of 4 lists of bitmasks. adj[c][i] is bitmask of neighbors of i in color c.
    """
    # We are adding (u, v).
    # Check for path v-x-y-z-u
    # x in N(v) \ {u, v}
    # y in N(x) \ {u, v, x}
    # z in N(y) \ {u, v, x, y} AND z in N(u)
    
    mask_u = 1 << u
    mask_v = 1 << v
    
    # Neighbors of v in color c
    # Exclude u and v
    cand_x = adj[c][v] & ~mask_u & ~mask_v
    
    while cand_x:
        lsb = cand_x & -cand_x
        x = lsb.bit_length() - 1
        cand_x ^= lsb
        mask_x = 1 << x
        
        # Neighbors of x in color c
        # Exclude u, v, x
        cand_y = adj[c][x] & ~mask_u & ~mask_v & ~mask_x
        
        while cand_y:
            lsb_y = cand_y & -cand_y
            y = lsb_y.bit_length() - 1
            cand_y ^= lsb_y
            mask_y = 1 << y
            
            # Neighbors of y in color c AND neighbors of u in color c
            # Exclude u, v, x, y
            # If there is any such z, we have a C6
            cand_z = adj[c][y] & adj[c][u] & ~mask_u & ~mask_v & ~mask_x & ~mask_y
            
            if cand_z:
                return True
                
    return False

def _solve_n(n, time_limit=10.0):
    """
    Try to find a valid 4-coloring for K_n using backtracking.
    Returns list of colors or None.
    """
    start_time = time.time()
    
    # adj[c][i] : bitmask of neighbors of i in color c
    adj = [[0] * n for _ in range(4)]
    colors = [] # Store colors in lexicographic order for output
    edge_index = 0
    total_edges = n * (n - 1) // 2
    
    # Pre-compute edge list in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Symmetry breaking: Fix color of (0,1) to 0
    # We will handle this by skipping other colors for the first edge
    
    def backtrack(idx):
        nonlocal adj, colors, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return False
        
        if idx == total_edges:
            return True
            
        u, v = edges[idx]
        
        # Heuristic: Try colors with fewer current edges first
        # Count edges per color
        counts = [0, 0, 0, 0]
        # This is O(n^2) per step, which is slow. 
        # Better to maintain counts incrementally.
        # But for simplicity in this draft, let's just iterate 0..3 
        # and sort by count if we want. 
        # For now, just try 0,1,2,3.
        
        # Symmetry breaking for first edge
        if idx == 0:
            possible_colors = [0]
        else:
            possible_colors = [0, 1, 2, 3]
            
        # Optional: Sort colors by current degree (fewer first)
        # We need to track degrees.
        # Let's add a degree tracker.
        
        for c in possible_colors:
            # Check if adding (u, v) with color c creates a C6
            if not _creates_c6(adj, c, u, v):
                # Tentatively add
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                colors.append(c)
                
                if backtrack(idx + 1):
                    return True
                
                # Backtrack
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                colors.pop()
                
        return False

    # We need to maintain counts to sort colors.
    # Let's modify the state to include counts.
    # Actually, _creates_c6 is fast. The bottleneck is the search tree.
    # Let's add a simple heuristic: prefer colors with lower degree.
    
    # Re-implementing with degree tracking
    adj = [[0] * n for _ in range(4)]
    counts = [0, 0, 0, 0]
    colors = []
    
    def backtrack(idx):
        nonlocal adj, counts, colors, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return False
        
        if idx == total_edges:
            return True
            
        u, v = edges[idx]
        
        if idx == 0:
            possible_colors = [0]
        else:
            # Sort colors by count
            possible_colors = sorted([0, 1, 2, 3], key=lambda c: counts[c])
            
        for c in possible_colors:
            if not _creates_c6(adj, c, u, v):
                # Add
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                counts[c] += 1
                colors.append(c)
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                counts[c] -= 1
                colors.pop()
                
        return False

    if backtrack(0):
        return colors
    return None

def construct():
    # Try n=19, 18, 17
    for n in [19, 18, 17]:
        result = _solve_n(n, time_limit=30.0)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback if all fail (should not happen for 17)
    # Return a small valid coloring? 
    # The task requires valid coloring.
    # If n=17 fails, we have a problem. 
    # But n=17 is known feasible.
    # If the solver is too slow, we might return invalid or None.
    # We must ensure validity.
    
    # To be safe, if 17 fails, we can try a very short local search for 17?
    # Or just return n=1? No, maximize n.
    # I will trust the solver for 17.
    
    # If all fail, return a dummy valid small n?
    # The contract says 1 <= n <= 20.
    # If we can't find 17, we return what we have?
    # _solve_n returns None if timeout.
    
    # Let's add a fallback: if 17 fails, try 16?
    # The loop handles this if we extend the list.
    # But 19, 18, 17 are the targets.
    
    # If 17 fails due to time, we might need to return a valid smaller n.
    # Let's try 16, 15...
    for n in [16, 15, 14]:
        result = _solve_n(n, time_limit=5.0)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Absolute fallback
    n = 1
    return {'n': n, 'colors': []}

