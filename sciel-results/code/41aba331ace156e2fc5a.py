import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin

    def paley_colors(n):
        """Generate colors for Paley graph of order n (n must be prime 4k+1)."""
        if n % 4 != 1:
            return None
        squares = {x * x % n for x in range(1, n)}
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(0 if (j - i) % n in squares else 1)
        return colors

    def build_adj(n, colors):
        """Build adjacency bitmasks for both colors."""
        adj = [[0] * n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    def count_conflicts(n, colors):
        """
        Count the number of monochromatic K2,2,2.
        Returns 0 if valid.
        """
        adj = build_adj(n, colors)
        conflicts = 0
        for c in [0, 1]:
            # Iterate over all pairs (u, v) which will form one part of K2,2,2
            for u in range(n):
                for v in range(u + 1, n):
                    # Common neighbors of u and v in color c
                    mask_uv = adj[c][u] & adj[c][v]
                    # u and v are not in their own neighbor sets, but ensure they are excluded
                    mask_uv &= ~(1 << u)
                    mask_uv &= ~(1 << v)
                    
                    if mask_uv.bit_count() < 4:
                        # Need at least 4 other vertices to form the other two parts of size 2
                        # Actually, we need to pick x,y and w,z from mask_uv.
                        # So we need at least 4 vertices in mask_uv.
                        continue
                    
                    # Extract vertices in mask_uv
                    # Optimization: We don't need to extract all, we can iterate pairs
                    # But for n=30, extracting is fine.
                    verts = []
                    m = mask_uv
                    while m:
                        lsb = m & -m
                        i = lsb.bit_length() - 1
                        verts.append(i)
                        m ^= lsb
                    
                    # Check all pairs (x, y) in verts
                    # If x, y are the second part, we need w, z (third part)
                    # w, z must be in N(x) & N(y) & mask_uv
                    for i in range(len(verts)):
                        x = verts[i]
                        for j in range(i + 1, len(verts)):
                            y = verts[j]
                            common_xy = adj[c][x] & adj[c][y] & mask_uv
                            bc = common_xy.bit_count()
                            if bc >= 2:
                                # Number of ways to pick w, z from common_xy is C(bc, 2)
                                conflicts += bc * (bc - 1) // 2
        return conflicts

    def count_conflicts_delta(n, colors, edge_idx):
        """
        Calculate the change in conflicts if we flip edge_idx.
        This is complex to do incrementally. For n=30, full recount is ~0.1ms.
        So we just do full recount.
        """
        return count_conflicts(n, colors)

    # Step 1: Check Paley(29)
    colors_29 = paley_colors(29)
    best_result = None
    if colors_29 and count_conflicts(29, colors_29) == 0:
        best_result = {'n': 29, 'colors': colors_29}
        
    # Step 2: Try to find n=30
    # Strategy: Simulated Annealing on the extension of P(29)
    
    # Pre-compute index for edges involving vertex 29
    # Total edges in K_29 is 29*28/2 = 406.
    # Edges (i, 29) for i in 0..28 are at indices 406+i.
    
    target_n = 30
    base_colors = list(colors_29)
    num_new_edges = 29
    
    # We will try multiple restarts
    max_restarts = 500 # We will break early if time runs out
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize new vertex edges randomly
        colors = base_colors + [random.randint(0, 1) for _ in range(num_new_edges)]
        
        current_conflicts = count_conflicts(target_n, colors)
        
        if current_conflicts == 0:
            return {'n': target_n, 'colors': colors}
        
        # Simulated Annealing
        T = 2.0 # Initial temperature
        min_T = 0.01
        cooling_rate = 0.99
        max_steps = 2000 # Steps per restart
        
        best_conflicts = current_conflicts
        best_colors = list(colors)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Pick a random edge to flip
            # Bias towards edges involving the new vertex? 
            # Or random? Random is safer to explore the space.
            edge_idx = random.randint(0, len(colors) - 1)
            
            old_color = colors[edge_idx]
            new_color = 1 - old_color
            colors[edge_idx] = new_color
            
            new_conflicts = count_conflicts(target_n, colors)
            
            delta = new_conflicts - current_conflicts
            
            if delta < 0:
                # Accept
                current_conflicts = new_conflicts
                if current_conflicts == 0:
                    return {'n': target_n, 'colors': colors}
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_colors = list(colors)
            else:
                # Accept with probability exp(-delta / T)
                if random.random() < math.exp(-delta / T):
                    current_conflicts = new_conflicts
                else:
                    # Revert
                    colors[edge_idx] = old_color
            
            T *= cooling_rate
            if T < min_T:
                T = min_T
        
        # If we found a better non-zero solution, we might restart from it?
        # Or just keep the best overall.
        if best_conflicts < 10: # If we got close, maybe keep this state for next restart?
            # For simplicity, we just record the best found so far
            pass

    # Fallback to best result
    if not best_result:
        colors_17 = paley_colors(17)
        if colors_17 and count_conflicts(17, colors_17) == 0:
            best_result = {'n': 17, 'colors': colors_17}
        else:
            return {'n': 1, 'colors': []}

    return best_result
