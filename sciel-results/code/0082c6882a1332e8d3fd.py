import time
import random

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

    def is_valid(n, colors):
        """
        Check if coloring has no monochromatic K2,2,2.
        Uses bitmask adjacency for speed.
        """
        adj = [[0] * n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in [0, 1]:
            # Check for K2,2,2 in color c
            # Iterate over all pairs (u, v)
            for u in range(n):
                for v in range(u + 1, n):
                    # Common neighbors of u and v in color c
                    mask = adj[c][u] & adj[c][v]
                    # u and v are not in their own neighbor sets, but ensure they are excluded
                    # just in case of logic errors, though bit i is not set in adj[i]
                    mask &= ~(1 << u)
                    mask &= ~(1 << v)
                    
                    # We need to find a K2,2 in the subgraph induced by 'mask'
                    # K2,2 exists if there are two disjoint pairs (x,y) and (w,z) in mask
                    # such that all edges between them exist in color c.
                    # This is equivalent to: any pair (x,y) in mask has at least 2 common neighbors in mask.
                    
                    # Extract vertices in mask
                    verts = []
                    m = mask
                    while m:
                        lsb = m & -m
                        i = lsb.bit_length() - 1
                        verts.append(i)
                        m ^= lsb
                    
                    # Check all pairs (x, y) in verts
                    for i in range(len(verts)):
                        x = verts[i]
                        for j in range(i + 1, len(verts)):
                            y = verts[j]
                            # Common neighbors of x and y in color c, restricted to mask
                            common = adj[c][x] & adj[c][y] & mask
                            if common.bit_count() >= 2:
                                return False
        return True

    # Helper to count violations (number of mono K2,2,2) for local search
    # This is expensive, so we might just use is_valid and random flips
    # Or a lighter "conflict count"
    
    def count_conflicts(n, colors):
        """
        Count the number of monochromatic K2,2,2.
        Used for local search.
        """
        adj = [[0] * n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        conflicts = 0
        for c in [0, 1]:
            for u in range(n):
                for v in range(u + 1, n):
                    mask = adj[c][u] & adj[c][v]
                    mask &= ~(1 << u)
                    mask &= ~(1 << v)
                    
                    verts = []
                    m = mask
                    while m:
                        lsb = m & -m
                        i = lsb.bit_length() - 1
                        verts.append(i)
                        m ^= lsb
                    
                    for i in range(len(verts)):
                        x = verts[i]
                        for j in range(i + 1, len(verts)):
                            y = verts[j]
                            common = adj[c][x] & adj[c][y] & mask
                            bc = common.bit_count()
                            if bc >= 2:
                                # Number of K2,2,2 formed by (u,v) and this (x,y) pair
                                # is C(bc, 2)
                                conflicts += bc * (bc - 1) // 2
        return conflicts

    # Step 1: Check Paley(29)
    colors_29 = paley_colors(29)
    best_result = None
    if colors_29 and is_valid(29, colors_29):
        best_result = {'n': 29, 'colors': colors_29}
        
    # Step 2: Try to find n=30
    # Strategy: Start with P(29) + random 30th vertex, then local search
    if time.time() < deadline - 10:
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Initialize with P(29)
            colors = list(colors_29)
            # Add vertex 29
            # Edges (29, i) for i in 0..28
            # Index for (i, 29) in flat array:
            # Total pairs before row 29: sum_{k=0}^{28} (29-k) = 29*28/2 = 406
            # Pairs (i, 29) are the last 29 elements
            # idx for (i, 29) is 406 + i
            
            # Randomly assign colors for the new vertex
            for i in range(29):
                colors.append(random.randint(0, 1))
            
            # Local search
            current_conflicts = count_conflicts(30, colors)
            
            improved = True
            steps = 0
            while improved and steps < 1000 and time.time() < deadline:
                improved = False
                steps += 1
                # Randomly pick an edge to flip
                # Focus on edges involving the new vertex or random edges
                # To be effective, we should pick edges that are part of conflicts
                # But finding them is hard. Random is okay if we do many iterations.
                # Better: Pick a random edge, flip it, if conflicts decrease, keep.
                
                edge_idx = random.randint(0, len(colors) - 1)
                old_color = colors[edge_idx]
                new_color = 1 - old_color
                colors[edge_idx] = new_color
                
                new_conflicts = count_conflicts(30, colors)
                
                if new_conflicts < current_conflicts:
                    current_conflicts = new_conflicts
                    improved = True
                    if current_conflicts == 0:
                        # Valid!
                        return {'n': 30, 'colors': colors}
                else:
                    # Revert
                    colors[edge_idx] = old_color
    
    # Step 3: Check Paley(17) as fallback
    if not best_result:
        colors_17 = paley_colors(17)
        if colors_17 and is_valid(17, colors_17):
            best_result = {'n': 17, 'colors': colors_17}
        else:
            # Should not happen
            return {'n': 1, 'colors': []}

    return best_result
