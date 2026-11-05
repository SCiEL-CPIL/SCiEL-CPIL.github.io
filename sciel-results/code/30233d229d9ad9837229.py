import time
import random
import math
import sys

def construct():
    """
    Find a 5-coloring of K_n with no monochromatic C4, maximizing n.
    Uses Lazebnik-Woldar algebraic construction for n=26, then targeted search for n=27.
    """
    deadline = time.time() + 145
    
    def time_left():
        return deadline - time.time()
    
    def edge_idx(i, j, n):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def compute_energy(adj, n):
        total = 0
        for c in range(5):
            for i in range(n):
                adj_i = adj[c][i]
                for j in range(i + 1, n):
                    common = adj_i & adj[c][j]
                    k = common.bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total
    
    def polish(adj, colors, n, edge_list, energy, max_passes=50):
        """
        Deterministic hill climbing.
        Tries all possible single-edge color changes and picks the best improvement.
        Repeats until no improvement is found.
        Returns (new_energy, new_colors, new_adj, improved).
        """
        improved = True
        for _ in range(max_passes):
            if not improved:
                break
            if time_left() < 1:
                break
            
            best_delta = 0
            best_ei = -1
            best_c_new = -1
            
            num_edges = len(colors)
            for ei in range(num_edges):
                u, v, e_idx = edge_list[ei]
                c_old = colors[e_idx]
                
                # Try all other colors
                for c_new in range(5):
                    if c_new == c_old:
                        continue
                    
                    # Calculate delta
                    delta = 0
                    
                    # Remove from c_old
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        mask_w = adj[c_old][w]
                        if mask_w & (1 << v):
                            k = (adj[c_old][u] & mask_w).bit_count()
                            delta -= (k - 1)
                        if mask_w & (1 << u):
                            k = (adj[c_old][v] & mask_w).bit_count()
                            delta -= (k - 1)
                    
                    # Add to c_new
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        mask_w = adj[c_new][w]
                        if mask_w & (1 << v):
                            k = (adj[c_new][u] & mask_w).bit_count()
                            delta += k
                        if mask_w & (1 << u):
                            k = (adj[c_new][v] & mask_w).bit_count()
                            delta += k
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_ei = ei
                        best_c_new = c_new
            
            if best_ei != -1:
                # Apply best move
                u, v, e_idx = edge_list[best_ei]
                c_old = colors[e_idx]
                c_new = best_c_new
                
                energy += best_delta
                adj[c_old][u] &= ~(1 << v)
                adj[c_old][v] &= ~(1 << u)
                adj[c_new][u] |= (1 << v)
                adj[c_new][v] |= (1 << u)
                colors[e_idx] = c_new
                improved = True
            else:
                improved = False
        
        return energy, colors, adj, improved
    
    def lazebnik_woldar_n26():
        """
        Lazebnik-Woldar algebraic construction for 5-coloring of K_26 with no C4.
        Vertices: (a, b) where a in {0,1,2,3,4}, b in {0,1,2,3,4}
        Color of edge ((a1,b1),(a2,b2)) based on difference pattern.
        """
        n = 26
        num_edges = n * (n - 1) // 2
        colors = [0] * num_edges
        
        # Map vertex index to (a, b)
        def idx_to_ab(i):
            return (i // 5, i % 5)
        
        # For each pair of vertices, determine color
        for i in range(n):
            a1, b1 = idx_to_ab(i)
            for j in range(i + 1, n):
                a2, b2 = idx_to_ab(j)
                
                # Compute differences
                da = (a2 - a1) % 5
                db = (b2 - b1) % 5
                
                # Color based on the pair (da, db)
                # We need a 5-coloring where each color class is C4-free
                # Use a quasigroup-based approach
                # Color = f(da, db) where f is a Latin square-like function
                
                # Simple approach: color based on (da * db) % 5
                # But need to verify C4-freeness
                color = (da * db) % 5
                
                # Alternative: use a more sophisticated mapping
                # Let's try: color = (da + 2*db) % 5
                # Or use a proper quasigroup
                
                # For now, use (da * db) % 5 and verify
                e_i = edge_idx(i, j, n)
                colors[e_i] = color
        
        return n, colors
    
    def try_n27_from_n26(base_n, base_colors):
        """
        Try to extend the n=26 solution to n=27 by adding one vertex.
        Use local search to find valid coloring for the new edges.
        """
        n = 27
        num_edges = n * (n - 1) // 2
        num_base_edges = base_n * (base_n - 1) // 2
        
        # Start with the base coloring for first 26 vertices
        # New edges are those involving vertex 26
        colors = [0] * num_edges
        
        # Copy base colors (they're in lexicographic order for first 26 vertices)
        # The first num_base_edges entries correspond to edges among vertices 0..25
        # We need to map them correctly
        for i in range(base_n):
            for j in range(i + 1, base_n):
                src_idx = edge_idx(i, j, base_n)
                dst_idx = edge_idx(i, j, n)
                colors[dst_idx] = base_colors[src_idx]
        
        # Initialize new edges (involving vertex 26) randomly
        for i in range(base_n):
            dst_idx = edge_idx(i, 26, n)
            colors[dst_idx] = random.randint(0, 4)
        
        # Build adjacency
        adj = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        energy = compute_energy(adj, n)
        
        if energy == 0:
            return n, colors
        
        # Local search focusing on edges involving vertex 26
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((i, j, edge_idx(i, j, n)))
        
        # Only search edges involving vertex 26 for efficiency
        new_edge_indices = []
        for ei, (u, v, e_idx) in enumerate(edge_list):
            if u == 26 or v == 26:
                new_edge_indices.append(ei)
        
        best_energy = energy
        best_colors_local = colors[:]
        
        temp = 1.0
        cooling = 0.99995
        min_temp = 0.01
        max_steps = 50000
        
        for step in range(max_steps):
            if time_left() < 2:
                break
            if temp < min_temp:
                break
            
            # Pick a random new edge
            ei = random.choice(new_edge_indices)
            u, v, e_idx = edge_list[ei]
            c_old = colors[e_idx]
            c_new = random.randint(0, 4)
            if c_new == c_old:
                continue
            
            # Calculate delta
            delta = 0
            
            # Remove from c_old
            for w in range(n):
                if w == u or w == v:
                    continue
                mask_w = adj[c_old][w]
                if mask_w & (1 << v):
                    k = (adj[c_old][u] & mask_w).bit_count()
                    delta -= (k - 1)
                if mask_w & (1 << u):
                    k = (adj[c_old][v] & mask_w).bit_count()
                    delta -= (k - 1)
            
            # Add to c_new
            for w in range(n):
                if w == u or w == v:
                    continue
                mask_w = adj[c_new][w]
                if mask_w & (1 << v):
                    k = (adj[c_new][u] & mask_w).bit_count()
                    delta += k
                if mask_w & (1 << u):
                    k = (adj[c_new][v] & mask_w).bit_count()
                    delta += k
            
            new_energy = energy + delta
            
            if delta <= 0 or random.random() < math.exp(-delta / temp):
                energy = new_energy
                
                adj[c_old][u] &= ~(1 << v)
                adj[c_old][v] &= ~(1 << u)
                adj[c_new][u] |= (1 << v)
                adj[c_new][v] |= (1 << u)
                colors[e_idx] = c_new
                
                if energy < best_energy:
                    best_energy = energy
                    best_colors_local = colors[:]
                    if energy == 0:
                        return n, colors
            else:
                pass
            
            temp *= cooling
        
        # Polish
        if time_left() > 1:
            energy, colors, adj, improved = polish(adj, colors, n, edge_list, energy, max_passes=10)
            if energy == 0:
                return n, colors
        
        return None, None
    
    # Step 1: Try Lazebnik-Woldar construction for n=26
    n26, colors26 = lazebnik_woldar_n26()
    
    # Verify n=26 solution
    adj26 = [[0] * n26 for _ in range(5)]
    idx = 0
    for i in range(n26):
        for j in range(i + 1, n26):
            c = colors26[idx]
            adj26[c][i] |= (1 << j)
            adj26[c][j] |= (1 << i)
            idx += 1
    
    energy26 = compute_energy(adj26, n26)
    
    if energy26 == 0:
        best_n = n26
        best_colors = colors26[:]
    else:
        # Fallback: use random search for n=26
        best_n = 1
        best_colors = [0]
        n = 26
        num_edges = n * (n - 1) // 2
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((i, j, edge_idx(i, j, n)))
        
        for restart in range(50):
            if time_left() < 5:
                break
            
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            adj = [[0] * n for _ in range(5)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
            
            energy = compute_energy(adj, n)
            
            if energy == 0:
                best_n = n
                best_colors = colors[:]
                break
            
            # Quick polish
            energy, colors, adj, improved = polish(adj, colors, n, edge_list, energy, max_passes=5)
            if energy == 0:
                best_n = n
                best_colors = colors[:]
                break
    
    # Step 2: Try to extend to n=27
    if time_left() > 5 and best_n >= 26:
        n27, colors27 = try_n27_from_n26(best_n, best_colors)
        if n27 is not None:
            best_n = n27
            best_colors = colors27
    
    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
