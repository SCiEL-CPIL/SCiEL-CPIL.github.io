"""Construct 5-colourings of K_n without monochromatic C4, maximising n."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 115.0  # Safety margin before 120s kill

    def full_check(n, colors):
        """Verify no colour class has a C4."""
        adj = [[set() for _ in range(5)] for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][c].add(j)
                adj[j][c].add(i)
                idx += 1
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    if len(adj[i][c] & adj[j][c]) >= 2:
                        return False
        return True

    def count_violations(n, colors):
        """Count number of C4 violations (pairs with >=2 common neighbours in same colour)."""
        adj = [[set() for _ in range(5)] for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][c].add(j)
                adj[j][c].add(i)
                idx += 1
        count = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = len(adj[i][c] & adj[j][c])
                    if common >= 2:
                        count += common - 1
        return count

    def local_search(n, colors, max_iters=500000):
        """Try to fix violations via local search."""
        colors = list(colors)
        total_edges = n * (n - 1) // 2
        
        def edge_idx(i, j):
            if i > j:
                i, j = j, i
            return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
        # Precompute adjacency for fast updates
        adj = [[set() for _ in range(5)] for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][c].add(j)
                adj[j][c].add(i)
                idx += 1
        
        def get_violations():
            """Return list of (i, j, c) where adj[i][c] & adj[j][c] has size >= 2."""
            viols = []
            for c in range(5):
                for i in range(n):
                    for j in range(i + 1, n):
                        common = adj[i][c] & adj[j][c]
                        if len(common) >= 2:
                            viols.append((i, j, c))
            return viols
        
        best_viol = count_violations(n, colors)
        if best_viol == 0:
            return colors
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            viols = get_violations()
            if not viols:
                return colors
            
            # Pick a random violation and try to fix it
            i, j, c = random.choice(viols)
            common = adj[i][c] & adj[j][c]
            # Try re-coloring edge (i,j) to a different color
            old_c = colors[edge_idx(i, j)]
            for new_c in range(5):
                if new_c == old_c:
                    continue
                # Check if re-coloring would reduce violations
                # Remove (i,j) from color old_c
                adj[i][old_c].discard(j)
                adj[j][old_c].discard(i)
                # Add (i,j) to color new_c
                adj[i][new_c].add(j)
                adj[j][new_c].add(i)
                colors[edge_idx(i, j)] = new_c
                
                new_viol = count_violations(n, colors)
                if new_viol < best_viol:
                    best_viol = new_viol
                    if best_viol == 0:
                        return colors
                else:
                    # Undo
                    adj[i][new_c].discard(j)
                    adj[j][new_c].discard(i)
                    adj[i][old_c].add(j)
                    adj[j][old_c].add(i)
                    colors[edge_idx(i, j)] = old_c
        
        return colors

    # Seed: n=11
    n_seed = 11
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best_n = n_seed
    best_colors = colors_seed

    # Try to build up from n=11 to larger n using local search
    # Start with n=11, then try to add vertices one at a time
    
    for target_n in range(12, 29):
        if time.time() > deadline - 10:
            break
        
        # Start from the current best coloring and add a new vertex
        n_curr = best_n
        colors_curr = best_colors
        
        # Create a new coloring for n_curr + 1 by adding a new vertex
        # The new vertex connects to all existing vertices
        # We need to assign colors to edges (new_vertex, k) for k in range(n_curr)
        
        new_n = n_curr + 1
        new_colors = list(colors_curr)
        # Add edges from new vertex (index n_curr) to all previous vertices
        # In lexicographic order, edges (i, n_curr) for i in 0..n_curr-1 come at the end
        # Actually, let's rebuild the color list properly
        
        # Rebuild: for new_n vertices, edges in lexicographic order
        # First all edges among 0..n_curr-1 (which is colors_curr), then edges (i, n_curr) for i=0..n_curr-1
        # Wait, that's not right. Lexicographic order: (0,1),(0,2),...,(0,n-1),(1,2),...
        # So for new_n = n_curr+1, the edges are:
        # (0,1),(0,2),...,(0,n_curr), (1,2),(1,3),...,(1,n_curr), ...
        # The edges involving n_curr are: (0,n_curr),(1,n_curr),...,(n_curr-1,n_curr)
        # These are NOT at the end. They are interleaved.
        
        # This is getting complicated. Let me use a different approach.
        # Instead of growing, let me try random initialization + local search for each n.
        
        # Try random initialization for target_n
        found = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Random coloring
            total_edges = target_n * (target_n - 1) // 2
            rand_colors = [random.randint(0, 4) for _ in range(total_edges)]
            
            # Local search
            improved = local_search(target_n, rand_colors, max_iters=100000)
            if count_violations(target_n, improved) == 0:
                if full_check(target_n, improved):
                    best_n = target_n
                    best_colors = improved
                    found = True
                    break
        
        if found:
            print(f"Found valid construction for n={target_n}")
    
    # Final verification
    if best_n > 11:
        if not full_check(best_n, best_colors):
            # Fall back to seed
            best_n = n_seed
            best_colors = colors_seed
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
