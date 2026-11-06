"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = None
_START = None

def _now():
    return time.perf_counter()

def _remaining():
    global _DEADLINE, _START
    if _DEADLINE is None:
        _DEADLINE = _START + 110.0  # safety margin before 120s limit
    return _DEADLINE - _now()

def construct():
    global _DEADLINE, _START
    _START = _now()
    _DEADLINE = _START + 110.0
    
    # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16)
    # doubled: two copies with colour 3 on all edges between them
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    # Base 32-vertex coloring
    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            base_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    # Build adjacency matrix for quick lookup: adj[i][j] = color
    def build_adj(n, colors):
        adj = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                adj[i][j] = colors[idx]
                adj[j][i] = colors[idx]
                idx += 1
        return adj
    
    adj = build_adj(32, base_colors)
    best_n = 32
    best_adj = [row[:] for row in adj]
    
    # Try to extend by adding vertices one by one
    def can_extend(n, adj, max_tries=500):
        """Try to add a new vertex n to the existing coloring."""
        if _remaining() < 0.5:
            return None
        
        # For a new vertex v, we need to assign colors to edges (v, i) for i in 0..n-1
        # Constraint: for any i, j < n, if color(v,i) == color(v,j) == color(i,j), then triangle
        
        # Precompute for each pair (i,j) which colors are forbidden for (v,i) and (v,j)
        # Actually, the constraint is: for each pair (i,j), if color(i,j) == c, 
        # then NOT (color(v,i) == c AND color(v,j) == c)
        
        # This is a constraint satisfaction problem. Use backtracking with randomization.
        
        # For efficiency, use a randomized local search approach:
        # 1. Initialize colors for (v, i) randomly
        # 2. Check for violations
        # 3. If violation found, fix by changing one of the edges
        
        for attempt in range(max_tries):
            if _remaining() < 0.5:
                break
            
            # Initialize with random colors
            new_edge_colors = [random.randint(0, 3) for _ in range(n)]
            
            # Local search: iteratively fix violations
            converged = True
            for iteration in range(100):
                converged = False
                # Check all pairs for violations
                for i in range(n):
                    for j in range(i + 1, n):
                        c_ij = adj[i][j]
                        if new_edge_colors[i] == c_ij and new_edge_colors[j] == c_ij:
                            # Violation! Fix by changing one of the two edges
                            # Choose randomly which to change
                            if random.random() < 0.5:
                                # Change new_edge_colors[i]
                                # It can't be c_ij if new_edge_colors[j] == c_ij
                                # Actually, it can be anything except creating new violations
                                # Simple: pick a random color that isn't c_ij
                                options = [c for c in range(4) if c != c_ij]
                                new_edge_colors[i] = random.choice(options)
                            else:
                                options = [c for c in range(4) if c != c_ij]
                                new_edge_colors[j] = random.choice(options)
                            converged = True
                            break
                    if converged:
                        break
                if not converged:
                    break
            
            if not converged:
                # No violations - valid extension!
                return new_edge_colors
        
        return None
    
    # Attempt to extend
    current_n = 32
    current_adj = best_adj
    
    while current_n < 62:
        if _remaining() < 1.0:
            break
        
        new_colors = can_extend(current_n, current_adj, max_tries=200)
        if new_colors is None:
            break
        
        # Extend the adjacency matrix
        new_n = current_n + 1
        new_adj = [[0] * new_n for _ in range(new_n)]
        for i in range(current_n):
            for j in range(current_n):
                new_adj[i][j] = current_adj[i][j]
            new_adj[i][current_n] = new_colors[i]
            new_adj[current_n][i] = new_colors[i]
        new_adj[current_n][current_n] = 0
        
        current_n = new_n
        current_adj = new_adj
        best_n = current_n
        best_adj = [row[:] for row in current_adj]
    
    # If we couldn't extend from 32, try a different strategy:
    # Start fresh with random initialization and local search to maximize n
    if best_n <= 32 and _remaining() > 5.0:
        # Try to build a larger coloring from scratch using local search
        def local_search_maximize(max_n=50):
            """Build a coloring via local search, maximizing n."""
            if _remaining() < 2.0:
                return None
            
            # Start with n=32 base, then try to add more
            # Or start with a random small coloring and grow it
            best_found_n = 32
            best_found_adj = None
            
            for start_n in [32, 24, 16]:
                if _remaining() < 2.0:
                    break
                
                if start_n == 32:
                    adj_cur = [row[:] for row in best_adj]
                elif start_n == 24:
                    # Extract first 24 from base
                    adj_cur = [[best_adj[i][j] for j in range(24)] for i in range(24)]
                else:
                    adj_cur = [[best_adj[i][j] for j in range(16)] for i in range(16)]
                
                cur_n = start_n
                
                for _ in range(50):
                    if _remaining() < 1.0:
                        break
                    if cur_n >= max_n:
                        break
                    
                    new_colors = can_extend(cur_n, adj_cur, max_tries=100)
                    if new_colors is None:
                        break
                    
                    new_n = cur_n + 1
                    new_adj = [[0] * new_n for _ in range(new_n)]
                    for i in range(cur_n):
                        for j in range(cur_n):
                            new_adj[i][j] = adj_cur[i][j]
                        new_adj[i][cur_n] = new_colors[i]
                        new_adj[cur_n][i] = new_colors[i]
                    new_adj[cur_n][cur_n] = 0
                    
                    cur_n = new_n
                    adj_cur = new_adj
                    if cur_n > best_found_n:
                        best_found_n = cur_n
                        best_found_adj = [row[:] for row in adj_cur]
            
            return best_found_n, best_found_adj
        
        result = local_search_maximize(50)
        if result is not None:
            n_found, adj_found = result
            if n_found > best_n:
                best_n = n_found
                best_adj = adj_found
    
    # Convert adjacency matrix to list of colors in lexicographic order
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            colors.append(best_adj[i][j])
    
    return {'n': best_n, 'colors': colors}
# EVOLVE-BLOCK-END
