import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s wall-clock limit
    
    # Start with the known 26-vertex construction from Lazebnik and Woldar
    # We'll try to find a valid 5-coloring of K_26 with no monochromatic C4
    
    n = 26
    num_colors = 5
    
    # Precompute edge indices for fast access
    # edge_idx[i][j] gives the index in the flat colors list for edge (i,j), i<j
    # We'll use a 2D array for internal work and convert at the end
    
    def build_adjacency(col_mat):
        adj = [[0]*n for _ in range(num_colors)]
        for i in range(n):
            for j in range(i+1, n):
                c = col_mat[i][j]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        return adj
    
    def count_c4s_fast(adj):
        total_penalty = 0
        for c in range(num_colors):
            for u in range(n):
                mask_u = adj[c][u]
                for v in range(u+1, n):
                    common = mask_u & adj[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        total_penalty += k * (k - 1) // 2
        return total_penalty
    
    def random_init():
        col = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                col[i][j] = random.randint(0, num_colors-1)
                col[j][i] = col[i][j]
        return col
    
    def delta_penalty(adj, i, j, old_c, new_c):
        """Compute change in C4 penalty when edge (i,j) changes from old_c to new_c."""
        delta = 0
        
        # In old_c: removing edge (i,j)
        # This affects pairs where i or j was part of a C4 with this edge
        # For each neighbor k of i in old_c (k != j):
        #   If i and k have >= 2 common neighbors in old_c, the count changes
        # Specifically, removing j from adj[old_c][i] affects:
        #   - Pairs (i, v) for v in adj[old_c][i]: common neighbors decrease by 1 if j was common
        #   - Pairs (j, v) for v in adj[old_c][j]: similar
        #   - The pair (i, j) itself: they were connected, now not
        
        # Let's think more carefully. A C4 is a set of 4 vertices {a,b,c,d} where
        # a-c, a-d, b-c, b-d are all in the same color class. Equivalently,
        # for pair (a,b), they have >= 2 common neighbors in that color.
        
        # When we remove edge (i,j) from color old_c:
        # 1. The pair (i,j) no longer exists in old_c, so we lose any C4s where (i,j) was a "diagonal"
        #    Actually, (i,j) being an edge means i and j are adjacent. For C4 counting via common neighbors,
        #    we count for each pair (u,v) the number of common neighbors. A C4 exists if any pair has >= 2 common neighbors.
        #    The penalty counts C(k,2) for each pair with k common neighbors.
        
        # Let me use a different approach: just recompute the penalty for affected pairs.
        # Affected pairs are those involving i or j.
        
        # This is getting complex. Let me just do a full recount for simplicity in the SA,
        # but optimize by only checking colors that changed.
        
        # Actually, for a 5-coloring of K_26, the total number of edges is 325.
        # A full recount is O(5 * 26 * 25 / 2) = O(1625) bit operations.
        # That's fast enough. Let me just do full recounts.
        
        return None  # Not used, we'll do full recounts
    
    def try_solve_sa():
        """Simulated annealing approach."""
        col = random_init()
        adj = build_adjacency(col)
        current_penalty = count_c4s_fast(adj)
        best_penalty = current_penalty
        best_col = [row[:] for row in col]
        
        T = 10.0
        cooling_rate = 0.9995
        
        no_improve_count = 0
        max_no_improve = 20000
        
        while time.time() < deadline:
            if current_penalty == 0:
                return col
            
            # Pick a random edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = col[i][j]
            new_c = random.randint(0, num_colors-1)
            if new_c == old_c:
                continue
            
            # Update colors
            col[i][j] = new_c
            col[j][i] = new_c
            
            # Update adjacency bitsets
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)
            
            # Recalculate penalty
            new_penalty = count_c4s_fast(adj)
            delta = new_penalty - current_penalty
            
            # Simulated annealing acceptance
            if delta <= 0:
                accept = True
            else:
                # Metropolis criterion
                r = random.random()
                if r < np.exp(-delta / T):
                    accept = True
                else:
                    accept = False
            
            if accept:
                current_penalty = new_penalty
                no_improve_count = 0
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                    best_col = [row[:] for row in col]
            else:
                # Revert
                col[i][j] = old_c
                col[j][i] = old_c
                adj[new_c][i] &= ~(1 << j)
                adj[new_c][j] &= ~(1 << i)
                adj[old_c][i] |= (1 << j)
                adj[old_c][j] |= (1 << i)
                
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            
            T *= cooling_rate
        
        return best_col
    
    def try_solve_hill_climb():
        """Steepest descent hill climbing."""
        col = random_init()
        adj = build_adjacency(col)
        current_penalty = count_c4s_fast(adj)
        best_penalty = current_penalty
        best_col = [row[:] for row in col]
        
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline:
            if current_penalty == 0:
                return col
            
            # Try all possible single-edge color changes, pick the best improvement
            best_delta = 0
            best_change = None
            
            for i in range(n):
                for j in range(i+1, n):
                    old_c = col[i][j]
                    for new_c in range(num_colors):
                        if new_c == old_c:
                            continue
                        
                        # Apply change
                        col[i][j] = new_c
                        col[j][i] = new_c
                        adj[old_c][i] &= ~(1 << j)
                        adj[old_c][j] &= ~(1 << i)
                        adj[new_c][i] |= (1 << j)
                        adj[new_c][j] |= (1 << i)
                        
                        new_penalty = count_c4s_fast(adj)
                        delta = new_penalty - current_penalty
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_change = (i, j, old_c, new_c)
                        
                        # Revert
                        col[i][j] = old_c
                        col[j][i] = old_c
                        adj[new_c][i] &= ~(1 << j)
                        adj[new_c][j] &= ~(1 << i)
                        adj[old_c][i] |= (1 << j)
                        adj[old_c][j] |= (1 << i)
            
            if best_change is not None:
                i, j, old_c, new_c = best_change
                col[i][j] = new_c
                col[j][i] = new_c
                adj[old_c][i] &= ~(1 << j)
                adj[old_c][j] &= ~(1 << i)
                adj[new_c][i] |= (1 << j)
                adj[new_c][j] |= (1 << i)
                current_penalty += best_delta
                no_improve_count = 0
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                    best_col = [row[:] for row in col]
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
        
        return best_col
    
    # Run multiple attempts with different strategies
    attempts = 0
    best_valid_col = None
    
    # First, try SA with multiple restarts
    while time.time() < deadline - 10:
        attempts += 1
        sol = try_solve_sa()
        if sol is not None:
            adj = build_adjacency(sol)
            if count_c4s_fast(adj) == 0:
                res_colors = []
                for i in range(n):
                    for j in range(i+1, n):
                        res_colors.append(sol[i][j])
                return {'n': n, 'colors': res_colors}
    
    # If SA didn't find it, try hill climbing
    while time.time() < deadline - 5:
        attempts += 1
        sol = try_solve_hill_climb()
        if sol is not None:
            adj = build_adjacency(sol)
            if count_c4s_fast(adj) == 0:
                res_colors = []
                for i in range(n):
                    for j in range(i+1, n):
                        res_colors.append(sol[i][j])
                return {'n': n, 'colors': res_colors}
    
    # Fallback to known valid smaller solution
    # Try n=22, 20, 18, 16, 14, 12, 11
    for n_try in [22, 20, 18, 16, 14, 12, 11]:
        if time.time() > deadline - 3:
            break
        
        n = n_try
        col = random_init()
        adj = build_adjacency(col)
        current_penalty = count_c4s_fast(adj)
        
        # Quick hill climb
        for _ in range(1000):
            if current_penalty == 0:
                break
            best_delta = 0
            best_change = None
            for i in range(n):
                for j in range(i+1, n):
                    old_c = col[i][j]
                    for new_c in range(num_colors):
                        if new_c == old_c:
                            continue
                        col[i][j] = new_c
                        col[j][i] = new_c
                        adj[old_c][i] &= ~(1 << j)
                        adj[old_c][j] &= ~(1 << i)
                        adj[new_c][i] |= (1 << j)
                        adj[new_c][j] |= (1 << i)
                        new_penalty = count_c4s_fast(adj)
                        delta = new_penalty - current_penalty
                        if delta < best_delta:
                            best_delta = delta
                            best_change = (i, j, old_c, new_c)
                        col[i][j] = old_c
                        col[j][i] = old_c
                        adj[new_c][i] &= ~(1 << j)
                        adj[new_c][j] &= ~(1 << i)
                        adj[old_c][i] |= (1 << j)
                        adj[old_c][j] |= (1 << i)
            
            if best_change is not None:
                i, j, old_c, new_c = best_change
                col[i][j] = new_c
                col[j][i] = new_c
                adj[old_c][i] &= ~(1 << j)
                adj[old_c][j] &= ~(1 << i)
                adj[new_c][i] |= (1 << j)
                adj[new_c][j] |= (1 << i)
                current_penalty += best_delta
            else:
                break
        
        if current_penalty == 0:
            res_colors = []
            for i in range(n):
                for j in range(i+1, n):
                    res_colors.append(col[i][j])
            return {'n': n, 'colors': res_colors}
    
    # Final fallback: seed solution n=11
    n_seed = 11
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    return {'n': n_seed, 'colors': colors_seed}
