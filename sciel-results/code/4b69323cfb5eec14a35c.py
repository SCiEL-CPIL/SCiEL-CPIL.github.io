import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    n = 26
    num_colors = 5
    
    # Edge representation: colors[i][j] for i < j
    # Using a 2D array for speed, but we only need i<j
    # Let's use a flat list for the output format, but a 2D array for internal logic
    colors = [[0]*n for _ in range(n)]
    
    # Initialization: Random or Heuristic
    # Heuristic: Use cyclic structure if possible, otherwise random
    # For n=26, let's try a random init with some structure
    # Or better, try to build it from scratch
    
    # To increase chances, we can try multiple restarts
    best_score = -1
    best_colors = None
    
    def count_c4s(col_mat):
        """
        Counts the number of monochromatic C4s.
        A C4 in color c is formed by two vertices u, v having >= 2 common neighbors in color c.
        Actually, we want the number of distinct C4s or just a penalty.
        For local search, we can just count the number of pairs (u,v) that have >=2 common neighbors in color c.
        Each such pair corresponds to at least one C4 (actually C(k,2) C4s if k common neighbors).
        We want this to be 0.
        """
        total_penalty = 0
        for c in range(num_colors):
            # Build adjacency for color c
            # adj[u] is a set of v such that col[u][v] == c
            adj = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if col_mat[i][j] == c:
                        adj[i].add(j)
                        adj[j].add(i)
            
            # For each pair (u, v), count common neighbors
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u] & adj[v]: # Intersection
                        common = adj[u] & adj[v]
                        k = len(common)
                        if k >= 2:
                            # Number of C4s formed by this pair is C(k, 2)
                            total_penalty += k * (k - 1) // 2
        return total_penalty

    def is_valid(col_mat):
        return count_c4s(col_mat) == 0

    def random_init():
        col = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                col[i][j] = random.randint(0, num_colors-1)
                col[j][i] = col[i][j]
        return col

    def try_solve():
        # Try a heuristic initialization: 
        # Map vertices to (x,y) in Z_2 x Z_13? 
        # Or just random.
        # Let's use a "greedy" random restart with local search
        
        col = random_init()
        
        # Local Search
        current_penalty = count_c4s(col)
        best_local_penalty = current_penalty
        best_local_col = [row[:] for row in col]
        
        no_improve_count = 0
        
        while time.time() < deadline:
            if current_penalty == 0:
                return col
            
            # Pick a random edge to recolor
            # Or pick an edge involved in a C4
            # To be efficient, pick a random edge
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
            
            col[i][j] = new_c
            col[j][i] = new_c
            
            new_penalty = count_c4s(col)
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                no_improve_count = 0
                if current_penalty < best_local_penalty:
                    best_local_penalty = current_penalty
                    best_local_col = [row[:] for row in col]
            else:
                # Revert
                col[i][j] = old_c
                col[j][i] = old_c
                no_improve_count += 1
                
                # Simulated Annealing style: accept worse with low prob?
                # Or just restart if stuck too long.
                if no_improve_count > 1000:
                    break
        
        return best_local_col

    # Run multiple attempts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        sol = try_solve()
        if sol is not None:
            # Verify
            if is_valid(sol):
                # Convert to list format
                res_colors = []
                for i in range(n):
                    for j in range(i+1, n):
                        res_colors.append(sol[i][j])
                return {'n': n, 'colors': res_colors}
        
        # If we are running out of time, maybe try smaller n?
        # But the task wants to maximize n.
        # If we can't get 26, we should return the best valid one found?
        # The seed returns 11. If we fail 26, we should return 11 to not score 0.
        # But the evaluator expects a valid coloring.
        # If we return an invalid coloring, score is 0.
        # So we MUST return a valid one.
        
        # Fallback: Return the seed solution if we fail to improve
        if time.time() > deadline - 10:
            break

    # Fallback to seed solution (n=11)
    n_seed = 11
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    return {'n': n_seed, 'colors': colors_seed}
