import time
import random
import math

def _is_valid_coloring(n, colors):
    """
    Verify if the coloring is valid.
    colors is a list of length n*(n-1)/2.
    """
    if n < 2:
        return True
    # Build adjacency sets for each color
    color_adj = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            color_adj[c][i].add(j)
            color_adj[c][j].add(i)
            idx += 1
    # Check for C4: for each color, no two vertices should have 2+ common neighbors
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = color_adj[c][u] & color_adj[c][v]
                if len(common) >= 2:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 seconds margin
    
    # Precompute edge indices for n up to 29
    # edge_idx[n] = list of (i, j) for i < j
    edge_indices = {}
    for n in range(2, 30):
        edge_indices[n] = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def run_annealing(n, initial_colors=None, time_limit=30.0):
        """
        Run simulated annealing to find a C4-free 5-coloring of K_n.
        Returns (n, colors) if valid, else (0, []).
        """
        if n < 2:
            return n, []
        
        num_edges = n * (n - 1) // 2
        pairs = edge_indices[n]
        
        # Initialize colors
        if initial_colors is None:
            colors = [random.randint(0, 4) for _ in range(num_edges)]
        else:
            colors = initial_colors[:]
        
        # Data structures for incremental updates
        # M[c][i][j]: number of common neighbors of i and j in color c
        M = [[[0] * n for _ in range(n)] for _ in range(5)]
        # N[c][i]: set of neighbors of i in color c
        N = [[set() for _ in range(n)] for _ in range(5)]
        
        # Build initial state
        for idx, (i, j) in enumerate(pairs):
            c = colors[idx]
            N[c][i].add(j)
            N[c][j].add(i)
        
        # Calculate initial M
        for c in range(5):
            for i in range(n):
                for k in N[c][i]:
                    for j in N[c][k]:
                        if i != j:
                            M[c][i][j] += 1
        
        # Calculate initial Cost
        # Cost = sum_{c} sum_{i<j} binom(M[c][i][j], 2)
        cost = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    m = M[c][i][j]
                    if m >= 2:
                        cost += m * (m - 1) // 2
        
        if cost == 0:
            return n, colors
        
        # Simulated Annealing
        T = 1.0
        # Cooling rate: we want T to drop significantly.
        # Let's run for a fixed number of steps or time.
        end_time = time.time() + time_limit
        best_cost = cost
        best_colors = colors[:]
        
        while time.time() < end_time and cost > 0:
            # Select random edge
            idx = random.randint(0, num_edges - 1)
            i, j = pairs[idx]
            c_old = colors[idx]
            
            # Select new color
            c_new = random.randint(0, 4)
            if c_new == c_old:
                continue
            
            # Calculate Delta Cost
            delta_cost = 0
            
            # Remove edge (i, j) from color c_old
            # Update M and Cost for c_old
            # For each neighbor k of i in c_old:
            #   M[c_old][k][j] and M[c_old][j][k] decrease by 1
            # For each neighbor k of j in c_old:
            #   M[c_old][k][i] and M[c_old][i][k] decrease by 1
            
            # Note: If k is in N[c_old][i] AND N[c_old][j], we update M[c_old][k][i] and M[c_old][k][j]?
            # No, the logic is:
            # M[x][y] = # common neighbors.
            # Removing (i, j) from c_old means i and j are no longer adjacent in c_old.
            # This affects M[x][y] if x or y is i or j?
            # Actually, M[x][y] counts paths x-k-y.
            # If we remove edge (i, j), does it affect M[x][y]?
            # Only if the path used the edge (i, j)? No, M counts common neighbors, i.e., length 2 paths.
            # A length 2 path x-k-y uses edges (x, k) and (k, y).
            # Removing (i, j) affects M[x][y] only if one of the edges in the path was (i, j).
            # So, if x=i and k=j, then edge (i, j) is used. Then M[i][y] is affected.
            # Specifically, M[i][y] decreases by 1 if (i, j) and (j, y) are in c_old.
            # So, for all y in N[c_old][j], M[i][y] decreases by 1.
            # And for all x in N[c_old][i], M[x][j] decreases by 1.
            
            # Let's implement this carefully.
            
            # Helper to update cost when M[c][a][b] changes by delta (delta is +/- 1)
            # old_val = M[c][a][b]
            # new_val = old_val + delta
            # cost_change = binom(new_val, 2) - binom(old_val, 2)
            
            # We need to store the changes to revert if rejected.
            changes = [] # List of (c, a, b, old_val)
            
            # 1. Remove (i, j) from c_old
            # For each k in N[c_old][j]:
            #   M[c_old][i][k] decreases by 1
            for k in N[c_old][j]:
                old_val = M[c_old][i][k]
                new_val = old_val - 1
                M[c_old][i][k] = new_val
                M[c_old][k][i] = new_val # Symmetric
                changes.append((c_old, i, k, old_val))
                # Update cost
                if old_val >= 2:
                    delta_cost -= (old_val - 1) # binom(k,2) - binom(k-1,2) = k-1
                # if old_val == 1, binom(1,2)=0, binom(0,2)=0, change 0.
                # if old_val == 0, shouldn't happen if k is neighbor.
            
            # For each k in N[c_old][i]:
            #   M[c_old][j][k] decreases by 1
            for k in N[c_old][i]:
                old_val = M[c_old][j][k]
                new_val = old_val - 1
                M[c_old][j][k] = new_val
                M[c_old][k][j] = new_val
                changes.append((c_old, j, k, old_val))
                if old_val >= 2:
                    delta_cost -= (old_val - 1)
            
            # Remove from N sets
            N[c_old][i].remove(j)
            N[c_old][j].remove(i)
            
            # 2. Add (i, j) to c_new
            # For each k in N[c_new][j]:
            #   M[c_new][i][k] increases by 1
            for k in N[c_new][j]:
                old_val = M[c_new][i][k]
                new_val = old_val + 1
                M[c_new][i][k] = new_val
                M[c_new][k][i] = new_val
                changes.append((c_new, i, k, old_val))
                # binom(k+1, 2) - binom(k, 2) = k
                delta_cost += old_val
            
            # For each k in N[c_new][i]:
            #   M[c_new][j][k] increases by 1
            for k in N[c_new][i]:
                old_val = M[c_new][j][k]
                new_val = old_val + 1
                M[c_new][j][k] = new_val
                M[c_new][k][j] = new_val
                changes.append((c_new, j, k, old_val))
                delta_cost += old_val
            
            # Add to N sets
            N[c_new][i].add(j)
            N[c_new][j].add(i)
            
            # Accept or Reject
            if delta_cost <= 0:
                # Accept
                cost += delta_cost
                colors[idx] = c_new
                if cost < best_cost:
                    best_cost = cost
                    best_colors = colors[:]
            else:
                # Accept with probability exp(-delta_cost / T)
                if random.random() < math.exp(-delta_cost / T):
                    cost += delta_cost
                    colors[idx] = c_new
                    if cost < best_cost:
                        best_cost = cost
                        best_colors = colors[:]
                else:
                    # Revert
                    for c, a, b, old_val in changes:
                        M[c][a][b] = old_val
                        M[c][b][a] = old_val
                    # Revert N sets
                    N[c_new][i].remove(j)
                    N[c_new][j].remove(i)
                    N[c_old][i].add(j)
                    N[c_old][j].add(i)
                    # colors[idx] remains c_old
            
            # Cooling
            T *= 0.9995
            if T < 0.01:
                T = 0.01
        
        if best_cost == 0:
            return n, best_colors
        else:
            return 0, []

    # Strategy:
    # 1. Try to find valid n=26
    # 2. If found, try to extend to 27
    # 3. If found, try to extend to 28
    
    current_n = 0
    current_colors = []
    
    # Phase 1: n=26
    # Use multiple restarts if time permits
    time_left = deadline - time.time()
    attempts = 0
    max_attempts = 5
    while current_n < 26 and time_left > 10 and attempts < max_attempts:
        t_limit = max(5.0, (deadline - time.time()) / (max_attempts - attempts))
        n_res, colors_res = run_annealing(26, None, t_limit)
        if n_res == 26:
            current_n = 26
            current_colors = colors_res
            break
        attempts += 1
    
    if current_n < 26:
        # Fallback: try smaller n if 26 fails
        for n_try in [25, 24, 23]:
            if time.time() > deadline - 5:
                break
            n_res, colors_res = run_annealing(n_try, None, 10.0)
            if n_res == n_try:
                current_n = n_try
                current_colors = colors_res
                break
    
    if current_n == 0:
        return {'n': 0, 'colors': []}
    
    best_n = current_n
    best_colors = current_colors[:]
    
    # Phase 2: Try to extend to n+1
    # We have a valid coloring for current_n.
    # Try to extend to current_n + 1.
    # We can do this by taking the best_n coloring, adding a vertex, and re-optimizing.
    
    target_n = best_n + 1
    if target_n <= 29 and time.time() < deadline - 10:
        # Create initial state for target_n
        # Copy best_colors to a new list of size target_n
        num_edges_old = best_n * (best_n - 1) // 2
        num_edges_new = target_n * (target_n - 1) // 2
        init_colors = best_colors[:]
        # Add random colors for the new edges (new vertex is index best_n)
        for _ in range(num_edges_new - num_edges_old):
            init_colors.append(random.randint(0, 4))
        
        # Run annealing on target_n with this initial state
        # Allocate remaining time
        t_limit = deadline - time.time() - 5
        if t_limit > 10:
            n_res, colors_res = run_annealing(target_n, init_colors, t_limit)
            if n_res == target_n:
                best_n = target_n
                best_colors = colors_res
        
        # If successful, try to extend again?
        # Unlikely to jump 2 steps, but let's check time.
        target_n = best_n + 1
        if target_n <= 29 and time.time() < deadline - 5:
            num_edges_old = best_n * (best_n - 1) // 2
            num_edges_new = target_n * (target_n - 1) // 2
            init_colors = best_colors[:]
            for _ in range(num_edges_new - num_edges_old):
                init_colors.append(random.randint(0, 4))
            
            t_limit = deadline - time.time() - 2
            if t_limit > 5:
                n_res, colors_res = run_annealing(target_n, init_colors, t_limit)
                if n_res == target_n:
                    best_n = target_n
                    best_colors = colors_res

    return {'n': best_n, 'colors': best_colors}
