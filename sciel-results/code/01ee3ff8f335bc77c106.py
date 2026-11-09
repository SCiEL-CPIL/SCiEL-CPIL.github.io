import numpy as np
import itertools
import time

def construct():
    n = 16
    deadline = time.time() + 140.0  # Leave 10s margin
    
    # Precompute all K4s
    k4s = list(itertools.combinations(range(n), 4))
    num_k4s = len(k4s)
    
    # Map each K4 to its index
    # For each edge (u, v), find which K4s contain it
    # edge_to_k4s[u][v] = list of k4 indices
    edge_to_k4s = [[[] for _ in range(n)] for _ in range(n)]
    for idx, quad in enumerate(k4s):
        # quad is (a, b, c, d)
        # Edges are (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
        edges = list(itertools.combinations(quad, 2))
        for u, v in edges:
            if u < v:
                edge_to_k4s[u][v].append(idx)
            else:
                edge_to_k4s[v][u].append(idx)

    def evaluate_coloring(colors, k):
        """
        colors: 16x16 matrix
        k: number of colors (0 to k-1)
        Returns total violations
        """
        # Initialize histograms for each K4
        # hist[k4_idx][color] = count
        # distinct[k4_idx] = number of distinct colors
        # We only care if distinct < 5
        
        # To save memory/time, we can just compute distinct counts directly?
        # No, we need to update efficiently.
        
        # Let's just do a full check for the initial state, 
        # then use incremental updates for the search.
        
        violations = 0
        distinct_counts = np.zeros(num_k4s, dtype=np.int8)
        # We don't need full histograms for the full check, just distinct count
        for idx, quad in enumerate(k4s):
            a, b, c, d = quad
            c1 = colors[a][b]
            c2 = colors[a][c]
            c3 = colors[a][d]
            c4 = colors[b][c]
            c5 = colors[b][d]
            c6 = colors[c][d]
            d = len({c1, c2, c3, c4, c5, c6})
            distinct_counts[idx] = d
            if d < 5:
                violations += 1
        return violations, distinct_counts

    def search(k, time_limit):
        """
        Search for a valid coloring using k colors.
        Returns (valid, colors_matrix)
        """
        start_time = time.time()
        
        # Initialize random coloring
        colors = np.random.randint(0, k, size=(n, n))
        # Symmetrize
        colors = (colors + colors.T) // 2 # This is wrong for random ints.
        # Better:
        colors = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i+1, n):
                c = np.random.randint(0, k)
                colors[i][j] = c
                colors[j][i] = c
        
        violations, distinct_counts = evaluate_coloring(colors, k)
        
        if violations == 0:
            return True, colors

        # For incremental updates, we need histograms per K4
        # hist[k4_idx] is a dict or array of size k
        # Using arrays of size k for each k4 might be heavy if k is large, but k<=15, num_k4s=1820.
        # 1820 * 15 * 1 byte = ~27KB. Very small.
        
        # Initialize histograms
        # hist[idx][c] = count of color c in K4 idx
        hist = np.zeros((num_k4s, k), dtype=np.int8)
        
        # Fill histograms
        for idx, quad in enumerate(k4s):
            a, b, c, d = quad
            edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            for u, v in edges:
                col = colors[u][v]
                hist[idx][col] += 1
            
        # Verify distinct counts from hist
        # distinct = sum(hist[idx] > 0)
        # We already have distinct_counts from evaluate_coloring, but let's recompute to be sure
        # Actually, evaluate_coloring did it.
        
        best_violations = violations
        best_colors = colors.copy()
        
        # Simulated Annealing
        # T = 1.0, decay
        T = 1.0
        decay = 0.999
        
        while time.time() - start_time < time_limit:
            # Pick random edge
            u = np.random.randint(0, n)
            v = np.random.randint(0, n)
            if u == v:
                continue
            
            old_col = colors[u][v]
            new_col = np.random.randint(0, k)
            if old_col == new_col:
                continue
            
            # Calculate change in violations
            # Affected K4s are those in edge_to_k4s[min(u,v)][max(u,v)]
            affected_k4s = edge_to_k4s[min(u,v)][max(u,v)]
            
            delta_violations = 0
            # We need to update hist and distinct_counts for these K4s
            # We store the changes to revert if rejected
            
            changes = [] # (k4_idx, old_distinct, new_distinct)
            
            for idx in affected_k4s:
                # Update hist
                old_count = hist[idx][old_col]
                new_count = hist[idx][new_col]
                
                # Determine if distinct count changes
                old_distinct = distinct_counts[idx]
                
                # Decrement old_col
                hist[idx][old_col] -= 1
                if hist[idx][old_col] == 0:
                    old_distinct -= 1 # This is the distinct count AFTER removing old_col's last instance
                    # Wait, distinct_counts[idx] is the current distinct count.
                    # If old_count was 1, then old_col was present. Removing it reduces distinct by 1.
                    # If old_count > 1, old_col is still present.
                
                # Increment new_col
                hist[idx][new_col] += 1
                if new_count == 0: # It was 0 before increment
                    old_distinct += 1 # Adding a new color increases distinct by 1
                
                new_distinct = old_distinct
                
                # Check violation status
                old_is_violation = (distinct_counts[idx] < 5)
                new_is_violation = (new_distinct < 5)
                
                if old_is_violation and not new_is_violation:
                    delta_violations -= 1
                elif not old_is_violation and new_is_violation:
                    delta_violations += 1
                
                changes.append((idx, distinct_counts[idx], new_distinct))
                distinct_counts[idx] = new_distinct
            
            # Accept/Reject
            accept = False
            if delta_violations < 0:
                accept = True
            elif delta_violations == 0:
                accept = True
            else:
                # Boltzmann probability
                # exp(-delta / T)
                if T > 0.001:
                    prob = np.exp(-delta_violations / T)
                    if np.random.random() < prob:
                        accept = True
            
            if accept:
                colors[u][v] = new_col
                colors[v][u] = new_col
                violations += delta_violations
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors.copy()
                    if violations == 0:
                        return True, colors
            else:
                # Revert changes
                for idx, old_d, new_d in changes:
                    distinct_counts[idx] = old_d
                # Revert hist changes
                for idx in affected_k4s:
                    hist[idx][old_col] += 1
                    hist[idx][new_col] -= 1
            
            T *= decay
        
        return False, best_colors

    # Try k=13
    valid, colors_13 = search(13, 40.0)
    if valid:
        # Convert to list
        return [int(colors_13[i][j]) for i in range(16) for j in range(i+1, 16)]
    
    # Try k=14
    if time.time() < deadline - 20:
        valid, colors_14 = search(14, 40.0)
        if valid:
            return [int(colors_14[i][j]) for i in range(16) for j in range(i+1, 16)]
        
    # Try k=15
    if time.time() < deadline - 10:
        valid, colors_15 = search(15, 30.0)
        if valid:
            return [int(colors_15[i][j]) for i in range(16) for j in range(i+1, 16)]
            
    # Fallback to seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
