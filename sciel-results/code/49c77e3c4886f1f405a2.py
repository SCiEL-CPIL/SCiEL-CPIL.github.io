import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 115
    best_n = 0
    best_colors = []

    # Fallback: n=17 quadratic residue coloring (Paley)
    def fallback():
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(0 if (j - i) % 17 in squares else 1)
        return {'n': n, 'colors': colors}

    # 1. Fast validity check using NumPy
    def is_valid_coloring(n, colors):
        """
        Check if the coloring of K_n has no monochromatic K2,2,2.
        colors: list or np.array of 0/1 of length n*(n-1)/2
        """
        if len(colors) != n * (n - 1) // 2:
            return False
        
        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i, j] = c
                adj[j, i] = c
                idx += 1

        # K2,2,2 is a complete tripartite graph K_{2,2,2}.
        # It has 6 vertices partitioned into 3 sets of 2.
        # All 12 edges between the sets must have the same color.
        # We check all 6-vertex subsets.
        # Optimization: Use bitmasks or vectorized ops if possible, 
        # but for n<=31, a loop over combinations is ~ C(31,6)*15 ~ 2M ops.
        # This might be slow in pure Python (~10-20s). 
        # We need a faster check for the search loop.
        
        # For the final verification, this is okay if we don't call it too often.
        # But we need a faster "violation count" for local search.
        
        partitions = [
            [(0,1),(2,3),(4,5)], [(0,1),(2,4),(3,5)], [(0,1),(2,5),(3,4)],
            [(0,2),(1,3),(4,5)], [(0,2),(1,4),(3,5)], [(0,2),(1,5),(3,4)],
            [(0,3),(1,2),(4,5)], [(0,3),(1,4),(2,5)], [(0,3),(1,5),(2,4)],
            [(0,4),(1,2),(3,5)], [(0,4),(1,3),(2,5)], [(0,4),(1,5),(2,3)],
            [(0,5),(1,2),(3,4)], [(0,5),(1,3),(2,4)], [(0,5),(1,4),(2,3)]
        ]
        
        for vertices in combinations(range(n), 6):
            v = vertices
            for part in partitions:
                p0 = [v[part[0][0]], v[part[0][1]]]
                p1 = [v[part[1][0]], v[part[1][1]]]
                p2 = [v[part[2][0]], v[part[2][1]]]
                
                c0 = 0
                c1 = 0
                # Edges between p0, p1
                if adj[p0[0], p1[0]] == 0: c0 += 1
                else: c1 += 1
                if adj[p0[0], p1[1]] == 0: c0 += 1
                else: c1 += 1
                if adj[p0[1], p1[0]] == 0: c0 += 1
                else: c1 += 1
                if adj[p0[1], p1[1]] == 0: c0 += 1
                else: c1 += 1
                
                # Edges between p0, p2
                if adj[p0[0], p2[0]] == 0: c0 += 1
                else: c1 += 1
                if adj[p0[0], p2[1]] == 0: c0 += 1
                else: c1 += 1
                if adj[p0[1], p2[0]] == 0: c0 += 1
                else: c1 += 1
                if adj[p0[1], p2[1]] == 0: c0 += 1
                else: c1 += 1
                
                # Edges between p1, p2
                if adj[p1[0], p2[0]] == 0: c0 += 1
                else: c1 += 1
                if adj[p1[0], p2[1]] == 0: c0 += 1
                else: c1 += 1
                if adj[p1[1], p2[0]] == 0: c0 += 1
                else: c1 += 1
                if adj[p1[1], p2[1]] == 0: c0 += 1
                else: c1 += 1
                
                if c0 == 12 or c1 == 12:
                    return False
        return True

    # 2. Faster violation counting for Local Search
    # Instead of checking all 6-sets, we can track violations incrementally.
    # However, building the instance list is the bottleneck.
    # Alternative: Randomized check? No, we need certainty.
    # Let's use a hybrid: 
    #   - Use a "conflict" metric that is cheaper to compute.
    #   - Or, precompute the "critical" structures.
    
    # Actually, for n=29, C(29,6) = 475k. 475k * 15 = 7M checks.
    # In Python, 7M checks take ~5-10 seconds.
    # We can't do this in every step of local search.
    
    # New Strategy: 
    # 1. Generate a random coloring.
    # 2. Use a "repair" strategy: Find one violation, fix it, repeat.
    #    To find a violation fast, we can sample random 6-sets until we find one.
    #    If the graph is close to valid, we find violations quickly.
    #    If it's far, we might not.
    
    def count_violations_sampled(n, adj, samples=1000):
        """
        Estimate number of violations by sampling.
        Returns 0 if no violations found in sample (heuristic).
        """
        partitions = [
            [(0,1),(2,3),(4,5)], [(0,1),(2,4),(3,5)], [(0,1),(2,5),(3,4)],
            [(0,2),(1,3),(4,5)], [(0,2),(1,4),(3,5)], [(0,2),(1,5),(3,4)],
            [(0,3),(1,2),(4,5)], [(0,3),(1,4),(2,5)], [(0,3),(1,5),(2,4)],
            [(0,4),(1,2),(3,5)], [(0,4),(1,3),(2,5)], [(0,4),(1,5),(2,3)],
            [(0,5),(1,2),(3,4)], [(0,5),(1,3),(2,4)], [(0,5),(1,4),(2,3)]
        ]
        
        found_viol = 0
        for _ in range(samples):
            vertices = random.sample(range(n), 6)
            for part in partitions:
                p0 = [vertices[part[0][0]], vertices[part[0][1]]]
                p1 = [vertices[part[1][0]], vertices[part[1][1]]]
                p2 = [vertices[part[2][0]], vertices[part[2][1]]]
                
                c0 = 0
                # Edges p0-p1
                if adj[p0[0], p1[0]] == 0: c0 += 1
                if adj[p0[0], p1[1]] == 0: c0 += 1
                if adj[p0[1], p1[0]] == 0: c0 += 1
                if adj[p0[1], p1[1]] == 0: c0 += 1
                # Edges p0-p2
                if adj[p0[0], p2[0]] == 0: c0 += 1
                if adj[p0[0], p2[1]] == 0: c0 += 1
                if adj[p0[1], p2[0]] == 0: c0 += 1
                if adj[p0[1], p2[1]] == 0: c0 += 1
                # Edges p1-p2
                if adj[p1[0], p2[0]] == 0: c0 += 1
                if adj[p1[0], p2[1]] == 0: c0 += 1
                if adj[p1[1], p2[0]] == 0: c0 += 1
                if adj[p1[1], p2[1]] == 0: c0 += 1
                
                if c0 == 12 or c0 == 0:
                    found_viol += 1
                    break
        return found_viol

    def local_search(n, initial_colors, time_limit):
        """
        Local search with sampled violation detection.
        """
        colors = np.array(initial_colors, dtype=np.int8)
        adj = np.zeros((n, n), dtype=np.int8)
        
        def update_adj():
            nonlocal adj
            adj[:] = 0
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[i, j] = c
                    adj[j, i] = c
                    idx += 1
        update_adj()
        
        # Initial check
        viol_count = count_violations_sampled(n, adj, 500)
        if viol_count == 0:
            # Verify fully
            if is_valid_coloring(n, colors.tolist()):
                return colors.tolist(), n
            # If sampled 0 but full check fails, we are in a rare local min or sampling error.
            # Proceed to search anyway.
        
        step = 0
        max_steps = 10000
        last_improvement = 0
        best_viol = viol_count
        
        while step < max_steps:
            if time.time() > time_limit - 2:
                break
            
            # Find a violation by sampling
            vertices = None
            part = None
            for _ in range(50): # Try 50 samples to find a violation
                v = random.sample(range(n), 6)
                for p in partitions:
                    p0 = [v[p[0][0]], v[p[0][1]]]
                    p1 = [v[p[1][0]], v[p[1][1]]]
                    p2 = [v[p[2][0]], v[p[2][1]]]
                    
                    c0 = 0
                    if adj[p0[0], p1[0]] == 0: c0 += 1
                    if adj[p0[0], p1[1]] == 0: c0 += 1
                    if adj[p0[1], p1[0]] == 0: c0 += 1
                    if adj[p0[1], p1[1]] == 0: c0 += 1
                    if adj[p0[0], p2[0]] == 0: c0 += 1
                    if adj[p0[0], p2[1]] == 0: c0 += 1
                    if adj[p0[1], p2[0]] == 0: c0 += 1
                    if adj[p0[1], p2[1]] == 0: c0 += 1
                    if adj[p1[0], p2[0]] == 0: c0 += 1
                    if adj[p1[0], p2[1]] == 0: c0 += 1
                    if adj[p1[1], p2[0]] == 0: c0 += 1
                    if adj[p1[1], p2[1]] == 0: c0 += 1
                    
                    if c0 == 12 or c0 == 0:
                        vertices = v
                        part = p
                        break
                if vertices is not None:
                    break
            
            if vertices is None:
                # No violation found in samples. Likely valid or very close.
                if is_valid_coloring(n, colors.tolist()):
                    return colors.tolist(), n
                # If not valid, we might be stuck. Try random flips.
                # Or break and return best.
                break
                
            # We found a violation. Flip a random edge in it.
            # The edges are between p0-p1, p0-p2, p1-p2.
            p0 = [vertices[part[0][0]], vertices[part[0][1]]]
            p1 = [vertices[part[1][0]], vertices[part[1][1]]]
            p2 = [vertices[part[2][0]], vertices[part[2][1]]]
            
            edges = []
            for u in p0:
                for w in p1:
                    edges.append((min(u,w), max(u,w)))
            for u in p0:
                for w in p2:
                    edges.append((min(u,w), max(u,w)))
            for u in p1:
                for w in p2:
                    edges.append((min(u,w), max(u,w)))
            
            # Pick random edge
            u, w = random.choice(edges)
            
            # Flip color
            # Need to find index in colors array
            # Index of (u, w) in lexicographic order
            # idx = u*n - u*(u+1)//2 + (w - u - 1)
            idx = u * (2*n - u - 1) // 2 + (w - u - 1)
            
            colors[idx] = 1 - colors[idx]
            adj[u, w] = colors[idx]
            adj[w, u] = colors[idx]
            
            step += 1
            
            # Every 50 steps, re-evaluate violation count to track progress
            if step % 50 == 0:
                viol_count = count_violations_sampled(n, adj, 100)
                if viol_count < best_viol:
                    best_viol = viol_count
                    last_improvement = step
                elif step - last_improvement > 500:
                    # Stagnation, restart or break?
                    # Let's just keep going for now, but maybe randomize more.
                    pass
            
        # Final check
        if is_valid_coloring(n, colors.tolist()):
            return colors.tolist(), n
        return None, None

    # 3. Greedy Construction
    def greedy_construct(n_target):
        """
        Try to build a valid coloring for n_target vertices.
        """
        # Start with n=17 Paley
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(0 if (j - i) % 17 in squares else 1)
        
        # Extend to n_target
        # This is complex. Let's just try local search on random/Paley starts for larger n.
        pass

    # 4. Main Search Loop
    # Try n from 29 down to 20
    for n in range(29, 19, -1):
        if time.time() > deadline - 10:
            break
            
        # Try Paley-like initialization if possible?
        # For n=29, we can use the Paley graph of order 29?
        # 29 is prime, 29 % 4 == 1. So Paley(29) exists.
        if n == 29:
            squares = {x * x % 29 for x in range(1, 29)}
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(0 if (j - i) % 29 in squares else 1)
            
            # Check if Paley(29) is valid
            if is_valid_coloring(n, colors):
                return {'n': n, 'colors': colors}
            
            # If not, try local search from it
            time_left = deadline - time.time() - 5
            if time_left > 5:
                res_colors, res_n = local_search(n, colors, time.time() + time_left)
                if res_colors is not None:
                    return {'n': res_n, 'colors': res_colors}
        
        # Try random initialization
        time_left = deadline - time.time() - 5
        if time_left > 5:
            num_edges = n * (n - 1) // 2
            random_colors = [random.randint(0, 1) for _ in range(num_edges)]
            res_colors, res_n = local_search(n, random_colors, time.time() + time_left)
            if res_colors is not None:
                return {'n': res_n, 'colors': res_colors}

    # Fallback
    return fallback()

# Precompute partitions for global access in local_search if needed, 
# but local_search defines it locally or uses global. 
# Let's define it globally to avoid re-creation.
partitions = [
    [(0,1),(2,3),(4,5)], [(0,1),(2,4),(3,5)], [(0,1),(2,5),(3,4)],
    [(0,2),(1,3),(4,5)], [(0,2),(1,4),(3,5)], [(0,2),(1,5),(3,4)],
    [(0,3),(1,2),(4,5)], [(0,3),(1,4),(2,5)], [(0,3),(1,5),(2,4)],
    [(0,4),(1,2),(3,5)], [(0,4),(1,3),(2,5)], [(0,4),(1,5),(2,3)],
    [(0,5),(1,2),(3,4)], [(0,5),(1,3),(2,4)], [(0,5),(1,4),(2,3)]
]

# Re-define local_search to use global partitions
def local_search(n, initial_colors, time_limit):
    colors = np.array(initial_colors, dtype=np.int8)
    adj = np.zeros((n, n), dtype=np.int8)
    
    def update_adj():
        nonlocal adj
        adj[:] = 0
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i, j] = c
                adj[j, i] = c
                idx += 1
    update_adj()
    
    # Helper to check a specific partition of 6 vertices
    def check_violation(v, p):
        p0 = [v[p[0][0]], v[p[0][1]]]
        p1 = [v[p[1][0]], v[p[1][1]]]
        p2 = [v[p[2][0]], v[p[2][1]]]
        
        c0 = 0
        if adj[p0[0], p1[0]] == 0: c0 += 1
        if adj[p0[0], p1[1]] == 0: c0 += 1
        if adj[p0[1], p1[0]] == 0: c0 += 1
        if adj[p0[1], p1[1]] == 0: c0 += 1
        if adj[p0[0], p2[0]] == 0: c0 += 1
        if adj[p0[0], p2[1]] == 0: c0 += 1
        if adj[p0[1], p2[0]] == 0: c0 += 1
        if adj[p0[1], p2[1]] == 0: c0 += 1
        if adj[p1[0], p2[0]] == 0: c0 += 1
        if adj[p1[0], p2[1]] == 0: c0 += 1
        if adj[p1[1], p2[0]] == 0: c0 += 1
        if adj[p1[1], p2[1]] == 0: c0 += 1
        
        return c0 == 12 or c0 == 0

    step = 0
    max_steps = 20000
    
    while step < max_steps:
        if time.time() > time_limit - 2:
            break
        
        # Find a violation
        found = False
        for _ in range(20): # Sample 20 random 6-sets
            v = random.sample(range(n), 6)
            for p in partitions:
                if check_violation(v, p):
                    # Found violation
                    p0 = [v[p[0][0]], v[p[0][1]]]
                    p1 = [v[p[1][0]], v[p[1][1]]]
                    p2 = [v[p[2][0]], v[p[2][1]]]
                    
                    edges = []
                    for u in p0:
                        for w in p1:
                            edges.append((min(u,w), max(u,w)))
                    for u in p0:
                        for w in p2:
                            edges.append((min(u,w), max(u,w)))
                    for u in p1:
                        for w in p2:
                            edges.append((min(u,w), max(u,w)))
                    
                    u, w = random.choice(edges)
                    idx = u * (2*n - u - 1) // 2 + (w - u - 1)
                    colors[idx] = 1 - colors[idx]
                    adj[u, w] = colors[idx]
                    adj[w, u] = colors[idx]
                    
                    found = True
                    break
            if found:
                break
        
        if not found:
            # No violation found in samples. Check full validity.
            if is_valid_coloring(n, colors.tolist()):
                return colors.tolist(), n
            # If still invalid, we might be stuck in a local minimum where violations are rare.
            # Try a random flip to escape.
            idx = random.randint(0, len(colors)-1)
            # Decode idx to (u, w)
            # This is slower. Just pick random u, w.
            u = random.randint(0, n-1)
            w = random.randint(0, n-1)
            if u == w: continue
            if u > w: u, w = w, u
            idx = u * (2*n - u - 1) // 2 + (w - u - 1)
            colors[idx] = 1 - colors[idx]
            adj[u, w] = colors[idx]
            adj[w, u] = colors[idx]
        
        step += 1
        
    # Final check
    if is_valid_coloring(n, colors.tolist()):
        return colors.tolist(), n
    return None, None

# Re-define construct to use the global local_search
def construct():
    deadline = time.time() + 115
    best_n = 0
    best_colors = []

    def fallback():
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(0 if (j - i) % 17 in squares else 1)
        return {'n': n, 'colors': colors}

    # Try n=29 with Paley
    n = 29
    squares = {x * x % 29 for x in range(1, 29)}
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % 29 in squares else 1)
    
    if is_valid_coloring(n, colors):
        return {'n': n, 'colors': colors}
    
    # Local search from Paley(29)
    time_left = deadline - time.time() - 5
    if time_left > 5:
        res_colors, res_n = local_search(n, colors, time.time() + time_left)
        if res_colors is not None:
            return {'n': res_n, 'colors': res_colors}
    
    # Try n=28 with random
    n = 28
    time_left = deadline - time.time() - 5
    if time_left > 5:
        num_edges = n * (n - 1) // 2
        random_colors = [random.randint(0, 1) for _ in range(num_edges)]
        res_colors, res_n = local_search(n, random_colors, time.time() + time_left)
        if res_colors is not None:
            return {'n': res_n, 'colors': res_colors}
            
    # Try n=27 with random
    n = 27
    time_left = deadline - time.time() - 5
    if time_left > 5:
        num_edges = n * (n - 1) // 2
        random_colors = [random.randint(0, 1) for _ in range(num_edges)]
        res_colors, res_n = local_search(n, random_colors, time.time() + time_left)
        if res_colors is not None:
            return {'n': res_n, 'colors': res_colors}

    return fallback()
