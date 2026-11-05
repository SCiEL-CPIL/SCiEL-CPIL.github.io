import random
import time

def construct():
    # Target n values to try, starting from high
    target_ns = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
    
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    for n in target_ns:
        if time.time() - start_time > time_limit:
            break
            
        # Initialize random coloring
        # Colors 0..4
        colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
        
        # Build adjacency sets for each color
        # adj[c] is a list of sets, adj[c][i] contains j if edge (i,j) is color c
        adj = [ [set() for _ in range(n)] for _ in range(5) ]
        
        # Map (i,j) to index
        # Index = i*n - i*(i+1)//2 + (j-i-1) ? 
        # Simpler: just iterate and assign
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
                
        # Function to calculate number of C4s in a color class
        # We maintain total_c4s as sum over colors
        # For efficiency, we calculate delta when swapping
        
        def count_c4s_in_color(c):
            count = 0
            # Iterate over all edges (u,v) in color c
            # For each edge, count common neighbors x,y that form C4?
            # No, count C4s containing edge (u,v).
            # As derived: sum over x in N(v), y in N(u) of [ (x,y) in E ]
            # This counts each C4 4 times (once for each edge).
            # So total C4s = total_count / 4.
            # We can just track the "4*count" value for integer arithmetic.
            total = 0
            for u in range(n):
                for v in adj[c][u]:
                    if u < v: # Only count each edge once? 
                        # The formula sum_{x in N(v)} sum_{y in N(u)} [ (x,y) in E ]
                        # counts the C4 u-v-x-y-u.
                        # This specific C4 is counted for edge (u,v).
                        # It is also counted for edge (v,x), (x,y), (y,u).
                        # So if we sum over all edges, we get 4 * (# C4s).
                        pass
            
            # Efficient counting:
            # For each pair of vertices (u, v), let d(u,v) be common neighbors.
            # Number of C4s is sum over all pairs? No.
            # Standard formula: #C4 = sum_{u,v} binom(d(u,v), 2) / 2 ?
            # No, that counts C4s where u,v are opposite?
            # Let's stick to the edge-based delta for SA.
            # For initialization, we can just compute it once.
            
            # To compute initial count:
            # For each color c:
            #   For each u:
            #     For each v in adj[c][u] with v > u:
            #        common = adj[c][u] & adj[c][v]
            #        # This counts paths u-x-v.
            #        # A C4 u-x-v-y-u requires x and y to be connected.
            #        # This is not just binom(common, 2).
            
            # Let's use the property:
            # #C4 = 1/4 * sum_{edges (u,v)} ( number of edges (x,y) such that x in N(v), y in N(u) )
            # This is O(m * deg^2) roughly.
            
            init_count = 0
            for u in range(n):
                for v in adj[c][u]:
                    if u < v:
                        # Count edges (x,y) with x in N(v), y in N(u)
                        # x must be in adj[c][v], y in adj[c][u]
                        # and (x,y) in E
                        count = 0
                        for x in adj[c][v]:
                            if x == u: continue
                            for y in adj[c][u]:
                                if y == v: continue
                                if x == y: continue
                                if y in adj[c][x]:
                                    count += 1
                        init_count += count
            return init_count // 4

        # We will maintain total_c4s
        # To avoid expensive init, we can just run SA and let it converge.
        # But we need a starting energy.
        # For n=26, random init has many C4s.
        
        # Let's just run SA and track if we hit 0.
        
        # To speed up, we only check for validity (0 C4s) periodically or on improvement.
        # Actually, we need the energy to guide SA.
        
        # Let's compute initial energy once.
        total_energy = 0
        for c in range(5):
            total_energy += count_c4s_in_color(c)
            
        best_energy = total_energy
        best_colors = colors.copy()
        
        # SA Parameters
        T = 100.0
        cooling = 0.999
        min_T = 0.01
        
        current_colors = colors.copy()
        
        # Precompute edge indices for faster access?
        # We need to map (i,j) to index in 'colors' list.
        # idx(i,j) = i*n - i*(i+1)//2 + (j-i-1)
        
        def get_idx(i, j):
            if i > j: i, j = j, i
            return i * n - i * (i + 1) // 2 + (j - i - 1)
            
        steps = 0
        max_steps = 50000
        time_per_n = (time_limit - (time.time() - start_time)) / (len(target_ns) - target_ns.index(n))
        # Roughly distribute time
        
        while T > min_T and steps < max_steps:
            if time.time() - start_time > time_limit:
                break
                
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j: continue
            
            idx_ij = get_idx(i, j)
            c_old = current_colors[idx_ij]
            c_new = random.randint(0, 4)
            if c_old == c_new: continue
            
            # Calculate Delta Energy
            # Remove (i,j) from c_old
            # Add (i,j) to c_new
            
            # Helper to count C4s contributed by edge (i,j) in color c
            # This is the number of (x,y) edges in G_c such that x in N_c(j), y in N_c(i)
            def edge_c4_contribution(c, i, j):
                # N_c(j) and N_c(i)
                # Note: (i,j) is currently in adj[c]
                # We need to count x in adj[c][j] \ {i}, y in adj[c][i] \ {j}
                # such that (x,y) in E_c
                cnt = 0
                for x in adj[c][j]:
                    if x == i: continue
                    for y in adj[c][i]:
                        if y == j: continue
                        if x == y: continue
                        if y in adj[c][x]:
                            cnt += 1
                return cnt
                
            # Delta for c_old (removal)
            # Before removal, (i,j) is in c_old.
            # The C4s using (i,j) are counted in total_energy.
            # Removing (i,j) removes these C4s.
            # But wait, removing (i,j) also changes N(i) and N(j), which affects C4s NOT using (i,j)?
            # No, C4s not using (i,j) are unaffected by the presence/absence of (i,j) 
            # UNLESS the C4 uses (i,j) as a "diagonal"? No, C4s are cycles.
            # A C4 is a set of 4 edges. If (i,j) is not one of them, it's unaffected.
            # So only C4s containing (i,j) are affected.
            
            delta_old = -edge_c4_contribution(c_old, i, j)
            
            # For c_new (addition)
            # Before addition, (i,j) is NOT in c_new.
            # Adding (i,j) creates new C4s containing (i,j).
            # We must calculate contribution based on state BEFORE adding.
            delta_new = edge_c4_contribution(c_new, i, j)
            
            delta_E = delta_old + delta_new
            
            # Acceptance
            if delta_E < 0 or random.random() < pow(2.718, -delta_E / T):
                # Apply move
                current_colors[idx_ij] = c_new
                
                # Update adj
                adj[c_old][i].remove(j)
                adj[c_old][j].remove(i)
                adj[c_new][i].add(j)
                adj[c_new][j].add(i)
                
                total_energy += delta_E
                
                if total_energy < best_energy:
                    best_energy = total_energy
                    best_colors = current_colors.copy()
                    if best_energy == 0:
                        # Found valid coloring
                        return {'n': n, 'colors': best_colors}
                        
            T *= cooling
            steps += 1
            
        # After SA, if we didn't find 0, check if best is 0
        if best_energy == 0:
            return {'n': n, 'colors': best_colors}
            
    # If we fail for all n, return the best we found or seed
    # The loop returns on success. If it finishes, we need a fallback.
    # But the loop tries n=26 down to 11.
    # If n=11 is tried, it will likely find a solution quickly.
    
    # Fallback to seed-like structure if all fail (unlikely)
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
