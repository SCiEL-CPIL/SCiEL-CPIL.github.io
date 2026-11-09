import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Fallback: A valid coloring for n=9 is easy to construct or find quickly.
    # We will start with n=9 and try to grow or find better ones.
    # Actually, let's just run SA for various n and keep the best valid one.
    
    best_n = 0
    best_colors = []
    
    # We prioritize finding a valid coloring for the largest n possible.
    # Known: R_4(C6) > 17 is known? The prompt says "best known colouring has 17 vertices".
    # So we should aim for n=17, 18, 19, 20.
    
    # Helper to generate initial random coloring
    def random_coloring(n):
        # Returns list of colors for edges in lexicographic order
        return [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
    
    # Helper to convert edge list to adjacency matrices
    def to_adj_matrices(n, colors):
        # adj[c][i][j] = 1 if edge (i,j) has color c
        adj = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i][j] = 1
                adj[c][j][i] = 1
                idx += 1
        return adj
    
    # Helper to count C6s in a single color class
    # A C6 is a cycle of length 6.
    # We can count closed walks of length 6 and subtract those that are not simple cycles?
    # That's hard. 
    # Alternative: Enumerate. For n<=20, n^6 is 64M, which is too slow in Python if done naively for every step.
    # However, we can optimize.
    # Let's use a faster heuristic: Check for C6s by checking paths of length 5.
    # Actually, for SA, we just need to know if we *introduced* a C6 or *removed* one.
    # Delta C6 count:
    # When we change edge (u,v) from c_old to c_new:
    # 1. Remove (u,v) from c_old graph. Count C6s lost.
    # 2. Add (u,v) to c_new graph. Count C6s gained.
    
    # To count C6s involving a specific edge (u,v) in graph G:
    # We need paths of length 5 from u to v in G - (u,v).
    # Let A be the adjacency matrix of G without (u,v).
    # Number of such paths is (A^5)[u][v].
    # Each C6 containing (u,v) corresponds to a path of length 5 from u to v.
    # So, C6s containing (u,v) = (A^5)[u][v].
    
    # Matrix power 5 for n=20 is fast with NumPy.
    
    def count_c6s_involving_edge(adj_matrices, u, v, c):
        # adj_matrices[c] is the current adjacency matrix for color c
        # We need to calculate paths of length 5 from u to v in color c graph, 
        # but excluding the edge (u,v) itself if it exists.
        # In our SA move, we are changing (u,v). 
        # If we are ADDING (u,v) to color c, we check if it forms C6s.
        # The C6s formed are exactly the paths of length 5 from u to v in the graph WITHOUT (u,v).
        
        A = adj_matrices[c].copy()
        # Ensure (u,v) is 0 in A for the calculation
        A[u][v] = 0
        A[v][u] = 0
        
        # Calculate A^5
        # For small n, explicit multiplication might be faster than generic power?
        # n=20 is small. A^2, A^4, A^5.
        A2 = A @ A
        A4 = A2 @ A2
        A5 = A4 @ A
        return int(A5[u][v])
        
    def total_c6_count(adj_matrices, n):
        # Sum of C6s in all 4 colors.
        # Note: A C6 is counted 6 times in the "edge involvement" sum?
        # No, a C6 has 6 edges. Each edge is involved in exactly 1 C6 (the one itself).
        # So if we sum count_c6s_involving_edge for all edges in color c, we get 6 * (#C6s in c).
        # So total C6s = sum( count_c6s_involving_edge(...) ) / 6.
        
        total = 0
        for c in range(4):
            A = adj_matrices[c]
            # We can compute this more efficiently by summing (A^5)[i][j] for all i<j?
            # No, (A^5)[i][j] counts paths of length 5. 
            # The number of C6s is exactly 1/6 * sum_{i!=j} (A^5)[i][j]? 
            # Let's verify. 
            # A C6 is a sequence v1..v6. 
            # It contributes 1 to (A^5)[v1][v2] (path v2-v3-v4-v5-v6-v1? No, path length 5 is 6 vertices).
            # Path length 5 from u to v has 6 vertices.
            # If we sum (A^5)[i][j] for all i,j, we count every C6 6 times (once for each starting vertex).
            # So #C6s = trace(A^6) / 12? No, trace(A^6) counts closed walks.
            # Let's stick to the edge-based delta for SA, but for initialization we need the total.
            
            # Efficient total count:
            # A5 = A^5
            # sum(A5) counts all paths of length 5.
            # Each C6 has 6 edges. Each edge (u,v) in the C6 is part of exactly 1 path of length 5 from u to v (the rest of the cycle).
            # Wait, if the graph has other edges, there might be more paths.
            # But we want to count *existing* C6s.
            # The number of C6s in graph G is 1/6 * sum_{u,v} (A^5)[u][v] is INCORRECT because A^5 counts ALL paths, not just those that close a cycle.
            # Actually, (A^5)[u][v] is the number of walks of length 5 from u to v.
            # A C6 corresponds to a walk of length 6 that is a simple cycle.
            # This is tricky to count directly from A^5 without inclusion-exclusion.
            
            # Given n<=20, we can just enumerate C6s for the initial state and total count if needed, 
            # but for SA, we only need DELTA.
            # For the initial score, we can just return a large number or compute it roughly.
            # Let's just compute the sum of (A^5)[i][j] for i<j as a proxy, or just 0 if we don't care about absolute value, only relative.
            # But SA needs a fitness function.
            # Let's use: Fitness = sum_{c} sum_{i<j} (A_c^5)[i][j].
            # This is proportional to the number of "potential" C6s.
            # Actually, if we add an edge (u,v) to color c, the increase in this sum is exactly the number of C6s created.
            # So we can use `score = sum(A_c^5)` as the objective to minimize.
            
            pass
        return 0 # Placeholder, we will compute delta only in SA

    # Let's refine the SA approach.
    # Objective: Minimize S = sum_{c=0}^3 sum_{i<j} (A_c^5)[i][j].
    # When we change edge (u,v) from c1 to c2:
    # Delta_S = (New_S in c2) - (Old_S in c1)
    # New_S in c2 involves adding (u,v) to A2.
    # The change in sum_{i<j} (A2^5)[i][j] when adding edge (u,v) to A2:
    # The new terms in A2^5 that involve the new edge (u,v) are paths that use (u,v).
    # A path of length 5 using (u,v) can have (u,v) at position 1, 2, 3, 4, or 5.
    # However, due to symmetry, the number of paths of length 5 from u to v in A2 (without (u,v)) is (A2^5)[u][v].
    # These paths, combined with (u,v), form closed walks of length 6.
    # Specifically, each path of length 5 from u to v in A2 (without (u,v)) creates 2 new entries in the sum?
    # Let's look at the term (A^5)[u][v]. It counts paths u->...->v.
    # If we add edge (u,v), does it affect (A^5)[u][v]?
    # No, (A^5)[u][v] is calculated from the matrix BEFORE adding the edge.
    # But the objective is the sum of the FINAL matrix power.
    # This is getting complicated.
    
    # Simpler Objective: Number of C6s.
    # We can maintain the count of C6s incrementally.
    # When adding (u,v) to color c:
    # New C6s = number of paths of length 5 from u to v in G_c - (u,v).
    # This is exactly (A_c^5)[u][v] where A_c is the current adjacency matrix of color c WITHOUT (u,v).
    # When removing (u,v) from color c:
    # Removed C6s = number of paths of length 5 from u to v in G_c - (u,v).
    # Same calculation.
    
    # So, we maintain `c6_counts[c]` = number of C6s in color c.
    # Total C6s = sum(c6_counts).
    
    def update_c6_count(adj_matrices, u, v, c, add_edge):
        # Returns the change in number of C6s for color c
        # and updates adj_matrices[c]
        A = adj_matrices[c]
        
        # Calculate paths of length 5 from u to v in current graph (which does NOT have (u,v) if we are adding, or DOES have it if we are removing?
        # Wait.
        # If we are ADDING (u,v): The current graph A does NOT have (u,v).
        # We compute paths = (A^5)[u][v].
        # Then we add (u,v) to A.
        # The number of new C6s is `paths`.
        
        # If we are REMOVING (u,v): The current graph A HAS (u,v).
        # We need paths in A - (u,v).
        # So we temporarily remove it, compute, then restore? Or compute with it removed.
        
        # Let's standardize: 
        # 1. Ensure A[u][v] == 0 and A[v][u] == 0 for the calculation.
        # 2. Compute P = (A^5)[u][v].
        # 3. If adding: c6_count += P. Then set A[u][v]=1, A[v][u]=1.
        # 4. If removing: c6_count -= P. Then set A[u][v]=0, A[v][u]=0.
        
        # Temporarily clear
        had_u_v = A[u][v]
        A[u][v] = 0
        A[v][u] = 0
        
        # Compute A^5
        # Optimization: For n=20, this is small.
        A2 = A @ A
        A4 = A2 @ A2
        A5 = A4 @ A
        
        P = int(A5[u][v])
        
        # Restore state based on action
        if add_edge:
            A[u][v] = 1
            A[v][u] = 1
            return P
        else:
            # A[u][v] and A[v][u] are already 0
            return -P

    # Initialize
    # Try n from 17 down to 9? Or start with 9 and grow?
    # Starting with a valid small one and growing is hard in SA.
    # Better: Start with random n=17, minimize C6s. If 0, return.
    # If not 0, try n=18? No, n=17 is the target.
    # Let's try n=17 first.
    
    for n in [17, 18, 19, 20]:
        if time.time() - start_time > deadline - 10: # Leave 10s for verification
            break
            
        # Run SA for this n
        # Initial random coloring
        colors = random_coloring(n)
        adj = to_adj_matrices(n, colors)
        
        # Initialize C6 counts
        c6_counts = [0] * 4
        # We need the initial total.
        # We can compute it by summing P for all edges?
        # That's O(E * n^3) which is 300 * 8000 = 2.4M ops. Fast enough.
        for c in range(4):
            A = adj[c]
            # We need sum of (A^5)[i][j] for all i<j?
            # No, we need the count of C6s.
            # As established, #C6s = 1/6 * sum_{u,v} (A^5)[u][v] is WRONG.
            # Let's just compute it by iterating edges?
            # No, let's just set initial score to a large number and rely on deltas?
            # No, we need to know if we are at 0.
            
            # Let's compute initial C6 count properly.
            # For each edge (u,v) in color c, add (A_c - (u,v))^5 [u][v] to a sum?
            # No, that counts each C6 6 times.
            # So initial_c6s_c = (1/6) * sum_{(u,v) in E_c} ( (A_c - (u,v))^5 )[u][v]
            
            count_c = 0
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u][v] == 1:
                        # Compute P
                        A_tmp = A.copy()
                        A_tmp[u][v] = 0
                        A_tmp[v][u] = 0
                        A2 = A_tmp @ A_tmp
                        A4 = A2 @ A2
                        A5 = A4 @ A_tmp
                        count_c += int(A5[u][v])
            c6_counts[c] = count_c // 6
            # Note: count_c should be divisible by 6.
            
        total_c6 = sum(c6_counts)
        
        if total_c6 == 0:
            best_n = n
            best_colors = colors
            break
            
        # SA Loop
        current_temp = 1.0
        cooling_rate = 0.995
        min_temp = 0.01
        max_iters = 50000
        
        for it in range(max_iters):
            if time.time() - start_time > deadline - 5:
                break
                
            if current_temp < min_temp:
                break
                
            # Pick random edge
            idx = random.randint(0, n * (n - 1) // 2 - 1)
            # Map idx to (u, v)
            # This mapping is annoying. Let's precompute edge list.
            # We didn't precompute it in this scope. Let's do it once outside?
            # For n=17, E=136.
            
            # Generate (u,v) from idx
            # k = idx
            # u = floor((sqrt(8*k+1)-1)/2)
            # v = k - u*(u+1)/2 + u + 1
            # Let's just use a precomputed list for simplicity and speed in loop.
            
            # Actually, let's just pick u, v randomly.
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Find current color
            # We need to find the color of (u,v).
            # We can maintain a 2D array edge_color[u][v].
            # Let's add that.
            
            # Re-initializing SA state to include edge_color
            # Let's restart the SA block with better state management.
            pass

    # If we didn't break with a valid n, we might have a best_n from previous iterations?
    # Let's restructure the loop to keep track of best valid coloring found.
    
    # Refactored SA Block
    for n in [17, 18, 19, 20]:
        if time.time() - start_time > deadline - 15:
            break
            
        # Precompute edge list for this n
        edges = []
        edge_to_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                edge_to_idx[(i, j)] = idx
                idx += 1
        
        colors = random_coloring(n)
        adj = to_adj_matrices(n, colors)
        edge_color = [[-1] * n for _ in range(n)]
        for i, j in edges:
            c = colors[edge_to_idx[(i, j)]]
            edge_color[i][j] = c
            edge_color[j][i] = c
            
        c6_counts = [0] * 4
        # Compute initial C6 counts
        for c in range(4):
            A = adj[c]
            count_c = 0
            for u, v in edges:
                if A[u][v] == 1:
                    A_tmp = A.copy()
                    A_tmp[u][v] = 0
                    A_tmp[v][u] = 0
                    A2 = A_tmp @ A_tmp
                    A4 = A2 @ A2
                    A5 = A4 @ A_tmp
                    count_c += int(A5[u][v])
            c6_counts[c] = count_c // 6
            
        total_c6 = sum(c6_counts)
        
        if total_c6 == 0:
            best_n = n
            best_colors = colors
            break
            
        current_temp = 1.0
        cooling_rate = 0.999
        min_temp = 0.001
        max_iters = 100000
        
        for it in range(max_iters):
            if time.time() - start_time > deadline - 5:
                break
            if current_temp < min_temp:
                break
                
            # Pick random edge
            e_idx = random.randint(0, len(edges) - 1)
            u, v = edges[e_idx]
            c_old = edge_color[u][v]
            
            # Pick new color
            c_new = random.randint(0, 3)
            if c_new == c_old:
                continue
                
            # Calculate Delta
            # Remove from c_old
            delta_remove = update_c6_count(adj, u, v, c_old, add_edge=False)
            # Add to c_new
            delta_add = update_c6_count(adj, u, v, c_new, add_edge=True)
            
            delta_total = delta_remove + delta_add
            
            # Acceptance
            if delta_total <= 0 or random.random() < np.exp(-delta_total / current_temp):
                # Accept
                # Update edge_color and colors list
                edge_color[u][v] = c_new
                edge_color[v][u] = c_new
                colors[e_idx] = c_new
                c6_counts[c_old] += delta_remove
                c6_counts[c_new] += delta_add
                total_c6 += delta_total
                
                if total_c6 == 0:
                    best_n = n
                    best_colors = colors
                    break
            else:
                # Reject: Revert changes
                # update_c6_count already modified adj.
                # We need to revert adj.
                # Remove from c_new
                update_c6_count(adj, u, v, c_new, add_edge=False)
                # Add to c_old
                update_c6_count(adj, u, v, c_old, add_edge=True)
                
            current_temp *= cooling_rate
            
        if best_n == n:
            break
            
    # Fallback if no valid coloring found for n>=17
    if best_n == 0:
        # Try smaller n just in case, or return a known small one
        # A 4-coloring of K_9 without C6 is likely easy to find or known.
        # Let's just try n=9 with a quick SA or fixed pattern.
        # Pattern: (i+j) % 4? 
        # Let's just return a random valid one if we found one for smaller n?
        # We didn't try smaller n.
        # Let's try n=9 quickly.
        n = 9
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        colors = [0] * len(edges)
        # Try to find a valid one for n=9
        # Just random restarts
        for _ in range(1000):
            if time.time() - start_time > deadline - 2:
                break
            colors = random_coloring(n)
            adj = to_adj_matrices(n, colors)
            c6_counts = [0]*4
            valid = True
            for c in range(4):
                A = adj[c]
                # Check for C6
                # For n=9, we can just check all 6-subsets? No.
                # Use the matrix power trick.
                # If (A^6) has any diagonal element > 0, it has a closed walk.
                # But we need simple cycles.
                # Let's just use the edge count method.
                count_c = 0
                for u, v in edges:
                    if A[u][v] == 1:
                        A_tmp = A.copy()
                        A_tmp[u][v] = 0
                        A_tmp[v][u] = 0
                        A2 = A_tmp @ A_tmp
                        A4 = A2 @ A2
                        A5 = A4 @ A_tmp
                        count_c += int(A5[u][v])
                c6_counts[c] = count_c // 6
                if c6_counts[c] > 0:
                    valid = False
                    break
            if valid:
                best_n = n
                best_colors = colors
                break
                
    if best_n == 0:
        # Absolute fallback: n=1 is always valid.
        best_n = 1
        best_colors = []
        
    return {'n': best_n, 'colors': best_colors}
