import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to improve n
    # We will attempt to solve for n starting from 12 up to some limit
    # If we solve n, we keep it and try n+1
    
    # Helper to check if a coloring is valid and return n
    def is_valid(n, colors):
        # colors is list of length n*(n-1)//2
        # Check for C4 in each color class
        # Build adjacency sets for each color
        adj = [set() for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[c].add((i, j))
                # For common neighbor counting, we need to be careful
                idx += 1
        
        # More efficient check:
        # For each color c, build adjacency matrix/list
        # common[c][u][v]
        # If any common[c][u][v] >= 2, invalid.
        
        # Re-initialize common
        # This is O(n^2 * 5)
        # We can just use the local search cost function.
        # But for a quick check of a candidate:
        common = [[0]*n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                # Update common for pairs involving i and j?
                # No, this is just building the state.
                idx += 1
                
        # Actually, just re-run the cost calculation from scratch for validation
        # to be safe.
        return calculate_cost(n, colors) == 0

    def calculate_cost(n, colors):
        # Returns number of C4s
        # common[c][u][v]
        common = [[0]*n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                # Edge (i,j) is color c
                # It contributes to common[c][i][k] for all k where (j,k) is c?
                # No, common[c][i][k] counts neighbors w such that (i,w) and (k,w) are c.
                # This direct update is tricky to do in one pass without knowing all edges.
                # Better to build adjacency lists first.
                idx += 1
        
        # Build adjacency lists for each color
        adj = [[] for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[c].append((i, j))
                idx += 1
        
        # For each color, count common neighbors
        # For n <= 20, this is fast enough.
        cost = 0
        for c in range(5):
            # Build set of edges for fast lookup? Or just use sets of neighbors
            neighbors = [set() for _ in range(n)]
            for u, v in adj[c]:
                neighbors[u].add(v)
                neighbors[v].add(u)
            
            # For each pair (u, v), count common neighbors
            # To be efficient, iterate u, then v > u
            for u in range(n):
                for v in range(u+1, n):
                    # Intersection of neighbors[u] and neighbors[v]
                    # Since n is small, set intersection is fast
                    common_count = len(neighbors[u].intersection(neighbors[v]))
                    if common_count >= 2:
                        # Number of C4s is C(common_count, 2)
                        cost += common_count * (common_count - 1) // 2
        return cost

    def local_search(n, max_time=10.0):
        # Initialize random colors
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # State for fast updates
        # adj[c] = list of sets of neighbors?
        # Let's use the "common" array approach for O(n) updates.
        
        # common[c][u][v]
        common = [[[0]*n for _ in range(n)] for _ in range(5)]
        
        # Initialize common
        # We need to know the edges.
        # colors is a flat list.
        # Let's create a 2D array for colors for easier access
        # color_matrix[i][j]
        color_matrix = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                color_matrix[i][j] = c
                color_matrix[j][i] = c
                idx += 1
        
        # Build initial common
        # For each color c, for each edge (u,v) in c:
        #   For each w in c-neighborhood of u:
        #       common[c][v][w] += 1 ?
        # No.
        # common[c][u][v] is count of w s.t. (u,w) in c and (v,w) in c.
        # Iterate all triples? O(n^3). For n=20, 8000. Fast.
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    count = 0
                    for w in range(n):
                        if w == u or w == v: continue
                        if color_matrix[u][w] == c and color_matrix[v][w] == c:
                            count += 1
                    common[c][u][v] = count
                    common[c][v][u] = count
        
        current_cost = 0
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    k = common[c][u][v]
                    if k >= 2:
                        current_cost += k * (k - 1) // 2
        
        if current_cost == 0:
            return colors
        
        # Local Search Loop
        search_start = time.time()
        
        while time.time() - search_start < max_time:
            # Pick a random edge
            idx = random.randint(0, num_edges - 1)
            # Map idx to (i, j)
            # This mapping is O(1) if we precompute, or O(sqrt(n))
            # For simplicity, just iterate? No, random edge selection is important.
            # Let's precompute edge indices
            # Or just pick random i, j
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j: continue
            if i > j: i, j = j, i
            
            c_old = color_matrix[i][j]
            
            best_new_color = c_old
            best_cost = current_cost
            
            for c_new in range(5):
                if c_new == c_old: continue
                
                # Calculate delta cost
                delta = 0
                
                # Removing edge (i,j) from c_old
                # Affects common[c_old][i][w] and common[c_old][j][w] for all w
                # Specifically, for any w != i,j:
                # If (j,w) is c_old, then common[c_old][i][w] decreases by 1 (because (i,j) was a common neighbor link? No.)
                # Let's re-verify the update logic.
                # common[c][u][v] = |N_c(u) \cap N_c(v)|
                # If we remove edge (i,j) from color c:
                #   N_c(i) loses j. N_c(j) loses i.
                #   For any pair (u, v):
                #   If u=i, v=w (w!=j):
                #       common[c][i][w] changes only if w is in N_c(j) AND j is in N_c(w)?
                #       No. common[c][i][w] counts k such that k in N_c(i) and k in N_c(w).
                #       If k=j, then j in N_c(i) (true before) and j in N_c(w) (true if (w,j) in c).
                #       So if (w,j) is c, then common[c][i][w] decreases by 1.
                #   Similarly, common[c][j][w] decreases by 1 if (w,i) is c.
                #   What about common[c][i][j]?
                #       common[c][i][j] counts k such that k in N_c(i) and k in N_c(j).
                #       Removing (i,j) doesn't change N_c(i) \ {j} or N_c(j) \ {i} in a way that affects the count of *other* neighbors?
                #       Wait, N_c(i) is the set of neighbors.
                #       Before: j in N_c(i).
                #       After: j not in N_c(i).
                #       Does j count as a common neighbor of i and some w?
                #       Yes, if j in N_c(w).
                #       So common[c][i][w] decreases by 1 if (w,j) is c.
                #       Does it affect common[c][i][j]?
                #       common[c][i][j] counts k in N_c(i) \cap N_c(j).
                #       j is not in N_c(j) (no self loops).
                #       So j was never counted in common[c][i][j].
                #       So common[c][i][j] is unaffected by removing (i,j)?
                #       Let's check: N_c(i) and N_c(j).
                #       The intersection is {k | (i,k) in c and (j,k) in c}.
                #       k cannot be i or j.
                #       So the status of edge (i,j) itself does not appear in the sum for common[c][i][j].
                #       So common[c][i][j] is UNCHANGED by changing the color of (i,j).
                
                # So, for c_old:
                # For all w != i, j:
                #   if color_matrix[j][w] == c_old:
                #       common[c_old][i][w] -= 1
                #   if color_matrix[i][w] == c_old:
                #       common[c_old][j][w] -= 1
                
                # For c_new:
                # For all w != i, j:
                #   if color_matrix[j][w] == c_new:
                #       common[c_new][i][w] += 1
                #   if color_matrix[i][w] == c_new:
                #       common[c_new][j][w] += 1
                
                # We need to calculate the change in cost for these updates.
                # Cost term for pair (u,v) in color c is C(common[c][u][v], 2).
                # If common changes from k to k-1:
                #   Delta = C(k-1, 2) - C(k, 2) = (k-1)(k-2)/2 - k(k-1)/2 = (k-1)( (k-2) - k ) / 2 = (k-1)(-2)/2 = -(k-1).
                # If common changes from k to k+1:
                #   Delta = C(k+1, 2) - C(k, 2) = (k+1)k/2 - k(k-1)/2 = k( (k+1) - (k-1) ) / 2 = k(2)/2 = k.
                
                # Let's simulate the changes for c_old and c_new.
                
                # To avoid modifying the actual `common` array until we commit,
                # we can just calculate the delta.
                
                # But we need the current values of common[c][i][w] and common[c][j][w].
                
                delta_c_old = 0
                delta_c_new = 0
                
                for w in range(n):
                    if w == i or w == j: continue
                    
                    # c_old part
                    if color_matrix[j][w] == c_old:
                        k = common[c_old][i][w]
                        # k will become k-1
                        if k >= 2:
                            delta_c_old -= (k - 1)
                        # if k == 1, delta is 0 (1->0, C(0,2)-C(1,2) = 0)
                        # if k == 0, impossible (we are removing a common neighbor, so k>=1)
                    
                    if color_matrix[i][w] == c_old:
                        k = common[c_old][j][w]
                        if k >= 2:
                            delta_c_old -= (k - 1)
                    
                    # c_new part
                    if color_matrix[j][w] == c_new:
                        k = common[c_new][i][w]
                        # k will become k+1
                        delta_c_new += k
                    
                    if color_matrix[i][w] == c_new:
                        k = common[c_new][j][w]
                        delta_c_new += k
                        
                total_delta = delta_c_old + delta_c_new
                new_cost = current_cost + total_delta
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_new_color = c_new
                    
            # Apply best move
            if best_new_color != c_old:
                # Update state
                for w in range(n):
                    if w == i or w == j: continue
                    
                    if color_matrix[j][w] == c_old:
                        common[c_old][i][w] -= 1
                        common[c_old][w][i] -= 1
                    if color_matrix[i][w] == c_old:
                        common[c_old][j][w] -= 1
                        common[c_old][w][j] -= 1
                        
                    if color_matrix[j][w] == best_new_color:
                        common[best_new_color][i][w] += 1
                        common[best_new_color][w][i] += 1
                    if color_matrix[i][w] == best_new_color:
                        common[best_new_color][j][w] += 1
                        common[best_new_color][w][j] += 1
                        
                color_matrix[i][j] = best_new_color
                color_matrix[j][i] = best_new_color
                # Update flat colors array
                # Need index of (i,j)
                # idx = i*n - i*(i+1)//2 + j - i - 1 ?
                # Let's just reconstruct or keep a mapping.
                # For speed, let's just keep color_matrix and convert at the end.
                current_cost = best_cost
                
                if current_cost == 0:
                    # Convert back to flat list
                    res_colors = [0] * num_edges
                    idx = 0
                    for u in range(n):
                        for v in range(u+1, n):
                            res_colors[idx] = color_matrix[u][v]
                            idx += 1
                    return res_colors
        
        return None

    # Try n from 12 upwards
    # We allocate a fraction of time to each n
    # If we find a solution for n, we save it and try n+1
    # If time is running out, we return the best n found
    
    # To be safe, let's try 12, 13, 14.
    # If 12 is hard, we might spend too much time.
    # Let's give each n a time slice.
    
    n_target = 12
    while n_target <= 20:
        if time.time() > deadline - 10:
            break
            
        # Try to solve for n_target
        # Use a time limit for this specific n
        time_limit_for_n = min(20.0, deadline - time.time() - 5)
        if time_limit_for_n < 1.0:
            break
            
        # Multiple restarts?
        # Let's just run local_search which has its own time limit.
        # But local_search as defined above runs until max_time or solution.
        # We should pass the time limit.
        
        # Actually, the local_search function above takes max_time.
        # Let's call it.
        
        # We might want to try a few restarts if it fails quickly?
        # No, local_search already runs for max_time.
        
        sol = local_search(n_target, max_time=time_limit_for_n)
        
        if sol is not None:
            # Verify
            if calculate_cost(n_target, sol) == 0:
                best_n = n_target
                best_colors = sol
                # Try next n
                n_target += 1
            else:
                # Should not happen
                n_target += 1
        else:
            # Failed to find solution in time
            # Should we try a smaller n? No, we started small.
            # If 12 failed, 13 will likely fail too.
            # But maybe 12 was just unlucky?
            # Let's break and return best so far.
            # Or try one more restart for current n?
            # Given the time budget, breaking is safer.
            break
            
    return {'n': best_n, 'colors': best_colors}
