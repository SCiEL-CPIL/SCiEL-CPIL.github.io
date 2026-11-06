import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Try n=40 first
    for n in [40, 39, 38, 37, 36, 35]:
        if time.time() > deadline:
            break
            
        # Initialize random graph
        # p=0.3 is a good start for triangle-free search
        adj = [0] * n
        # adj[i] is a bitmask of neighbors
        
        # Generate random edges
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.3:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Count triangles
        tri_count = 0
        for i in range(n):
            # Triangles containing i as the smallest index?
            # To avoid double counting, sum over i < j < k
            # Faster: For each i, count common neighbors with j > i
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    # Common neighbors of i and j with index > j?
                    # Or just count all common neighbors and divide by 3?
                    # Common neighbors mask:
                    common = adj[i] & adj[j]
                    # Number of bits set
                    tri_count += bin(common).count('1')
        tri_count //= 3  # Each triangle counted 3 times (once for each vertex as i)
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        best_cost = float('inf')
        best_adj = adj[:]
        
        # Helper: Check if independent set of size 10 exists
        def has_ind_set_10(adj_mask, n):
            # Backtracking
            # available: bitmask of vertices available to be in the set
            # size_needed: 10
            
            # Heuristic: If the number of available vertices is less than needed, return False
            # Also, if the maximum possible independent set in available is < needed, return False.
            # Simple bound: The independence number is at most the number of vertices in the largest independent set of the greedy subgraph?
            # Let's just use the count of vertices as a weak bound first.
            
            def backtrack(available, needed):
                if needed == 0:
                    return True
                if bin(available).count('1') < needed:
                    return False
                
                # Pick a vertex with highest degree in the available subgraph?
                # Or just pick the first one?
                # To prune better, pick a vertex that has many neighbors in available.
                
                # Find vertex v in available with max degree within available
                best_v = -1
                max_deg = -1
                temp_avail = available
                while temp_avail:
                    # Get lowest set bit
                    lsb = temp_avail & -temp_avail
                    v = lsb.bit_length() - 1
                    # Degree of v in available
                    deg = bin(adj_mask[v] & available).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        best_v = v
                    temp_avail ^= lsb
                
                if best_v == -1:
                    return False # No vertices left but needed > 0
                
                v = best_v
                # Branch 1: Include v
                # New available = available - v - neighbors(v)
                new_avail_1 = available & ~(adj_mask[v] | (1 << v))
                if backtrack(new_avail_1, needed - 1):
                    return True
                
                # Branch 2: Exclude v
                new_avail_2 = available & ~(1 << v)
                if backtrack(new_avail_2, needed):
                    return True
                
                return False

            return backtrack((1 << n) - 1, 10)

        # Main SA Loop
        steps = 0
        while time.time() < deadline:
            steps += 1
            
            # Check if current is valid
            if tri_count == 0:
                if not has_ind_set_10(adj, n):
                    # Construct edge list
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
            
            # Pick random edge to flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
                
            # Ensure i < j for consistency? Not strictly needed for mask, but good for triangle update logic
            if i > j:
                i, j = j, i
            
            # Check if edge exists
            is_edge = bool(adj[i] & (1 << j))
            
            # Calculate delta triangles
            # Common neighbors of i and j
            common = adj[i] & adj[j]
            common_count = bin(common).count('1')
            
            if is_edge:
                # Removing edge
                # Triangles decrease by common_count
                delta_tri = -common_count
            else:
                # Adding edge
                # Triangles increase by common_count
                delta_tri = common_count
            
            new_tri_count = tri_count + delta_tri
            
            # Cost function
            # We want to minimize triangles.
            # If triangles > 0, cost = triangles * 100
            # If triangles == 0, we check alpha. 
            # For SA acceptance, we need a cost.
            # If we just minimize triangles, we might get stuck in a triangle-free state with alpha >= 10.
            # So, if tri_count == 0, we should penalize based on alpha?
            # But calculating alpha is expensive.
            # Strategy: 
            # 1. If tri_count > 0, Cost = tri_count * 100.
            # 2. If tri_count == 0, Cost = 0 (optimistic). 
            #    But we need to escape if alpha >= 10.
            #    So, if tri_count == 0, we do a local check.
            #    If alpha >= 10, we want to increase triangles slightly to allow restructuring?
            #    Or just keep searching.
            
            # Let's use a simple cost:
            # If tri_count > 0: cost = tri_count
            # If tri_count == 0: cost = 0
            
            old_cost = tri_count
            new_cost = new_tri_count
            
            delta_cost = new_cost - old_cost
            
            # Acceptance
            accept = False
            if delta_cost < 0:
                accept = True
            else:
                if T > 0.01:
                    # Metropolis
                    if random.random() < pow(2.718, -delta_cost / T):
                        accept = True
            
            if accept:
                # Flip edge
                if is_edge:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                
                tri_count = new_tri_count
                
                # Track best
                if tri_count < best_cost:
                    best_cost = tri_count
                    best_adj = adj[:]
            
            T *= cooling
            if T < 0.01:
                T = 0.01 # Floor temperature to allow exploration

        # If we didn't find a valid one in this n, continue to next n
        # Or return the best found if it's valid?
        # We only return if valid.
        
    # Fallback: Return seed
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    return {'n': n_seed, 'edges': edges_seed}
