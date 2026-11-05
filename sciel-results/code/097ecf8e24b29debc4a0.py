import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Target n
    n_target = 42
    
    # Helper to generate Paley 37 edges
    def get_paley_edges(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return edges

    paley_edges = get_paley_edges(37)
    
    # Function to count edges in a subgraph defined by a bitmask
    # adj_mask is a list of ints
    def count_edges_in_mask(mask, adj_mask):
        count = 0
        m = mask
        while m:
            # Extract lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            m ^= lsb
            # Count bits in intersection of adj_mask[idx] and remaining m
            count += (adj_mask[idx] & m).bit_count()
        return count

    # Function to calculate total cost (K5 + Ind5) for a graph represented by adj_mask
    # This is used for final verification and initial cost
    def calculate_total_cost(adj_mask, n):
        k5_count = 0
        ind5_count = 0
        # Iterate over all 5-combinations
        # For performance, we can use the property that a K5 is a set of 5 vertices 
        # where each is connected to all others.
        # With bitmasks, for a combination (v1, v2, v3, v4, v5):
        # Check if v5 is in adj[v1] & adj[v2] & adj[v3] & adj[v4]
        # Check if v5 is NOT in adj[v1] | adj[v2] | adj[v3] | adj[v4] for independent
        
        # This is O(C(n,5)) which is ~850k for n=42.
        # 850k iterations in Python might take ~1s.
        
        for comb in itertools.combinations(range(n), 5):
            v1, v2, v3, v4, v5 = comb
            
            # K5 check
            mask = adj_mask[v1] & adj_mask[v2] & adj_mask[v3] & adj_mask[v4]
            if mask & (1 << v5):
                k5_count += 1
                
            # Independent set check
            # v5 must not be connected to any of v1..v4
            # i.e., v5 is not in adj[v1] | adj[v2] | adj[v3] | adj[v4]
            union_mask = adj_mask[v1] | adj_mask[v2] | adj_mask[v3] | adj_mask[v4]
            if not (union_mask & (1 << v5)):
                ind5_count += 1
                
        return k5_count + ind5_count

    # Initialize graph
    # Start with Paley 37
    adj_mask = [0] * n_target
    all_vertices_mask = (1 << n_target) - 1
    
    for u, v in paley_edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Initialize vertices 37-41 randomly
    random.seed(42)
    for i in range(37, n_target):
        for j in range(i + 1, n_target):
            if random.random() < 0.5:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
        for k in range(i):
            if random.random() < 0.5:
                adj_mask[i] |= (1 << k)
                adj_mask[k] |= (1 << i)

    current_cost = calculate_total_cost(adj_mask, n_target)
    best_cost = current_cost
    best_adj_mask = adj_mask[:]
    
    # Simulated Annealing with improved schedule
    T = 5.0
    T_min = 0.01
    steps = 0
    total_steps = 500000  # Will be limited by time anyway
    
    # Use exponential cooling that reaches T_min near the end
    # T = T_min + (T_initial - T_min) * (T_min/T_initial)^(steps/total_steps)
    # Simpler: linear interpolation in log space
    log_T_initial = math.log(T)
    log_T_min = math.log(T_min)
    
    while time.time() < deadline:
        # Pick random edge to flip
        u = random.randint(0, n_target - 1)
        v = random.randint(0, n_target - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        edge_exists = bool(adj_mask[u] & (1 << v))
        
        # Calculate Delta Cost
        # Common neighbors of u and v
        common_neigh = adj_mask[u] & adj_mask[v]
        # Remove u and v from common_neigh (they are not neighbors of themselves)
        common_neigh &= ~(1 << u)
        common_neigh &= ~(1 << v)
        
        # Triangles in common_neigh
        # This is the number of edges in the subgraph induced by common_neigh
        delta_k5 = count_edges_in_mask(common_neigh, adj_mask)
        
        # Common non-neighbors of u and v
        # Nodes w such that w is not neighbor of u AND w is not neighbor of v
        # ~(adj_mask[u] | adj_mask[v]) & all_vertices_mask
        common_non_neigh = ~(adj_mask[u] | adj_mask[v]) & all_vertices_mask
        common_non_neigh &= ~(1 << u)
        common_non_neigh &= ~(1 << v)
        
        # Independent triangles in common_non_neigh
        # This is the number of edges in the COMPLEMENT graph induced by common_non_neigh
        # Or, number of non-edges in G induced by common_non_neigh
        # count_edges_in_mask counts edges in G.
        # We want to count edges in G_bar, which is (total_pairs - edges_in_G)
        # Total pairs in a set of size k is k*(k-1)/2
        k = common_non_neigh.bit_count()
        total_pairs = k * (k - 1) // 2
        edges_in_g = count_edges_in_mask(common_non_neigh, adj_mask)
        edges_in_g_bar = total_pairs - edges_in_g
        
        delta_ind = edges_in_g_bar
        
        if edge_exists:
            # Removing edge
            # We lose K5s containing this edge: -delta_k5
            # We gain Ind5s containing this edge (because it was blocking them): +delta_ind
            delta_cost = -delta_k5 + delta_ind
        else:
            # Adding edge
            # We gain K5s: +delta_k5
            # We lose Ind5s: -delta_ind
            delta_cost = delta_k5 - delta_ind
            
        # Acceptance with log-space temperature
        # Use linear interpolation in log-space for smoother annealing
        progress = min(steps / total_steps, 1.0)
        log_T = log_T_initial + progress * (log_T_min - log_T_initial)
        T = math.exp(log_T)
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            # Apply move
            if edge_exists:
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
            else:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                
            current_cost += delta_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_adj_mask = adj_mask[:]
                
                if best_cost == 0:
                    # Verify!
                    if calculate_total_cost(best_adj_mask, n_target) == 0:
                        edges = []
                        for i in range(n_target):
                            for j in range(i + 1, n_target):
                                if best_adj_mask[i] & (1 << j):
                                    edges.append((i, j))
                        return n_target, edges

        steps += 1
        
        # Periodic full check to correct any drift
        if steps % 5000 == 0:
            current_cost = calculate_total_cost(adj_mask, n_target)
            if current_cost < best_cost:
                best_cost = current_cost
                best_adj_mask = adj_mask[:]
                if best_cost == 0:
                     if calculate_total_cost(best_adj_mask, n_target) == 0:
                        edges = []
                        for i in range(n_target):
                            for j in range(i + 1, n_target):
                                if best_adj_mask[i] & (1 << j):
                                    edges.append((i, j))
                        return n_target, edges

    # Final polish pass: try to find a valid graph from best_adj_mask
    # Use greedy descent on the best solution found
    if best_cost > 0:
        polish_adj = best_adj_mask[:]
        polish_cost = best_cost
        polish_deadline = min(start_time + 295, deadline + 5)
        polish_T = 0.1
        polish_steps = 0
        
        while time.time() < polish_deadline and polish_cost > 0:
            u = random.randint(0, n_target - 1)
            v = random.randint(0, n_target - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            edge_exists = bool(polish_adj[u] & (1 << v))
            
            common_neigh = polish_adj[u] & polish_adj[v]
            common_neigh &= ~(1 << u)
            common_neigh &= ~(1 << v)
            delta_k5 = count_edges_in_mask(common_neigh, polish_adj)
            
            common_non_neigh = ~(polish_adj[u] | polish_adj[v]) & all_vertices_mask
            common_non_neigh &= ~(1 << u)
            common_non_neigh &= ~(1 << v)
            k = common_non_neigh.bit_count()
            total_pairs = k * (k - 1) // 2
            edges_in_g = count_edges_in_mask(common_non_neigh, polish_adj)
            delta_ind = total_pairs - edges_in_g
            
            if edge_exists:
                delta_cost = -delta_k5 + delta_ind
            else:
                delta_cost = delta_k5 - delta_ind
            
            if delta_cost < 0 or (delta_cost == 0 and random.random() < 0.5) or random.random() < math.exp(-delta_cost / polish_T):
                if edge_exists:
                    polish_adj[u] &= ~(1 << v)
                    polish_adj[v] &= ~(1 << u)
                else:
                    polish_adj[u] |= (1 << v)
                    polish_adj[v] |= (1 << u)
                    
                polish_cost += delta_cost
                
                if polish_cost < best_cost:
                    best_cost = polish_cost
                    best_adj_mask = polish_adj[:]
                    
                    if best_cost == 0:
                        if calculate_total_cost(best_adj_mask, n_target) == 0:
                            edges = []
                            for i in range(n_target):
                                for j in range(i + 1, n_target):
                                    if best_adj_mask[i] & (1 << j):
                                        edges.append((i, j))
                            return n_target, edges
                
                polish_T = max(0.001, polish_T * 0.999)
            polish_steps += 1
    
    # If we didn't find a valid 42-graph, check if best_adj_mask is valid (cost 0)
    if best_cost == 0:
        if calculate_total_cost(best_adj_mask, n_target) == 0:
            edges = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if best_adj_mask[i] & (1 << j):
                        edges.append((i, j))
            return n_target, edges

    # Fallback: Return Paley 37
    return 37, paley_edges
