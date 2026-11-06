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
    def count_edges_in_mask(mask, adj_mask):
        count = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            m ^= lsb
            count += (adj_mask[idx] & m).bit_count()
        return count

    # Function to calculate total cost (K5 + Ind5) for a graph represented by adj_mask
    def calculate_total_cost(adj_mask, n):
        k5_count = 0
        ind5_count = 0
        for comb in itertools.combinations(range(n), 5):
            v1, v2, v3, v4, v5 = comb
            mask = adj_mask[v1] & adj_mask[v2] & adj_mask[v3] & adj_mask[v4]
            if mask & (1 << v5):
                k5_count += 1
            union_mask = adj_mask[v1] | adj_mask[v2] | adj_mask[v3] | adj_mask[v4]
            if not (union_mask & (1 << v5)):
                ind5_count += 1
        return k5_count + ind5_count

    # Helper to initialize the graph with Paley 37 + structured extension
    def initialize_graph(seed_val, density=0.5):
        adj = [0] * n_target
        all_mask = (1 << n_target) - 1
        
        for u, v in paley_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        rng = random.Random(seed_val)
        # Extend with random edges between new vertices and old, and among new
        for i in range(37, n_target):
            for j in range(i + 1, n_target):
                if rng.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            for k in range(i):
                if rng.random() < density:
                    adj[i] |= (1 << k)
                    adj[k] |= (1 << i)
        
        return adj

    # Simulated annealing function
    def simulated_annealing(adj_mask, time_limit, T_initial=10.0, T_min=0.01):
        current_cost = calculate_total_cost(adj_mask, n_target)
        best_cost = current_cost
        best_adj = adj_mask[:]
        all_vertices_mask = (1 << n_target) - 1
        
        steps = 0
        total_steps = 300000
        log_T_initial = math.log(T_initial)
        log_T_min = math.log(T_min)
        
        while time.time() < time_limit:
            # Pick random edge to flip
            u = random.randint(0, n_target - 1)
            v = random.randint(0, n_target - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            edge_exists = bool(adj_mask[u] & (1 << v))
            
            # Calculate Delta Cost
            common_neigh = adj_mask[u] & adj_mask[v]
            common_neigh &= ~(1 << u)
            common_neigh &= ~(1 << v)
            delta_k5 = count_edges_in_mask(common_neigh, adj_mask)
            
            common_non_neigh = ~(adj_mask[u] | adj_mask[v]) & all_vertices_mask
            common_non_neigh &= ~(1 << u)
            common_non_neigh &= ~(1 << v)
            k = common_non_neigh.bit_count()
            total_pairs = k * (k - 1) // 2
            edges_in_g = count_edges_in_mask(common_non_neigh, adj_mask)
            delta_ind = total_pairs - edges_in_g
            
            if edge_exists:
                delta_cost = -delta_k5 + delta_ind
            else:
                delta_cost = delta_k5 - delta_ind
            
            # Temperature schedule - geometric cooling
            progress = min(steps / total_steps, 1.0)
            log_T = log_T_initial + progress * (log_T_min - log_T_initial)
            T = math.exp(log_T)
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                if edge_exists:
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                else:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj_mask[:]
                    
                    if best_cost == 0:
                        if calculate_total_cost(best_adj, n_target) == 0:
                            return best_adj, 0

            steps += 1
            
            # Periodic full check to correct drift
            if steps % 2000 == 0:
                current_cost = calculate_total_cost(adj_mask, n_target)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj_mask[:]
                    if best_cost == 0:
                        if calculate_total_cost(best_adj, n_target) == 0:
                            return best_adj, 0

        return best_adj, best_cost

    # Targeted local search: specifically try to eliminate violations
    def targeted_local_search(adj_mask, time_limit):
        current_cost = calculate_total_cost(adj_mask, n_target)
        best_cost = current_cost
        best_adj = adj_mask[:]
        all_vertices_mask = (1 << n_target) - 1
        
        if best_cost == 0:
            return best_adj, 0
        
        # Find all violating 5-sets
        def get_violations(adj):
            violations = []
            for comb in itertools.combinations(range(n_target), 5):
                v1, v2, v3, v4, v5 = comb
                mask = adj[v1] & adj[v2] & adj[v3] & adj[v4]
                if mask & (1 << v5):
                    violations.append(('k5', comb))
                union_mask = adj[v1] | adj[v2] | adj[v3] | adj[v4]
                if not (union_mask & (1 << v5)):
                    violations.append(('ind', comb))
            return violations
        
        violations = get_violations(adj_mask)
        
        while time.time() < time_limit and best_cost > 0:
            if not violations:
                break
            
            # Pick a random violation and try to fix it by flipping edges
            vtype, comb = random.choice(violations)
            v1, v2, v3, v4, v5 = comb
            
            # Try flipping edges within this 5-set
            candidate_edges = []
            for a, b in itertools.combinations(range(5), 2):
                va, vb = comb[a], comb[b]
                candidate_edges.append((va, vb))
            
            random.shuffle(candidate_edges)
            
            found_improvement = False
            for va, vb in candidate_edges:
                if time.time() >= time_limit:
                    break
                    
                edge_exists = bool(adj_mask[va] & (1 << vb))
                
                common_neigh = adj_mask[va] & adj_mask[vb]
                common_neigh &= ~(1 << va)
                common_neigh &= ~(1 << vb)
                delta_k5 = count_edges_in_mask(common_neigh, adj_mask)
                
                common_non_neigh = ~(adj_mask[va] | adj_mask[vb]) & all_vertices_mask
                common_non_neigh &= ~(1 << va)
                common_non_neigh &= ~(1 << vb)
                k = common_non_neigh.bit_count()
                total_pairs = k * (k - 1) // 2
                edges_in_g = count_edges_in_mask(common_non_neigh, adj_mask)
                delta_ind = total_pairs - edges_in_g
                
                if edge_exists:
                    delta_cost = -delta_k5 + delta_ind
                else:
                    delta_cost = delta_k5 - delta_ind
                
                if delta_cost < 0:
                    if edge_exists:
                        adj_mask[va] &= ~(1 << vb)
                        adj_mask[vb] &= ~(1 << va)
                    else:
                        adj_mask[va] |= (1 << vb)
                        adj_mask[vb] |= (1 << va)
                    
                    current_cost += delta_cost
                    found_improvement = True
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_adj = adj_mask[:]
                        if best_cost == 0:
                            if calculate_total_cost(best_adj, n_target) == 0:
                                return best_adj, 0
                    
                    # Recalculate violations
                    violations = get_violations(adj_mask)
                    break
            
            if not found_improvement:
                # Try random moves if no targeted improvement found
                u = random.randint(0, n_target - 1)
                v = random.randint(0, n_target - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                edge_exists = bool(adj_mask[u] & (1 << v))
                
                common_neigh = adj_mask[u] & adj_mask[v]
                common_neigh &= ~(1 << u)
                common_neigh &= ~(1 << v)
                delta_k5 = count_edges_in_mask(common_neigh, adj_mask)
                
                common_non_neigh = ~(adj_mask[u] | adj_mask[v]) & all_vertices_mask
                common_non_neigh &= ~(1 << u)
                common_non_neigh &= ~(1 << v)
                k = common_non_neigh.bit_count()
                total_pairs = k * (k - 1) // 2
                edges_in_g = count_edges_in_mask(common_non_neigh, adj_mask)
                delta_ind = total_pairs - edges_in_g
                
                if edge_exists:
                    delta_cost = -delta_k5 + delta_ind
                else:
                    delta_cost = delta_k5 - delta_ind
                
                # Accept with small probability to escape local minima
                if delta_cost < 0 or random.random() < 0.01:
                    if edge_exists:
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                    else:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                    
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_adj = adj_mask[:]
                        if best_cost == 0:
                            if calculate_total_cost(best_adj, n_target) == 0:
                                return best_adj, 0
                    
                    violations = get_violations(adj_mask)
        
        return best_adj, best_cost

    # Try multiple random initializations with different seeds and densities
    best_final_adj = None
    best_final_cost = float('inf')
    
    # Allocate time for multiple attempts
    num_attempts = 6
    time_per_attempt = (deadline - start_time) / num_attempts
    
    densities = [0.3, 0.4, 0.5, 0.5, 0.6, 0.7]
    
    for attempt in range(num_attempts):
        if time.time() >= deadline:
            break
        
        seed_val = 42 + attempt * 1000
        density = densities[attempt % len(densities)]
        adj = initialize_graph(seed_val, density)
        
        # Give more time to early attempts, less to later ones
        attempt_deadline = start_time + (attempt + 1) * time_per_attempt
        attempt_deadline = min(attempt_deadline, deadline)
        
        # Phase 1: Simulated annealing
        adj, cost = simulated_annealing(adj, attempt_deadline)
        
        # Phase 2: Targeted local search if we have time
        if time.time() < deadline - 5:
            target_deadline = min(attempt_deadline + 10, deadline - 2)
            adj, cost = targeted_local_search(adj, target_deadline)
        
        if cost < best_final_cost:
            best_final_cost = cost
            best_final_adj = adj[:]
            
            if cost == 0:
                if calculate_total_cost(best_final_adj, n_target) == 0:
                    edges = []
                    for i in range(n_target):
                        for j in range(i + 1, n_target):
                            if best_final_adj[i] & (1 << j):
                                edges.append((i, j))
                    return n_target, edges

    # Final polish on the best solution found
    if best_final_adj is not None and best_final_cost > 0 and time.time() < deadline - 10:
        polish_deadline = deadline - 2
        polish_adj = best_final_adj[:]
        polish_cost = best_final_cost
        all_vertices_mask = (1 << n_target) - 1
        polish_T = 0.5
        
        # Final aggressive local search
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
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / polish_T):
                if edge_exists:
                    polish_adj[u] &= ~(1 << v)
                    polish_adj[v] &= ~(1 << u)
                else:
                    polish_adj[u] |= (1 << v)
                    polish_adj[v] |= (1 << u)
                    
                polish_cost += delta_cost
                
                if polish_cost < best_final_cost:
                    best_final_cost = polish_cost
                    best_final_adj = polish_adj[:]
                    
                    if best_final_cost == 0:
                        if calculate_total_cost(best_final_adj, n_target) == 0:
                            edges = []
                            for i in range(n_target):
                                for j in range(i + 1, n_target):
                                    if best_final_adj[i] & (1 << j):
                                        edges.append((i, j))
                            return n_target, edges
                
                polish_T = max(0.001, polish_T * 0.999)

    # If we found a valid 42-graph, return it
    if best_final_cost == 0 and best_final_adj is not None:
        if calculate_total_cost(best_final_adj, n_target) == 0:
            edges = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if best_final_adj[i] & (1 << j):
                        edges.append((i, j))
            return n_target, edges

    # Fallback: Return Paley 37
    return 37, paley_edges
