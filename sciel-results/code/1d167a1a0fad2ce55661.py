import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin

    # 1. Generate Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    
    # Build adjacency bitsets for Paley 37
    # adj[i] is an integer with bits set for neighbors of i
    adj = [0] * q
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute all K4 and Ind4 masks for the current graph
    # This will be updated as we add vertices, but for the first step (37->38)
    # we only care about K4s and Ind4s in the existing 37 vertices.
    
    def get_bad_masks(n, adj_list):
        k4_masks = []
        ind4_masks = []
        for combo in combinations(range(n), 4):
            a, b, c, d = combo
            # Check K4
            if (adj_list[a] >> b) & 1 and (adj_list[a] >> c) & 1 and (adj_list[a] >> d) & 1 and \
               (adj_list[b] >> c) & 1 and (adj_list[b] >> d) & 1 and \
               (adj_list[c] >> d) & 1:
                mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                k4_masks.append(mask)
            # Check Ind4
            if not ((adj_list[a] >> b) & 1) and not ((adj_list[a] >> c) & 1) and not ((adj_list[a] >> d) & 1) and \
               not ((adj_list[b] >> c) & 1) and not ((adj_list[b] >> d) & 1) and \
               not ((adj_list[c] >> d) & 1):
                mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                ind4_masks.append(mask)
        return k4_masks, ind4_masks

    # Initial state
    current_n = q
    current_edges = edges
    current_adj = adj
    
    # Try to extend to 38, 39, ...
    best_n = current_n
    best_edges = current_edges
    
    for target_n in range(current_n + 1, 43): # Try up to 42
        if time.time() > deadline - 10:
            break
            
        new_vertex_idx = current_n
        k4_masks, ind4_masks = get_bad_masks(current_n, current_adj)
        
        # Local search for the neighborhood of the new vertex
        # S: set of neighbors of new_vertex_idx
        # T: set of non-neighbors
        # Cost: number of K4s in S + number of Ind4s in T
        
        # Initial guess: random
        S_mask = 0
        for i in range(current_n):
            if random.random() < 0.5:
                S_mask |= (1 << i)
        
        def calculate_cost(S_mask):
            T_mask = ((1 << current_n) - 1) ^ S_mask
            cost = 0
            for mask in k4_masks:
                if (mask & S_mask) == mask:
                    cost += 1
            for mask in ind4_masks:
                if (mask & T_mask) == mask:
                    cost += 1
            return cost

        current_cost = calculate_cost(S_mask)
        best_S_mask = S_mask
        best_cost = current_cost
        
        # Simulated Annealing / Hill Climbing
        # We want to find S_mask with cost 0.
        
        temperature = 1.0
        cooling = 0.99
        min_temp = 0.01
        steps = 0
        
        while temperature > min_temp and time.time() < deadline - 5:
            # Pick a random vertex to flip
            flip_idx = random.randint(0, current_n - 1)
            new_S_mask = S_mask ^ (1 << flip_idx)
            new_cost = calculate_cost(new_S_mask)
            
            if new_cost < current_cost:
                S_mask = new_S_mask
                current_cost = new_cost
                if current_cost == 0:
                    break
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_S_mask = S_mask
            else:
                # Accept with probability exp(-delta/T)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
                    S_mask = new_S_mask
                    current_cost = new_cost
            
            temperature *= cooling
            steps += 1
            if steps % 100 == 0 and current_cost == 0:
                break
        
        if best_cost == 0:
            # Found a valid neighborhood!
            # Add the new vertex and edges
            new_edges = list(current_edges)
            new_adj = list(current_adj)
            new_adj.append(0) # Placeholder for new vertex
            
            for i in range(current_n):
                if (best_S_mask >> i) & 1:
                    new_edges.append((i, new_vertex_idx))
                    new_adj[i] |= (1 << new_vertex_idx)
                    new_adj[new_vertex_idx] |= (1 << i)
            
            current_n = target_n
            current_edges = new_edges
            current_adj = new_adj
            best_n = current_n
            best_edges = current_edges
        else:
            # Failed to find a valid neighborhood for this target
            # Break? Or try different restarts?
            # Given the time, if we fail for 38, we probably won't find 39.
            # But maybe a different random seed would work?
            # Let's try a few more restarts for this target_n before giving up.
            found = False
            for restart in range(5):
                if time.time() > deadline - 5:
                    break
                S_mask = 0
                for i in range(current_n):
                    if random.random() < 0.5:
                        S_mask |= (1 << i)
                
                current_cost = calculate_cost(S_mask)
                best_S_mask_local = S_mask
                best_cost_local = current_cost
                
                temperature = 1.0
                steps = 0
                while temperature > 0.01 and time.time() < deadline - 5:
                    flip_idx = random.randint(0, current_n - 1)
                    new_S_mask = S_mask ^ (1 << flip_idx)
                    new_cost = calculate_cost(new_S_mask)
                    
                    if new_cost < current_cost:
                        S_mask = new_S_mask
                        current_cost = new_cost
                        if current_cost == 0:
                            break
                        if current_cost < best_cost_local:
                            best_cost_local = current_cost
                            best_S_mask_local = S_mask
                    else:
                        delta = new_cost - current_cost
                        if random.random() < pow(2.71828, -delta / temperature):
                            S_mask = new_S_mask
                            current_cost = new_cost
                    
                    temperature *= 0.99
                    steps += 1
                    if steps % 100 == 0 and current_cost == 0:
                        break
                
                if best_cost_local == 0:
                    found = True
                    best_S_mask = best_S_mask_local
                    break
            
            if found:
                new_edges = list(current_edges)
                new_adj = list(current_adj)
                new_adj.append(0)
                
                for i in range(current_n):
                    if (best_S_mask >> i) & 1:
                        new_edges.append((i, new_vertex_idx))
                        new_adj[i] |= (1 << new_vertex_idx)
                        new_adj[new_vertex_idx] |= (1 << i)
                
                current_n = target_n
                current_edges = new_edges
                current_adj = new_adj
                best_n = current_n
                best_edges = current_edges
            else:
                break # If we can't extend to this size, stop.

    return best_n, best_edges
