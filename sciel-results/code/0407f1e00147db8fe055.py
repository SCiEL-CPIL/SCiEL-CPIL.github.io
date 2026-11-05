import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 35s margin
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges_total = len(all_edges)
    
    # Precompute all pairs for faster iteration
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Precompute: for each pair (i,j), list of k > j
    triple_indices = []
    for i in range(n):
        for j in range(i+1, n):
            triple_indices.append((i, j, list(range(j+1, n))))
    
    def count_edges(masks):
        return sum(bin(m).count('1') for m in masks) // 2
    
    def get_violation(masks):
        """Find one K3,3 violation or return None. Optimized."""
        for i, j, k_list in triple_indices:
            common_ij = masks[i] & masks[j]
            if bin(common_ij).count('1') < 3:
                continue
            for k in k_list:
                if bin(common_ij & masks[k]).count('1') >= 3:
                    return (i, j, k, common_ij & masks[k])
        return None
    
    def is_valid(masks):
        return get_violation(masks) is None
    
    def try_add_edge(masks, u, v):
        """Try to add edge (u,v), return True if valid."""
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
        if is_valid(masks):
            return True
        else:
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
            return False
    
    def try_remove_edge(masks, u, v):
        """Remove edge (u,v)."""
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
    
    def local_search_add_edges(masks):
        """Try to add all possible missing edges in random order. Returns new edge count."""
        current = count_edges(masks)
        # Get all missing edges
        missing = [(i, j) for i, j in pairs if not (masks[i] & (1 << j))]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 3:
                break
            if try_add_edge(masks, u, v):
                current += 1
        return current
    
    def simulated_annealing(masks, current_edges, T_start=3.0, T_end=0.05, max_steps=80000):
        """Simulated annealing to improve the graph."""
        best_local = current_edges
        best_local_masks = masks[:]
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / max(1, max_steps))
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if T < T_end:
                break
            
            # Choose move type
            r = random.random()
            
            if r < 0.4:
                # Add a random missing edge
                missing = [(i, j) for i, j in pairs if not (masks[i] & (1 << j))]
                if not missing:
                    break
                u, v = random.choice(missing)
                old_u, old_v = masks[u], masks[v]
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                
                if is_valid(masks):
                    new_edges = count_edges(masks)
                    delta = new_edges - current_edges
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        current_edges = new_edges
                        if current_edges > best_local:
                            best_local = current_edges
                            best_local_masks = masks[:]
                    else:
                        masks[u] = old_u
                        masks[v] = old_v
                else:
                    masks[u] = old_u
                    masks[v] = old_v
                    
            elif r < 0.7:
                # Remove a random existing edge
                existing = [(i, j) for i, j in pairs if (masks[i] & (1 << j))]
                if not existing:
                    break
                u, v = random.choice(existing)
                old_u, old_v = masks[u], masks[v]
                try_remove_edge(masks, u, v)
                
                new_edges = count_edges(masks)
                delta = new_edges - current_edges
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    current_edges = new_edges
                    if current_edges > best_local:
                        best_local = current_edges
                        best_local_masks = masks[:]
                else:
                    masks[u] = old_u
                    masks[v] = old_v
                    
            else:
                # 2-opt swap: remove one edge, add another
                existing = [(i, j) for i, j in pairs if (masks[i] & (1 << j))]
                missing = [(i, j) for i, j in pairs if not (masks[i] & (1 << j))]
                if not existing or not missing:
                    break
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                    continue
                    
                old_u1, old_v1, old_u2, old_v2 = masks[u1], masks[v1], masks[u2], masks[v2]
                try_remove_edge(masks, u1, v1)
                masks[u2] |= (1 << v2)
                masks[v2] |= (1 << u2)
                
                if is_valid(masks):
                    new_edges = count_edges(masks)
                    delta = new_edges - current_edges
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        current_edges = new_edges
                        if current_edges > best_local:
                            best_local = current_edges
                            best_local_masks = masks[:]
                    else:
                        masks[u1] = old_u1
                        masks[v1] = old_v1
                        masks[u2] = old_u2
                        masks[v2] = old_v2
                else:
                    masks[u1] = old_u1
                    masks[v1] = old_v1
                    masks[u2] = old_u2
                    masks[v2] = old_v2
            
            T *= cooling
        
        return best_local, best_local_masks
    
    best_edges = 0
    best_masks = [0] * n
    
    # Multiple restarts with different initial densities
    initial_densities = [0.85, 0.80, 0.75, 0.90, 0.70]
    num_restarts = 15
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        p = initial_densities[restart % len(initial_densities)]
        
        # Initialize with random graph
        masks = [0] * n
        for i, j in pairs:
            if random.random() < p:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
        
        # Repair: remove edges until valid
        max_repair = 300
        for _ in range(max_repair):
            if time.time() > deadline - 10:
                break
            viol = get_violation(masks)
            if viol is None:
                break
            i, j, k, common_mask = viol
            # Get 3 common neighbors
            neighbors = []
            m = common_mask
            while len(neighbors) < 3 and m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                neighbors.append(idx)
                m ^= lsb
            
            # Remove a random edge from the K3,3
            u = random.choice([i, j, k])
            v = random.choice(neighbors)
            try_remove_edge(masks, u, v)
        
        if not is_valid(masks):
            continue
        
        current_edges = count_edges(masks)
        
        # First, do local search to add edges
        current_edges = local_search_add_edges(masks)
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_masks = masks[:]
        
        # Then simulated annealing
        sa_edges, sa_masks = simulated_annealing(masks, current_edges)
        
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_masks = sa_masks[:]
        
        # Final local search on the SA result
        final_edges = local_search_add_edges(best_masks)
        if final_edges > best_edges:
            best_edges = final_edges
    
    # Convert best_masks to edge list
    edges = []
    for i in range(n):
        m = best_masks[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            m ^= lsb
            
    return edges
