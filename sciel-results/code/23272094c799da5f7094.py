import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s kill

    def build_masks(n, adj):
        masks = [0] * n
        for u in range(n):
            m = 0
            for v in range(n):
                if adj[u, v]:
                    m |= (1 << v)
            masks[u] = m
        return masks

    def count_k4(n, masks):
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
                common = masks[u] & masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            k4_count += 1
        return k4_count

    def count_ind6(n, masks):
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        
        ind6_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (comp_masks[u] & (1 << v)):
                    continue
                common = comp_masks[u] & comp_masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if comp_masks[wi] & (1 << xi):
                            common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                            common2 &= ~((1 << (xi + 1)) - 1)
                            y_mask = common2
                            while y_mask:
                                lsb_y = y_mask & -y_mask
                                yi = lsb_y.bit_length() - 1
                                y_mask ^= lsb_y
                                z_mask = common2 & ~((1 << (yi + 1)) - 1)
                                while z_mask:
                                    lsb_z = z_mask & -z_mask
                                    zi = lsb_z.bit_length() - 1
                                    z_mask ^= lsb_z
                                    if comp_masks[yi] & (1 << zi):
                                        ind6_count += 1
        return ind6_count

    def total_violations(n, adj):
        masks = build_masks(n, adj)
        return count_k4(n, masks) + count_ind6(n, masks)

    def edges_from_adj(n, adj):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]

    # Clebsch graph edges (16 vertices, K4-free, alpha=5)
    clebsch_edges = [
        (0, 6), (0, 7), (0, 10), (0, 11),
        (1, 5), (1, 8), (1, 10), (1, 13),
        (2, 4), (2, 9), (2, 11), (2, 13),
        (3, 4), (3, 7), (3, 9), (3, 12),
        (5, 6), (5, 8), (5, 12), (5, 15),
        (6, 7), (6, 9), (6, 14), (6, 15),
        (7, 8), (7, 11), (7, 14), (7, 15),
        (8, 9), (8, 12), (8, 15),
        (9, 10), (9, 14),
        (10, 11), (10, 13), (10, 14),
        (11, 12), (11, 14), (11, 15),
        (12, 13), (12, 15),
        (13, 14), (13, 15)
    ]
    
    best_n = 16
    best_edges = list(clebsch_edges)

    # Verify baseline
    adj_16 = np.zeros((16, 16), dtype=bool)
    for u, v in clebsch_edges:
        adj_16[u, v] = True
        adj_16[v, u] = True
    
    masks_16 = build_masks(16, adj_16)
    if count_k4(16, masks_16) > 0 or count_ind6(16, masks_16) > 0:
        best_n = 5
        best_edges = []

    def incremental_build(target_n, edge_prob, seed):
        """Build graph incrementally by adding vertices one at a time."""
        rng = random.Random(seed)
        adj = np.zeros((target_n, target_n), dtype=bool)
        
        for new_v in range(1, target_n):
            # Try all 2^new_v possible edge patterns, pick the one with min violations
            best_pattern = None
            best_cost = float('inf')
            
            # For efficiency, only try a subset of patterns if new_v is large
            num_patterns = 1 << new_v
            if num_patterns > 256:
                # Sample patterns
                num_tries = 256
                for _ in range(num_tries):
                    pattern = rng.getrandbits(new_v)
                    # Apply pattern
                    for i in range(new_v):
                        if (pattern >> i) & 1:
                            adj[i, new_v] = True
                            adj[new_v, i] = True
                    else:
                        pass
                    # Need to undo and try next - this is complex, let's just use random init + local search for large n
                    # Instead, use a simpler approach: random edge assignments with local search
                    break
            else:
                for pattern in range(num_patterns):
                    for i in range(new_v):
                        if (pattern >> i) & 1:
                            adj[i, new_v] = True
                            adj[new_v, i] = True
                        else:
                            adj[i, new_v] = False
                            adj[new_v, i] = False
                    masks = build_masks(new_v + 1, adj)
                    cost = count_k4(new_v + 1, masks) + count_ind6(new_v + 1, masks)
                    if cost < best_cost:
                        best_cost = cost
                        best_pattern = pattern
                    # Undo
                    for i in range(new_v):
                        adj[i, new_v] = False
                        adj[new_v, i] = False
                
                # Apply best pattern
                if best_pattern is not None:
                    for i in range(new_v):
                        if (best_pattern >> i) & 1:
                            adj[i, new_v] = True
                            adj[new_v, i] = True
        
        return adj

    def local_search_2flip(n, adj, budget):
        """Local search with single-edge and double-edge flips."""
        current_cost = total_violations(n, adj)
        T = 1.0
        no_improve = 0
        max_steps = int(budget * 50)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if no_improve > 300:
                break
            
            if random.random() < 0.7:
                # Single edge flip
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                old = adj[u, v]
                adj[u, v] = not old
                adj[v, u] = not old
                new_cost = total_violations(n, adj)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improve = 0
                    if new_cost == 0:
                        return adj, 0
                else:
                    delta = new_cost - current_cost
                    T = max(0.05, T * 0.997)
                    if random.random() < np.exp(-delta / T):
                        current_cost = new_cost
                    else:
                        adj[u, v] = old
                        adj[v, u] = old
                    no_improve += 1
            else:
                # Double edge flip (2-flip)
                u1 = random.randint(0, n - 1)
                v1 = random.randint(0, n - 1)
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u1 == v1 or u2 == v2:
                    continue
                old1 = adj[u1, v1]
                old2 = adj[u2, v2]
                adj[u1, v1] = not old1
                adj[v1, u1] = not old1
                adj[u2, v2] = not old2
                adj[v2, u2] = not old2
                new_cost = total_violations(n, adj)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improve = 0
                    if new_cost == 0:
                        return adj, 0
                else:
                    delta = new_cost - current_cost
                    T = max(0.05, T * 0.997)
                    if random.random() < np.exp(-delta / T):
                        current_cost = new_cost
                    else:
                        adj[u1, v1] = old1
                        adj[v1, u1] = old1
                        adj[u2, v2] = old2
                        adj[v2, u2] = old2
                    no_improve += 1
        
        return adj, current_cost

    # Try to improve with incremental construction + local search
    for target_n in range(17, 36):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Allocate budget per target_n
        if target_n <= 20:
            budget = min(remaining * 0.2, 8.0)
        elif target_n <= 25:
            budget = min(remaining * 0.15, 5.0)
        else:
            budget = min(remaining * 0.1, 4.0)
        
        found = False
        num_restarts = max(2, int(budget / 1.5))
        
        for restart in range(num_restarts):
            if time.time() > deadline or found:
                break
            
            # Diverse initializations
            if target_n <= 20:
                # Use incremental construction
                adj = incremental_build(target_n, 0.5, seed=restart * 1000 + 42)
            else:
                # Random init with varying density
                density = random.choice([0.3, 0.4, 0.5, 0.6])
                adj = np.zeros((target_n, target_n), dtype=bool)
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < density:
                            adj[i, j] = True
                            adj[j, i] = True
            
            current_cost = total_violations(target_n, adj)
            
            if current_cost == 0:
                found = True
                best_n = target_n
                best_edges = edges_from_adj(target_n, adj)
                break
            
            # Local search with 2-flips
            adj, final_cost = local_search_2flip(target_n, adj, budget / num_restarts)
            
            if final_cost == 0:
                found = True
                best_n = target_n
                best_edges = edges_from_adj(target_n, adj)
                break
        
        if found:
            if best_n >= 35:
                break
            # Continue to try larger n
        else:
            # If we couldn't find valid graph for this n, continue to next
            pass
    
    return {'n': best_n, 'edges': best_edges}
