import time
import random
import numpy as np
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    best_n = 17
    best_edges = None
    
    # Seed: Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_edges = edges_17
    best_adj = np.zeros((17, 17), dtype=bool)
    for u, v in edges_17:
        best_adj[u, v] = True
        best_adj[v, u] = True

    def build_masks(n, adj):
        """Build bitmask neighborhoods from numpy adj matrix."""
        masks = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    masks[u] |= (1 << v)
        return masks

    def has_k4(n, masks):
        """Check if any K4 exists. Early exit."""
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
                            return True
        return False

    def has_ind6(n, masks):
        """Check if any independent 6-set exists. Early exit."""
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        
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
                                        return True
        return False

    def is_valid(n, adj):
        """Check validity with early exit."""
        masks = build_masks(n, adj)
        if has_k4(n, masks):
            return False
        if has_ind6(n, masks):
            return False
        return True

    def count_k4_fast(n, masks):
        """Count K4s using bitmask operations."""
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

    def count_ind6_fast(n, masks):
        """Count independent 6-sets."""
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

    def cost(n, adj, masks):
        """Returns (k4_count, ind6_count)."""
        k4 = count_k4_fast(n, masks)
        ind6 = count_ind6_fast(n, masks)
        return k4 + ind6

    def local_search(n, adj, masks, time_budget):
        """Local search with incremental mask updates."""
        end_time = time.time() + time_budget
        current_cost = cost(n, adj, masks)
        T = 2.0
        no_improve = 0
        max_no_improve = 1000
        
        while time.time() < end_time and no_improve < max_no_improve:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            old_state = adj[u, v]
            adj[u, v] = not old_state
            adj[v, u] = not old_state
            
            # Update masks
            if old_state:
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
            else:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
            
            new_cost = cost(n, adj, masks)
            
            if new_cost < current_cost:
                current_cost = new_cost
                no_improve = 0
                if new_cost == 0:
                    return adj, masks, True
            else:
                delta = new_cost - current_cost
                T = max(0.1, T * 0.999)
                if random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                    no_improve += 1
                else:
                    # Revert
                    adj[u, v] = old_state
                    adj[v, u] = old_state
                    if old_state:
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                    else:
                        masks[u] &= ~(1 << v)
                        masks[v] &= ~(1 << u)
                    no_improve += 1
        
        return adj, masks, False

    def try_find_graph(n, time_budget, seed_adj=None):
        """Local search to find valid graph on n vertices."""
        end_time = time.time() + time_budget
        
        best_cost = float('inf')
        best_adj = None
        best_masks = None
        
        num_restarts = 5
        restart_budget = time_budget / num_restarts
        
        for restart in range(num_restarts):
            if time.time() >= end_time:
                break
            
            if seed_adj is not None and restart == 0:
                adj = seed_adj.copy()
            else:
                adj = np.zeros((n, n), dtype=bool)
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i, j] = True
                            adj[j, i] = True
            
            masks = build_masks(n, adj)
            adj, masks, found = local_search(n, adj, masks, restart_budget)
            
            final_cost = cost(n, adj, masks)
            if final_cost < best_cost:
                best_cost = final_cost
                best_adj = adj
                best_masks = masks
                
            if final_cost == 0:
                edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                return edges
        
        if best_adj is not None and best_cost == 0:
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if best_adj[i, j]]
            return edges
            
        return None

    # Try to find graphs from n=18 upward
    current_adj = best_adj
    current_n = best_n
    
    for n in range(18, 36):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        
        # Growth strategy: Use the previous best graph as a seed for the first restart
        # We need to extend the adjacency matrix
        seed_adj = None
        if current_n > 0:
            seed_adj = np.zeros((n, n), dtype=bool)
            seed_adj[:current_n, :current_n] = current_adj
            
            # Connect new vertices (current_n .. n-1) to existing ones
            # Simple heuristic: Random connection with p=0.5
            for i in range(current_n, n):
                for j in range(i):
                    if random.random() < 0.5:
                        seed_adj[i, j] = True
                        seed_adj[j, i] = True
                # Connect new vertices to each other
                for k in range(i + 1, n):
                    if random.random() < 0.5:
                        seed_adj[i, k] = True
                        seed_adj[k, i] = True

        # Allocate time
        if n <= 20:
            budget = min(remaining * 0.3, 5.0)
        elif n <= 25:
            budget = min(remaining * 0.2, 3.0)
        else:
            budget = min(remaining * 0.1, 2.0)
        
        edges = try_find_graph(n, budget, seed_adj)
        if edges is not None:
            best_n = n
            best_edges = edges
            # Update current_adj for next growth step
            current_adj = np.zeros((n, n), dtype=bool)
            for u, v in edges:
                current_adj[u, v] = True
                current_adj[v, u] = True
            current_n = n
            if n >= 35:
                break

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
