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

    def build_masks(n, adj):
        """Build bitmask neighborhoods from numpy adj matrix."""
        masks = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    masks[u] |= (1 << v)
        return masks

    def count_k4_fast(n, masks):
        """Count K4s using bitmask operations. O(n^3) with early termination."""
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
                common = masks[u] & masks[v]
                # Only consider w > v to avoid duplicates
                common &= ~((1 << (v + 1)) - 1)
                # For each pair (w, x) in common with w < x, check if edge w-x exists
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    # x must be > w and in common
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            k4_count += 1
        return k4_count

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

    def count_ind6_fast(n, masks):
        """Count independent 6-sets = count K6 in complement graph."""
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
                            # Triangle in complement: u,v,w,x. Need 2 more common neighbors
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

    def cost(n, adj):
        """Returns (k4_count, ind6_count)."""
        masks = build_masks(n, adj)
        k4 = count_k4_fast(n, masks)
        ind6 = count_ind6_fast(n, masks)
        return k4 + ind6

    def is_valid(n, adj):
        """Check validity with early exit."""
        masks = build_masks(n, adj)
        if has_k4(n, masks):
            return False
        if has_ind6(n, masks):
            return False
        return True

    def try_find_graph(n, time_budget):
        """Local search to find valid graph on n vertices."""
        end_time = time.time() + time_budget
        
        # Multiple random restarts
        best_cost = float('inf')
        best_adj = None
        
        num_restarts = 10
        restart_budget = time_budget / num_restarts
        
        for restart in range(num_restarts):
            if time.time() >= end_time:
                break
            
            # Random initial graph
            adj = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i, j] = True
                        adj[j, i] = True
            
            current_cost = cost(n, adj)
            if current_cost == 0:
                edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                return edges
            
            # Simulated annealing / local search
            T = 2.0
            no_improve = 0
            max_no_improve = 500
            
            while time.time() < end_time and no_improve < max_no_improve:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                old_state = adj[u, v]
                adj[u, v] = not old_state
                adj[v, u] = not old_state
                
                new_cost = cost(n, adj)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improve = 0
                    if new_cost == 0:
                        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                        return edges
                else:
                    delta = new_cost - current_cost
                    T = max(0.1, T * 0.999)
                    if random.random() < np.exp(-delta / T):
                        current_cost = new_cost
                        no_improve += 1
                    else:
                        adj[u, v] = old_state
                        adj[v, u] = old_state
                        no_improve += 1
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_adj = adj.copy()
        
        # Return best found if valid
        if best_adj is not None and is_valid(n, best_adj):
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if best_adj[i, j]]
            return edges
        
        return None

    # Try to find graphs from n=18 upward
    for n in range(18, 36):
        if time.time() > deadline:
            break
        
        # Allocate time based on n
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        
        # Give more time to smaller n (easier), less to larger n
        if n <= 20:
            budget = min(remaining * 0.3, 5.0)
        elif n <= 25:
            budget = min(remaining * 0.2, 3.0)
        else:
            budget = min(remaining * 0.1, 2.0)
        
        edges = try_find_graph(n, budget)
        if edges is not None:
            best_n = n
            best_edges = edges
            if n >= 35:
                break

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
