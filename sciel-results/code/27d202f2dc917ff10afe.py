import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    # --- Fast bitmask-based K4 and independent-6-set detection ---
    def build_masks(n, adj):
        masks = [0] * n
        for u in range(n):
            m = 0
            for v in range(n):
                if adj[u, v]:
                    m |= (1 << v)
            masks[u] = m
        return masks

    def has_k4(n, masks):
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
        masks = build_masks(n, adj)
        if has_k4(n, masks):
            return False
        if has_ind6(n, masks):
            return False
        return True

    def violation_count(n, adj):
        masks = build_masks(n, adj)
        # Count K4s
        k4 = 0
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
                            k4 += 1
        # Count independent 6-sets (K6 in complement)
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        ind6 = 0
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
                                        ind6 += 1
        return k4 + ind6

    # --- Paley 17 seed ---
    def paley17():
        n = 17
        squares = {x * x % n for x in range(1, n)}
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in squares:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj

    # --- Construct 35-vertex graph from two Paley 17s ---
    def construct_35():
        n = 35
        adj = np.zeros((n, n), dtype=bool)
        # First Paley 17: vertices 0-16
        p1 = paley17()
        for i in range(17):
            for j in range(17):
                adj[i, j] = p1[i, j]
        # Second Paley 17: vertices 17-33
        p2 = paley17()
        for i in range(17):
            for j in range(17):
                adj[17 + i, 17 + j] = p2[i, j]
        # Bipartite connection: vertex i in first part connects to vertex j in second part
        # if (i + j) % 17 is in squares (Paley-like bipartite)
        squares = {x * x % 17 for x in range(1, 17)}
        for i in range(17):
            for j in range(17):
                if (i + j) % 17 in squares:
                    adj[i, 17 + j] = True
                    adj[17 + j, i] = True
        # Vertex 34: connect to some vertices in both parts
        # Connect to all vertices 0-16 (full connection to first part)
        for i in range(17):
            adj[34, i] = True
            adj[i, 34] = True
        # Connect to vertices in second part where (j) % 17 in squares
        for j in range(17):
            if j in squares:
                adj[34, 17 + j] = True
                adj[17 + j, 34] = True
        return adj

    # --- Local search with incremental updates ---
    def local_search(n, adj, time_budget):
        end_time = time.time() + time_budget
        adj = adj.copy()
        current_viol = violation_count(n, adj)
        if current_viol == 0:
            return adj
        
        T = 3.0
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < end_time and no_improve < max_no_improve:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            old_state = adj[u, v]
            adj[u, v] = not old_state
            adj[v, u] = not old_state
            
            new_viol = violation_count(n, adj)
            
            if new_viol < current_viol:
                current_viol = new_viol
                no_improve = 0
                if new_viol == 0:
                    return adj
            else:
                delta = new_viol - current_viol
                T = max(0.3, T * 0.9995)
                if random.random() < np.exp(-delta / T):
                    current_viol = new_viol
                    no_improve += 1
                else:
                    adj[u, v] = old_state
                    adj[v, u] = old_state
                    no_improve += 1
        
        return adj

    # --- Multi-restart search for a given n ---
    def try_find(n, time_budget):
        end_time = time.time() + time_budget
        best_adj = None
        best_viol = float('inf')
        
        num_restarts = 8
        restart_budget = time_budget / num_restarts
        
        for restart in range(num_restarts):
            if time.time() >= end_time:
                break
            
            # Different initializations
            if restart == 0:
                # Use algebraic seed if n == 35
                if n == 35:
                    adj = construct_35()
                else:
                    # Paley-like initialization
                    adj = np.zeros((n, n), dtype=bool)
                    # Use quadratic residues for edges
                    # Find a prime p >= n
                    p = n
                    while True:
                        is_prime = True
                        for d in range(2, int(p**0.5) + 1):
                            if p % d == 0:
                                is_prime = False
                                break
                        if is_prime:
                            break
                        p += 1
                    squares = {x * x % p for x in range(1, p)}
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (j - i) % p in squares:
                                adj[i, j] = True
                                adj[j, i] = True
            elif restart == 1:
                # Random
                adj = np.zeros((n, n), dtype=bool)
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i, j] = True
                            adj[j, i] = True
            else:
                # Structured random with bias
                adj = np.zeros((n, n), dtype=bool)
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.45:
                            adj[i, j] = True
                            adj[j, i] = True
            
            current_viol = violation_count(n, adj)
            if current_viol < best_viol:
                best_viol = current_viol
                best_adj = adj.copy()
            if current_viol == 0:
                return adj
            
            # Local search
            adj = local_search(n, adj, restart_budget)
            final_viol = violation_count(n, adj)
            if final_viol < best_viol:
                best_viol = final_viol
                best_adj = adj.copy()
            if final_viol == 0:
                return adj
        
        if best_adj is not None and best_viol == 0:
            return best_adj
        return None

    # --- Main search: try to find valid graphs from n=18 upward ---
    best_n = 17
    best_edges = None
    
    # Start with Paley 17
    p17 = paley17()
    best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if p17[i, j]]
    
    # Try to find larger graphs
    for n in range(18, 36):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 3.0:
            break
        
        # Allocate time
        if n <= 20:
            budget = min(remaining * 0.3, 6.0)
        elif n <= 25:
            budget = min(remaining * 0.2, 5.0)
        elif n <= 30:
            budget = min(remaining * 0.15, 4.0)
        elif n <= 34:
            budget = min(remaining * 0.1, 3.0)
        else:
            budget = min(remaining * 0.1, 3.0)
        
        adj = try_find(n, budget)
        if adj is not None:
            best_n = n
            best_edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            if n >= 35:
                break
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
