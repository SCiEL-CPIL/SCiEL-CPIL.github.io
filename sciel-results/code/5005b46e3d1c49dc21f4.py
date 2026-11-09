import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def build_masks(adj):
        n = adj.shape[0]
        masks = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    masks[i] |= (1 << j)
        return masks

    def has_k4(masks, n):
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

    def has_ind6(adj, n):
        comp_masks = [0] * n
        full = (1 << n) - 1
        for i in range(n):
            for j in range(n):
                if not adj[i, j] and i != j:
                    comp_masks[i] |= (1 << j)
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
                        if not (comp_masks[wi] & (1 << xi)):
                            continue
                        common2 = common & comp_masks[wi] & comp_masks[xi]
                        common2 &= ~((1 << (xi + 1)) - 1)
                        y_mask = common2
                        while y_mask:
                            lsb_y = y_mask & -y_mask
                            yi = lsb_y.bit_length() - 1
                            y_mask ^= lsb_y
                            if not (comp_masks[wi] & (1 << yi)) or not (comp_masks[xi] & (1 << yi)):
                                continue
                            common3 = common2 & comp_masks[yi]
                            common3 &= ~((1 << (yi + 1)) - 1)
                            z_mask = common3
                            while z_mask:
                                lsb_z = z_mask & -z_mask
                                zi = lsb_z.bit_length() - 1
                                z_mask ^= lsb_z
                                if (comp_masks[wi] & (1 << zi)) and (comp_masks[xi] & (1 << zi)) and (comp_masks[yi] & (1 << zi)):
                                    return True
        return False

    def is_valid(adj):
        n = adj.shape[0]
        masks = build_masks(adj)
        if has_k4(masks, n):
            return False
        if has_ind6(adj, n):
            return False
        return True

    def paley_graph(p):
        if p % 4 != 1:
            return None
        adj = np.zeros((p, p), dtype=bool)
        squares = set()
        for x in range(1, p):
            squares.add(x * x % p)
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj

    def local_search(adj, time_budget):
        n = adj.shape[0]
        end_time = time.time() + time_budget
        masks = build_masks(adj)
        
        # Initial check
        if not has_k4(masks, n) and not has_ind6(adj, n):
            return adj, True
        
        no_improve = 0
        max_no_improve = 500
        T = 5.0
        
        while time.time() < end_time and no_improve < max_no_improve:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            old = adj[u, v]
            adj[u, v] = not old
            adj[v, u] = not old
            
            # Update masks
            if old:
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
            else:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
            
            # Check validity
            if not has_k4(masks, n) and not has_ind6(adj, n):
                return adj, True
            
            # If still invalid, decide whether to keep or revert
            T = max(0.5, T * 0.99)
            if random.random() < 0.3:
                no_improve += 1
            else:
                adj[u, v] = old
                adj[v, u] = old
                if old:
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                else:
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                no_improve += 1
        
        # Final check
        if not has_k4(masks, n) and not has_ind6(adj, n):
            return adj, True
        return adj, False

    # --- Main ---
    best_n = 17
    best_edges = None

    # Start with Paley(17)
    p17 = paley_graph(17)
    if p17 is not None:
        best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if p17[i, j]]

    # Try to construct 35-vertex graph using known structure
    if time_left() > 10.0:
        # Known construction: two Paley(17) + universal vertex + bipartite
        adj35 = np.zeros((35, 35), dtype=bool)
        adj35[0:17, 0:17] = p17
        adj35[18:35, 18:35] = p17
        
        # Universal vertex 17
        adj35[17, :] = True
        adj35[:, 17] = True
        adj35[17, 17] = False
        
        # Bipartite connection: i in [0,17) connects to 18+j if (i+j)%17 is non-square
        squares17 = set()
        for x in range(1, 17):
            squares17.add(x * x % 17)
        
        for i in range(17):
            for j in range(17):
                if (i + j) % 17 not in squares17:
                    adj35[i, 18 + j] = True
                    adj35[18 + j, i] = True
        
        if is_valid(adj35):
            best_n = 35
            best_edges = [[i, j] for i in range(35) for j in range(i + 1, 35) if adj35[i, j]]
            return {'n': best_n, 'edges': best_edges}
        
        # If standard construction doesn't work, try local search from it
        adj35, found = local_search(adj35, min(time_left() * 0.4, 40.0))
        if found:
            best_n = 35
            best_edges = [[i, j] for i in range(35) for j in range(i + 1, 35) if adj35[i, j]]
            return {'n': best_n, 'edges': best_edges}
        
        # Try more variations
        for variant in range(10):
            if time_left() < 5.0:
                break
            
            adj35v = np.zeros((35, 35), dtype=bool)
            adj35v[0:17, 0:17] = p17
            adj35v[18:35, 18:35] = p17
            adj35v[17, :] = True
            adj35v[:, 17] = True
            adj35v[17, 17] = False
            
            # Random bipartite
            for i in range(17):
                for j in range(17):
                    if random.random() < 0.5:
                        adj35v[i, 18 + j] = True
                        adj35v[18 + j, i] = True
            
            adj35v, found = local_search(adj35v, min(time_left() * 0.3, 20.0))
            if found:
                best_n = 35
                best_edges = [[i, j] for i in range(35) for j in range(i + 1, 35) if adj35v[i, j]]
                return {'n': best_n, 'edges': best_edges}

    # Try smaller n if 35 failed
    for target_n in [30, 25, 20]:
        if time_left() < 5.0:
            break
        
        for trial in range(10):
            if time_left() < 3.0:
                break
            
            adj = np.zeros((target_n, target_n), dtype=bool)
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.4:
                        adj[i, j] = True
                        adj[j, i] = True
            
            adj, found = local_search(adj, min(time_left() * 0.2, 15.0))
            if found:
                best_n = target_n
                best_edges = [[i, j] for i in range(target_n) for j in range(i + 1, target_n) if adj[i, j]]
                break
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
