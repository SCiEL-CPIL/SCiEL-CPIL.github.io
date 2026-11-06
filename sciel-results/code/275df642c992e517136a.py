import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 320.0
    
    q = 37
    residues = set((x * x) % q for x in range(1, q))
    n0 = q
    adj0 = [0] * n0
    edges0 = []
    for i in range(n0):
        for j in range(i + 1, n0):
            if (j - i) % q in residues:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
                edges0.append((i, j))
                
    def get_forbidden_masks(adj, n):
        k4_masks = []
        ind4_masks = []
        for comb in combinations(range(n), 4):
            u, v, w, x = comb
            is_c4 = ((adj[u] >> v) & 1) and ((adj[u] >> w) & 1) and ((adj[u] >> x) & 1) and \
                    ((adj[v] >> w) & 1) and ((adj[v] >> x) & 1) and \
                    ((adj[w] >> x) & 1)
            mask = (1 << u) | (1 << v) | (1 << w) | (1 << x)
            if is_c4:
                k4_masks.append(mask)
            else:
                is_ind4 = not ((adj[u] >> v) & 1) and not ((adj[u] >> w) & 1) and not ((adj[u] >> x) & 1) and \
                          not ((adj[v] >> w) & 1) and not ((adj[v] >> x) & 1) and \
                          not ((adj[w] >> x) & 1)
                if is_ind4:
                    ind4_masks.append(mask)
        return k4_masks, ind4_masks

    def init_state():
        k4_masks, ind4_masks = get_forbidden_masks(adj0, n0)
        return adj0[:], n0, edges0[:], k4_masks, ind4_masks

    def is_valid_mask(M, n, k4_masks, ind4_masks):
        for mask in k4_masks:
            if (M & mask) == mask:
                return False
        full_mask = (1 << n) - 1
        not_M = full_mask ^ M
        for mask in ind4_masks:
            if (not_M & mask) == mask:
                return False
        return True

    def add_vertex(state, M):
        adj, n, edges, k4_masks, ind4_masks = state
        new_adj = adj[:] + [0]
        new_n = n + 1
        new_edges = edges[:]
        
        new_adj[n] = M
        for i in range(n):
            if (M >> i) & 1:
                new_adj[i] |= (1 << n)
                new_edges.append((i, n))
        
        # Update forbidden masks
        new_k4_masks = k4_masks[:]
        neighbors = [i for i in range(n) if (M >> i) & 1]
        for tri in combinations(neighbors, 3):
            a, b, c = tri
            if ((new_adj[a] >> b) & 1) and ((new_adj[a] >> c) & 1) and ((new_adj[b] >> c) & 1):
                new_k4_masks.append((1 << a) | (1 << b) | (1 << c) | (1 << n))
                
        new_ind4_masks = ind4_masks[:]
        non_neighbors = [i for i in range(n) if not ((M >> i) & 1)]
        for tri in combinations(non_neighbors, 3):
            a, b, c = tri
            if not ((new_adj[a] >> b) & 1) and not ((new_adj[a] >> c) & 1) and not ((new_adj[b] >> c) & 1):
                new_ind4_masks.append((1 << a) | (1 << b) | (1 << c) | (1 << n))
                
        return (new_adj, new_n, new_edges, new_k4_masks, new_ind4_masks)

    best_state = init_state()
    best_n = best_state[1]
    
    while time.time() < deadline and best_n < 46:
        current_state = best_state
        added = False
        # Try to find a valid mask
        # Use a heuristic: Try masks that are "balanced"
        for _ in range(5000):
            M = 0
            for i in range(current_state[1]):
                if random.random() < 0.5:
                    M |= (1 << i)
            
            if is_valid_mask(M, current_state[1], current_state[3], current_state[4]):
                new_state = add_vertex(current_state, M)
                best_state = new_state
                best_n = new_state[1]
                added = True
                break
        
        if not added:
            # Stuck. To avoid infinite loop on stuck state, we can't easily backtrack.
            # But since we are maximizing, and 43 is known, we might just stop or try slightly different randomness.
            # Given the time limit, if we are stuck at 38, we return 38.
            # But we should try to restart the random seed? 
            # Actually, if we are stuck, it means no single vertex addition is possible? 
            # That's unlikely for these graphs. 
            # If we are stuck, we break.
            break

    return best_n, best_state[2]
