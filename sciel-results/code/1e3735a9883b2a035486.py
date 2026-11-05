import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_vc(n, edges):
        if not edges:
            return []
        
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # 1. Greedy Solution for Upper Bound
        S_greedy = set()
        rem_mask = (1 << n) - 1
        while True:
            edge_exists = False
            for i in range(n):
                if (rem_mask >> i) & 1 and (adj[i] & rem_mask):
                    edge_exists = True
                    break
            if not edge_exists:
                break
            max_deg = -1
            best_v = -1
            for i in range(n):
                if (rem_mask >> i) & 1:
                    deg = bin(adj[i] & rem_mask).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        best_v = i
            S_greedy.add(best_v)
            rem_mask &= ~(1 << best_v)
            
        # Prune Greedy
        changed = True
        while changed:
            changed = False
            mask_S = 0
            for v in S_greedy:
                mask_S |= (1 << v)
            for u in list(S_greedy):
                if (adj[u] & ~mask_S) == 0:
                    S_greedy.remove(u)
                    changed = True
                    mask_S = 0
                    for v in S_greedy:
                        mask_S |= (1 << v)
                    break # Restart loop after change
        
        best_size = len(S_greedy)
        best_set = S_greedy.copy()

        # 2. Branch and Bound
        # Use recursion with bitmasks
        # State: current_set (list), current_size, rem_mask
        # To avoid copying sets, we can just track the set at the end.
        
        def dfs(current_set, current_size, rem_mask):
            nonlocal best_size, best_set
            if time.time() > deadline:
                return
            if current_size >= best_size:
                return
            
            # Find an edge in rem_mask
            u = -1
            v = -1
            for i in range(n):
                if (rem_mask >> i) & 1:
                    neighbors = adj[i] & rem_mask
                    if neighbors:
                        lsb = neighbors & -neighbors
                        j = lsb.bit_length() - 1
                        u = i
                        v = j
                        break
                if u != -1:
                    break
            
            if u == -1:
                # No edges, cover complete
                if current_size < best_size:
                    best_size = current_size
                    best_set = set(current_set)
                return

            # Branch 1: Include u
            new_set = current_set.copy()
            new_set.add(u)
            dfs(new_set, current_size + 1, rem_mask & ~(1 << u))
            
            # Branch 2: Include v
            new_set = current_set.copy()
            new_set.add(v)
            dfs(new_set, current_size + 1, rem_mask & ~(1 << v))

        # Start B&B
        dfs(set(), 0, (1 << n) - 1)
        
        return sorted(best_set)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        sol = solve_vc(n, edges)
        out.append(sol)
        
        if time.time() > deadline:
            break

    return out
