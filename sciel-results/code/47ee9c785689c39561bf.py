import time
import random

def construct():
    n = 15
    start = time.time()
    deadline = start + 85  # Safety margin for 90s limit
    
    def get_degs(adj):
        return [adj[i].bit_count() for i in range(n)]

    def is_valid(adj, n):
        """Full validity check for initial state or final verification."""
        deg = get_degs(adj)
        for u in range(n):
            neighbors_u = adj[u]
            v_mask = neighbors_u & ~((1 << (u + 1)) - 1)
            while v_mask:
                v_bit = v_mask & (-v_mask)
                v = v_bit.bit_length() - 1
                v_mask ^= v_bit
                s_mask = neighbors_u & ~v_bit
                while s_mask:
                    s_bit = s_mask & (-s_mask)
                    s = s_bit.bit_length() - 1
                    s_mask ^= s_bit
                    t_mask = adj[v] & ~(1 << u) & ~s_bit
                    while t_mask:
                        t_bit = t_mask & (-t_mask)
                        t = t_bit.bit_length() - 1
                        t_mask ^= t_bit
                        if deg[s] == deg[t]:
                            return False
        return True

    def can_add_edge(adj, u, v, degs):
        """
        Check if adding edge (u, v) maintains validity.
        Assumes adj is currently valid.
        """
        s_mask = adj[u] & ~(1 << v)
        while s_mask:
            s_bit = s_mask & (-s_mask)
            s = s_bit.bit_length() - 1
            s_mask ^= s_bit
            
            target_deg = degs[s]
            
            t_mask = adj[v] & ~(1 << u) & ~s_bit
            while t_mask:
                t_bit = t_mask & (-t_mask)
                t = t_bit.bit_length() - 1
                t_mask ^= t_bit
                if degs[t] == target_deg:
                    return False
        return True

    def try_add_edges(adj, degs, deadline):
        """Greedy: try to add all missing edges that maintain validity."""
        added = True
        while added:
            if time.time() > deadline:
                break
            added = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if not (adj[i] >> j) & 1:
                        # Try adding edge (i, j)
                        if can_add_edge(adj, i, j, degs):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            degs[i] += 1
                            degs[j] += 1
                            added = True
        return adj, degs

    def edges_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    best_adj = None
    best_degs = None
    best_count = -1

    # Try all bipartite splits
    for split in range(2, n - 1):
        if time.time() > deadline - 5:
            break
        adj = [0] * n
        for i in range(split):
            for j in range(split, n):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        degs = get_degs(adj)
        
        adj, degs = try_add_edges(adj, degs, deadline)
        cnt = edges_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj)
            best_degs = list(degs)

    # Local search: try removing edges and re-greedy-filling
    if time.time() < deadline - 10:
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            if attempt == 0 or random.random() < 0.5:
                adj = list(best_adj)
                degs = list(best_degs)
            else:
                split = random.randint(2, n - 2)
                adj = [0] * n
                for i in range(split):
                    for j in range(split, n):
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                degs = get_degs(adj)
            
            # Perturbation: remove 1-3 random edges
            num_remove = random.randint(1, 3)
            for _ in range(num_remove):
                all_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            all_edges.append((i, j))
                if not all_edges:
                    break
                i, j = random.choice(all_edges)
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
                degs[i] -= 1
                degs[j] -= 1
            
            # Greedy fill
            adj, degs = try_add_edges(adj, degs, deadline)
            cnt = edges_count(adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = list(adj)
                best_degs = list(degs)

    # Final verification
    if best_adj:
        if not is_valid(best_adj, n):
             # Fallback to empty graph if something went wrong (should not happen)
             best_adj = [0]*n
             best_count = 0
        result = adj_to_edges(best_adj)
        return result
    return []
