import random
import time

def construct():
    n = 20
    start_time = time.time()
    max_time = 110.0  # Safety margin before 120s limit

    best_edges = []
    best_count = 0

    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edges_set = set(all_edges)

    def is_valid_add(u, v, adj_mask, common):
        if (adj_mask[u] >> v) & 1:
            return False
        if common[u][v] > 2:
            return False

        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != u:
                if common[u][w] >= 2:
                    return False
            m -= lsb

        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != v:
                if common[v][w] >= 2:
                    return False
            m -= lsb

        return True

    def add_edge(u, v, adj_mask, common, edges):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append([u, v])

        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != u:
                common[u][w] += 1
                common[w][u] += 1
            m -= lsb

        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != v:
                common[v][w] += 1
                common[w][v] += 1
            m -= lsb

    def remove_edge(u, v, adj_mask, common, edges):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

        # Remove from edges list
        # Since n=20, linear search is fine
        for i, e in enumerate(edges):
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                edges.pop(i)
                break

        # Update common neighbors
        # When removing (u, v):
        # common[u][w] decreases by 1 for all w in N(v) (old)
        # But we already cleared the bit in adj_mask.
        # Old N(v) = Current N(v) U {u}
        # The condition for decrementing common[u][w] is that w was a common neighbor of u and v?
        # No.
        # common[u][w] = |N(u) \cap N(w)|.
        # Before removal, v \in N(u).
        # If w \in N(v) (old), then v \in N(w).
        # So v \in N(u) \cap N(w).
        # So common[u][w] should decrease by 1 for all w \in N(v) (old).
        # N(v) (old) = Current N(v) U {u}.
        # But common[u][u] is not used/tracked? We track common[u][v] for u!=v.
        # If w=u, common[u][u] is irrelevant.
        # So we iterate over Current N(v).
        # Wait, is that correct?
        # Let's check w \in Current N(v).
        # Then w \in N(v) (old).
        # Since w \in N(v), v \in N(w).
        # Since u \in N(u) (trivially? No, simple graph, no self loops).
        # u \in N(u) is false.
        # But v \in N(u) was true.
        # So v \in N(u) \cap N(w).
        # So yes, common[u][w] decreases by 1.

        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            m -= lsb

        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            m -= lsb

    def fill_edges(adj_mask, common, edges):
        """
        Systematically try to add all valid missing edges.
        """
        changed = True
        while changed:
            changed = False
            for u, v in all_edges:
                if (adj_mask[u] >> v) & 1:
                    continue
                if is_valid_add(u, v, adj_mask, common):
                    add_edge(u, v, adj_mask, common, edges)
                    changed = True
        return edges

    def run_local_search(initial_edges, seed):
        random.seed(seed)
        
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []

        for u, v in initial_edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edges.append([u, v])

        # Compute initial common
        for u in range(n):
            for v in range(u + 1, n):
                mask = adj_mask[u] & adj_mask[v]
                cnt = bin(mask).count('1')
                common[u][v] = cnt
                common[v][u] = cnt

        best_local_edges = list(edges)
        best_local_count = len(edges)

        # Local Search Loop
        # We will try to add edges. If stuck, remove a random one and retry.
        
        # Precompute list of all possible edges for random selection
        # all_edges is already global
        
        # To make it faster, we can try to add edges in a specific order or random order.
        # Random order helps escape local optima?
        # Actually, just picking a random valid edge from the list of all missing edges is good.
        
        max_iterations = 5000
        stuck_count = 0
        
        for it in range(max_iterations):
            if time.time() - start_time > max_time - 5.0:
                break
                
            # Try to add a random valid edge
            # We can iterate through all missing edges and pick a random valid one
            # Or just pick a random edge and check if valid?
            # Picking random edge and checking is O(1) check + O(degree) update.
            # But if we fail, we might waste time.
            # Better: iterate through missing edges, collect valid ones, pick one.
            # This is O(E * (degree)) per iteration.
            # E ~ 190. Degree ~ 10. 1900 ops.
            # 5000 iterations * 1900 = 9.5M ops. Fast.
            
            valid_edges = []
            for u, v in all_edges:
                if not ((adj_mask[u] >> v) & 1):
                    if is_valid_add(u, v, adj_mask, common):
                        valid_edges.append((u, v))
                        
            if valid_edges:
                u, v = random.choice(valid_edges)
                add_edge(u, v, adj_mask, common, edges)
                stuck_count = 0
                if len(edges) > best_local_count:
                    best_local_count = len(edges)
                    best_local_edges = list(edges)
            else:
                # No edge can be added. Remove a random edge to escape local optimum.
                if not edges:
                    break
                idx = random.randint(0, len(edges) - 1)
                u, v = edges[idx]
                remove_edge(u, v, adj_mask, common, edges)
                stuck_count += 1
                
                # If we've been stuck for a long time, maybe restart?
                # Or just continue.
                if stuck_count > 100:
                    # Try a more aggressive removal?
                    # Or just break and let outer loop restart.
                    # For now, just continue.
                    pass

        # Final fill pass
        if time.time() - start_time < max_time - 3.0:
            fill_edges(adj_mask, common, edges)
            if len(edges) > best_local_count:
                best_local_count = len(edges)
                best_local_edges = list(edges)

        return best_local_edges, best_local_edges

    # Main loop with restarts
    seed = 42
    while time.time() - start_time < max_time:
        # Random initialization: sparse random graph
        # Or empty?
        # Let's start with a random subset of edges to provide a better starting point.
        # Greedy construction is good.
        
        # Greedy initialization
        init_edges = []
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        
        # Random order of edges
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if is_valid_add(u, v, adj_mask, common):
                add_edge(u, v, adj_mask, common, init_edges)

        # Local search from this state
        # We need to pass the state to run_local_search or replicate the logic.
        # My run_local_search takes initial_edges and rebuilds state.
        # This is fine for n=20.
        
        local_best_edges, _ = run_local_search(init_edges, seed)
        
        if len(local_best_edges) > best_count:
            best_count = len(local_best_edges)
            best_edges = local_best_edges
            
        seed += 1

    return best_edges

if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(f"Found {len(edges)} edges")
