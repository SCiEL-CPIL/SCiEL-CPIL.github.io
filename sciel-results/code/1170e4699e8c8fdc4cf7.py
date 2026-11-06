import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 295

    def time_left():
        return deadline - time.time()

    # Fallback: Paley Graph P(37)
    def get_paley_37():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return q, edges

    # Helper: compute all K4s and Ind4s in a graph given as adjacency list of bitmasks
    def compute_k4s_ind4s(adj, n):
        k4s = []
        ind4s = []
        # Precompute common neighbors for each pair? No, just iterate.
        # For each 4-subset, check if it's a K4 or Ind4.
        # C(37, 4) ~ 66k. C(42, 4) ~ 111k.
        # Checking each takes O(1) if we use bitmasks cleverly?
        # To check if {a,b,c,d} is K4:
        # adj[a] has b,c,d
        # adj[b] has a,c,d
        # adj[c] has a,b,d
        # adj[d] has a,b,c
        # It's a K4 if:
        # (adj[a] & (1<<b)) and (adj[a] & (1<<c)) and (adj[a] & (1<<d)) and ...
        # Actually, if we fix a,b,c, we just need to check d against a,b,c.
        
        # Let's iterate over triangles to find K4s?
        # Or just brute force 4-subsets. 100k * 12 checks = 1.2M ops. Fast enough.
        
        for a, b, c, d in itertools.combinations(range(n), 4):
            is_k4 = True
            is_ind4 = True
            # Check pairs
            pairs = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            for u, v in pairs:
                if adj[u] & (1 << v):
                    is_ind4 = False
                else:
                    is_k4 = False
                if not is_k4 and not is_ind4:
                    break
            if is_k4:
                mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                k4s.append(mask)
            if is_ind4:
                mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                ind4s.append(mask)
        
        return k4s, ind4s

    # Helper: count violations in a graph
    def count_violations(adj, n):
        # Count K5s and Ind5s
        k5_count = 0
        ind5_count = 0
        # Brute force 5-subsets
        for combo in itertools.combinations(range(n), 5):
            is_k5 = True
            is_ind5 = True
            for i in range(5):
                for j in range(i+1, 5):
                    u = combo[i]
                    v = combo[j]
                    if adj[u] & (1 << v):
                        is_ind5 = False
                    else:
                        is_k5 = False
                    if not is_k5 and not is_ind5:
                        break
                if not is_k5 and not is_ind5:
                    break
            if is_k5:
                k5_count += 1
            if is_ind5:
                ind5_count += 1
        return k5_count + ind5_count

    # Helper: try to add a vertex to the graph
    # Returns (success, new_adj, new_edges)
    def try_add_vertex(adj, n, k4s, ind4s, time_limit):
        # We want to find a mask M (neighborhood of new vertex) such that:
        # 1. No K4 in G is a subset of M (otherwise M+K4 is a K5)
        # 2. No Ind4 in G is a subset of ~M (otherwise ~M+Ind4 is an Ind5)
        
        # Cost function for a mask M:
        # cost(M) = (# K4s subset of M) + (# Ind4s subset of ~M)
        
        # We will use local search on the mask.
        # Start with a random mask.
        # Flip bits to reduce cost.
        
        all_mask = (1 << n) - 1
        
        def cost(mask):
            c = 0
            # Count K4s subset of mask
            for k4 in k4s:
                if k4 & ~mask == 0:
                    c += 1
            # Count Ind4s subset of ~mask
            comp = all_mask & ~mask
            for ind4 in ind4s:
                if ind4 & ~comp == 0:
                    c += 1
            return c

        best_mask = 0
        best_cost = float('inf')
        
        # Try a few random initializations
        for _ in range(5):
            if time_left() < time_limit:
                break
            current_mask = 0
            for i in range(n):
                if random.random() < 0.5:
                    current_mask |= (1 << i)
            
            current_cost = cost(current_mask)
            if current_cost < best_cost:
                best_cost = current_cost
                best_mask = current_mask
            
            # Local search
            improved = True
            while improved and time_left() > time_limit:
                improved = False
                # Try flipping each bit
                for i in range(n):
                    new_mask = current_mask ^ (1 << i)
                    new_cost = cost(new_mask)
                    if new_cost < current_cost:
                        current_mask = new_mask
                        current_cost = new_cost
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_mask = current_mask
                        if current_cost == 0:
                            return True, best_mask
                # If no single bit flip improves, try a random flip (simulated annealing style)
                # Or just break? Let's try a random walk if stuck.
                if not improved:
                    # Random restart of local search?
                    # Or just accept a random move?
                    # Let's just break for now to save time, as cost function is expensive.
                    # Actually, let's try a few random flips.
                    for _ in range(10):
                        i = random.randint(0, n - 1)
                        new_mask = current_mask ^ (1 << i)
                        new_cost = cost(new_mask)
                        if new_cost < current_cost:
                            current_mask = new_mask
                            current_cost = new_cost
                            improved = True
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_mask = current_mask
                            if current_cost == 0:
                                return True, best_mask

        if best_cost == 0:
            return True, best_mask
        else:
            return False, best_mask

    # Main logic: Start with P(37) and try to extend
    def try_extend_paley():
        n, edges = get_paley_37()
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        k4s, ind4s = compute_k4s_ind4s(adj, n)
        
        # Try to add vertices one by one
        for _ in range(10): # Max 10 more vertices
            if time_left() < 10:
                break
            
            success, best_mask = try_add_vertex(adj, n, k4s, ind4s, 5.0)
            
            if success:
                # Add the vertex
                new_vertex = n
                adj.append(0)
                for i in range(n):
                    if best_mask & (1 << i):
                        adj[i] |= (1 << new_vertex)
                        adj[new_vertex] |= (1 << i)
                        edges.append((i, new_vertex))
                n = new_vertex + 1
                # Recompute K4s and Ind4s
                # This is expensive, but we only do it when we add a vertex.
                k4s, ind4s = compute_k4s_ind4s(adj, n)
            else:
                break
        
        return n, edges

    # Try the extension
    try:
        n, edges = try_extend_paley()
        # Verify the result
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Quick check for K5/Ind5
        # If it fails, return P(37)
        # We trust the local search, but let's be safe.
        # Actually, the local search ensures cost 0, which means no new K5/Ind5.
        # But we should verify the whole graph just in case of bugs.
        
        # For speed, we only verify if n > 37
        if n > 37:
            if count_violations(adj, n) > 0:
                return get_paley_37()
                
        return n, edges
    except Exception:
        pass
    
    # Fallback
    return get_paley_37()
