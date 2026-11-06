import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def greedy_independent_set_size(n, adj):
        candidates = set(range(n))
        ind_set = 0
        while candidates:
            # Pick a vertex with minimum degree in the candidate subgraph
            # to potentially allow more vertices later?
            # Actually, for a lower bound (to prove alpha > k), 
            # we just need any independent set.
            # A greedy approach picking min-degree vertices often yields larger sets.
            min_deg = float('inf')
            v = -1
            for u in candidates:
                deg = sum(1 for w in candidates if w in adj[u])
                if deg < min_deg:
                    min_deg = deg
                    v = u
            if v == -1: break
            ind_set += 1
            candidates.discard(v)
            candidates -= adj[v]
        return ind_set

    def max_independent_set_size_exact(n, adj):
        # Only called if greedy <= 10
        # Use branch and bound
        best = 0
        
        # Precompute adjacency masks
        adj_mask = [0] * n
        for u in range(n):
            mask = 0
            for v in adj[u]:
                mask |= (1 << v)
            adj_mask[u] = mask

        # Sort vertices by degree descending for better pruning?
        # The candidates list is dynamic.
        
        def solve(candidates, current_size):
            nonlocal best
            if current_size + len(candidates) <= best:
                return
            if not candidates:
                best = max(best, current_size)
                return
            
            # Heuristic: pick vertex with highest degree in candidates
            # This tends to reduce the candidate set faster when including it
            v = max(candidates, key=lambda u: sum(1 for w in candidates if w in adj[u]))
            
            # Option 1: Include v
            # New candidates: candidates \ {v} \ N(v)
            new_candidates = []
            v_mask = 1 << v
            for u in candidates:
                if u != v and not (adj_mask[v] & (1 << u)):
                    new_candidates.append(u)
            solve(new_candidates, current_size + 1)
            
            # Option 2: Exclude v
            solve([u for u in candidates if u != v], current_size)

        candidates = list(range(n))
        solve(candidates, 0)
        return best

    def try_build(n):
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_pairs)
        
        edges = set()
        adj = [set() for _ in range(n)]
        
        for u, v in all_pairs:
            common = adj[u].intersection(adj[v])
            if len(common) < 2:
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                
        # Check alpha
        g_alpha = greedy_independent_set_size(n, adj)
        if g_alpha > 10:
            return None
            
        if g_alpha <= 10:
             # It's possible g_alpha <= 10 but exact alpha > 10? 
             # No, greedy gives a lower bound. If lower bound > 10, alpha > 10.
             # If lower bound <= 10, alpha could be > 10.
             # So we must check exact.
             exact_alpha = max_independent_set_size_exact(n, adj)
             if exact_alpha <= 10:
                 return list(edges)
        return None

    # Try n from 39 down to 30
    for n in range(39, 29, -1):
        if time.time() > deadline:
            break
        
        for attempt in range(10):
            if time.time() > deadline:
                break
            edges = try_build(n)
            if edges:
                edge_list = [[u, v] for u, v in edges]
                return {'n': n, 'edges': edge_list}
                
    # Fallback
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
