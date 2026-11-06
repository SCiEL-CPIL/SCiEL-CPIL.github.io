import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def is_k23_free(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free using bitmasks."""
        cu = (adj[u] & adj[v]).bit_count()
        if cu > 2:
            return False
        
        # For each w adjacent to v (but not u): adding (u,v) makes u a common neighbor of w and v
        # So we need common(w,v) <= 1 for all w in N(v) \ {u}
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[v] >> w) & 1:
                common_uw = (adj[u] & adj[w]).bit_count()
                if common_uw > 1:
                    return False
            if (adj[u] >> w) & 1:
                common_vw = (adj[v] & adj[w]).bit_count()
                if common_vw > 1:
                    return False
        return True

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            for v in range(u + 1, n):
                if (mask >> v) & 1:
                    edges.append([u, v])
        return edges

    def greedy_build(seed=None, degree_balanced=False):
        """Greedy random construction with optional degree-balancing."""
        rng = random.Random(seed)
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        
        if degree_balanced:
            # Process edges in order of current degree sum (prefer low-degree pairs)
            for _ in range(10):  # Multiple passes with re-sorting
                remaining = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
                remaining.sort(key=lambda e: adj[e[0]].bit_count() + adj[e[1]].bit_count())
                for (u, v) in remaining:
                    if ((adj[u] >> v) & 1):
                        continue
                    if is_k23_free(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        else:
            for (u, v) in all_edges:
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def structured_initial(seed=None, circulant=True):
        """Create a structured initial graph using circulant base + greedy fill."""
        rng = random.Random(seed)
        adj = [0] * n
        
        if circulant:
            # Start with a circulant: each vertex connects to ±1, ±2, +5 mod 20
            for i in range(n):
                for d in [1, 2, 5]:
                    j = (i + d) % n
                    if j != i and not ((adj[i] >> j) & 1):
                        if is_k23_free(adj, i, j):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
        
        # Fill remaining edges greedily in random order
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        for (u, v) in all_edges:
            if not ((adj[u] >> v) & 1):
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        return adj

    def local_search(adj, rng, time_budget=5.0):
        """Local search with edge additions, 1-swaps, and 2-swaps."""
        adj = list(adj)
        search_start = time.time()
        
        # Build edge sets
        present = set()
        missing = set()
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    present.add((u, v))
                else:
                    missing.add((u, v))
        
        improved = True
        while improved:
            if time.time() - search_start > time_budget:
                break
            improved = False
            
            # Phase 1: Try adding any missing edge
            for (u, v) in list(missing):
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    present.add((u, v))
                    missing.discard((u, v))
                    improved = True
                    break
            if improved:
                continue
            
            # Phase 2: Try 1-swap (remove one present edge, add one missing edge)
            if not present or not missing:
                break
            
            present_list = list(present)
            missing_list = list(missing)
            
            found_swap = False
            for _ in range(500):
                if time.time() - search_start > time_budget:
                    break
                a, b = rng.choice(present_list)
                c, d = rng.choice(missing_list)
                
                # Remove (a,b)
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                
                # Try adding (c,d)
                if is_k23_free(adj, c, d):
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    present.discard((a, b))
                    present.add((c, d))
                    missing.add((a, b))
                    missing.discard((c, d))
                    improved = True
                    found_swap = True
                    break
                else:
                    # Restore (a,b)
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
            
            if found_swap:
                continue
            
            # Phase 3: Try 2-swap (remove two present edges, add two missing edges)
            if len(present) < 2 or len(missing) < 2:
                break
            
            present_list = list(present)
            missing_list = list(missing)
            
            found_2swap = False
            for _ in range(200):
                if time.time() - search_start > time_budget:
                    break
                (a, b), (c, d) = rng.sample(present_list, 2)
                (e, f), (g, h) = rng.sample(missing_list, 2)
                
                # Remove (a,b) and (c,d)
                adj[a] &= ~(1 << b); adj[b] &= ~(1 << a)
                adj[c] &= ~(1 << d); adj[d] &= ~(1 << c)
                
                # Try adding (e,f) and (g,h)
                if is_k23_free(adj, e, f):
                    adj[e] |= (1 << f); adj[f] |= (1 << e)
                    if is_k23_free(adj, g, h):
                        adj[g] |= (1 << h); adj[h] |= (1 << g)
                        present.discard((a, b))
                        present.discard((c, d))
                        present.add((e, f))
                        present.add((g, h))
                        missing.add((a, b))
                        missing.add((c, d))
                        missing.discard((e, f))
                        missing.discard((g, h))
                        improved = True
                        found_2swap = True
                        break
                    else:
                        adj[e] &= ~(1 << f); adj[f] &= ~(1 << e)
                
                # Restore (a,b) and (c,d)
                adj[a] |= (1 << b); adj[b] |= (1 << a)
                adj[c] |= (1 << d); adj[d] |= (1 << c)
            
            if not found_2swap:
                break
        
        return adj

    best_adj = None
    best_count = 0

    # Generate diverse initial constructions
    inits = []
    
    # Circulant-based structured
    for s in range(3):
        inits.append(structured_initial(seed=s, circulant=True))
    
    # Random structured
    for s in range(3):
        inits.append(structured_initial(seed=100+s, circulant=False))
    
    # Pure greedy with different seeds
    for s in range(5):
        inits.append(greedy_build(seed=s, degree_balanced=False))
    
    # Degree-balanced greedy
    for s in range(3):
        inits.append(greedy_build(seed=200+s, degree_balanced=True))
    
    # Process each initial with local search
    for i, adj in enumerate(inits):
        if time.time() > deadline:
            break
        
        cnt = edge_count(adj)
        if cnt > best_count:
            best_adj = adj[:]
            best_count = cnt
        
        # Apply local search with time budget
        remaining = deadline - time.time()
        if remaining > 2.0:
            budget = min(5.0, remaining * 0.8)
            rng = random.Random(i * 1000)
            improved_adj = local_search(adj, rng, time_budget=budget)
            cnt2 = edge_count(improved_adj)
            if cnt2 > best_count:
                best_adj = improved_adj
                best_count = cnt2
    
    # Final polish: try more local search iterations on best
    if best_adj is not None and time.time() < deadline - 1.0:
        remaining = deadline - time.time() - 1.0
        if remaining > 1.0:
            rng = random.Random(999)
            polished = local_search(best_adj, rng, time_budget=remaining)
            if edge_count(polished) > best_count:
                best_adj = polished
                best_count = edge_count(best_adj)
    
    return adj_to_edges(best_adj) if best_adj else []
