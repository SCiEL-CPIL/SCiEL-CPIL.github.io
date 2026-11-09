import time
import random
import math
from collections import defaultdict

def _compute_book_sizes(edges, n=40):
    """Compute the largest book size for the given edges."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    max_book = 0
    for u, v in edges:
        common = (adj[u] & adj[v]).bit_count()
        if common > max_book:
            max_book = common
    return max_book

def _greedy_construct(n=40, target_edges=401, seed=42):
    """
    Greedy construction: add edges one at a time, choosing the edge
    that minimizes the maximum book size increase.
    """
    random.seed(seed)
    adj = [0] * n
    edge_set = set()
    edges = []
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Track current max book size
    current_max = 0
    
    for _ in range(target_edges):
        best_edge = None
        best_max_after = float('inf')
        
        # Sample a subset of candidate edges for speed
        # Try all non-edges but with early termination
        candidates = []
        for e in all_edges:
            if e not in edge_set:
                candidates.append(e)
        
        if not candidates:
            break
            
        # For large candidate sets, sample
        if len(candidates) > 200:
            candidates = random.sample(candidates, 200)
        
        for u, v in candidates:
            # Compute book size if we add (u,v)
            # Book size = |N(u) ∩ N(v)|
            common = (adj[u] & adj[v]).bit_count()
            new_max = max(current_max, common)
            
            if new_max < best_max_after:
                best_max_after = new_max
                best_edge = (u, v)
                # Early termination if we can't do better
                if best_max_after == current_max:
                    # Check if we can still maintain current_max
                    # Actually we want minimum, so if we found one that doesn't increase, keep looking for safety
                    pass
        
        if best_edge is None:
            best_edge = candidates[0]
        
        u, v = best_edge
        edge_set.add((u, v))
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.append([u, v])
        current_max = max(current_max, (adj[u] & adj[v]).bit_count())
    
    return edges

def _build_structural_init(a_size, b_size, seed):
    """
    Build a graph with a bipartite core A-B and a bipartite subgraph in B.
    This ensures the graph on B is triangle-free, so intra-B edges have book size = a_size.
    A-B edges have book size = degree of B-vertex in the B-subgraph.
    """
    random.seed(seed)
    n = 40
    target_edges = 401
    
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    edges = []
    edge_set = set()
    
    # 1. Add all A-B edges
    for a in A:
        for b in B:
            edges.append([a, b])
            edge_set.add((a, b))
    
    current_count = len(edges)
    needed_in_B = target_edges - current_count
    
    if needed_in_B < 0:
        # This configuration is impossible with full bipartite A-B
        # Reduce A-B edges? For now, just return invalid or handle differently.
        # 9*31 = 279. 401-279=122. OK.
        # 10*30 = 300. 401-300=101. OK.
        # 8*32 = 256. 401-256=145. OK.
        pass

    # 2. Split B into B1 and B2 to make B-subgraph bipartite
    mid = len(B) // 2
    B1 = B[:mid]
    B2 = B[mid:]
    
    # Potential edges between B1 and B2
    potential = []
    for u in B1:
        for v in B2:
            potential.append((u, v))
    
    random.shuffle(potential)
    
    # Degrees in B-subgraph
    deg_B = {v: 0 for v in B}
    
    added = 0
    # Greedy add edges, preferring those that keep degrees balanced
    # We just shuffle and add, then we could balance, but random shuffle is often decent for balance
    # Better: sort by current degrees? 
    # Let's just add randomly first, then check max degree.
    # If max degree is too high, we can try to swap edges.
    
    for u, v in potential:
        if added == needed_in_B:
            break
        edges.append([u, v])
        edge_set.add((u, v))
        deg_B[u] += 1
        deg_B[v] += 1
        added += 1
        
    if added < needed_in_B:
        # Not enough potential edges? (Should not happen for these sizes)
        # Fill with any remaining B-B edges (might create triangles, bad)
        # But for 9-31, B has 31 vertices. B1=15, B2=16. 15*16=240 potential. Need 122. OK.
        pass

    return edges

def _simulate_annealing(n=40, target_edges=401, time_limit=10.0, seed=42, initial_edges=None):
    """Simulated annealing with efficient bitmask updates and precomputed neighbors."""
    start_time = time.time()
    random.seed(seed)
    
    if initial_edges is None:
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        initial_edges = [list(e) for e in all_possible[:target_edges]]
        
    current_edges = [list(e) for e in initial_edges]
    edge_set = set((min(u,v), max(u,v)) for u,v in current_edges)
    
    # Build adjacency masks
    adj = [0] * n
    for u, v in current_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Compute initial book sizes
    book_sizes = {}
    current_max = 0
    for u, v in current_edges:
        key = (min(u,v), max(u,v))
        bs = (adj[u] & adj[v]).bit_count()
        book_sizes[key] = bs
        if bs > current_max:
            current_max = bs
            
    best_edges = [list(e) for e in current_edges]
    best_max = current_max
    
    # Precompute all possible edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # SA parameters
    T0 = 5.0
    Tend = 0.05
    alpha = 0.9995
    
    T = T0
    while T > Tend and time.time() - start_time < time_limit:
        # Pick a random edge to remove
        remove_idx = random.randint(0, len(current_edges) - 1)
        ru, rv = current_edges[remove_idx]
        rkey = (min(ru, rv), max(ru, rv))
        
        # Pick a random non-edge to add
        ckey = None
        for _ in range(10):
            cu, cv = random.choice(all_possible)
            ckey = (min(cu, cv), max(cu, cv))
            if ckey not in edge_set:
                break
        if ckey is None or ckey in edge_set:
            continue
            
        cu, cv = ckey
        
        # Identify affected edges efficiently using adjacency masks
        affected = set()
        affected.add(rkey)
        affected.add(ckey)
        
        # Get neighbors from masks
        def get_neighbors(mask):
            neighbors = []
            m = mask
            while m:
                lsb = m & -m
                idx = (lsb.bit_length() - 1)
                neighbors.append(idx)
                m ^= lsb
            return neighbors
        
        # Affected edges are those incident to ru, rv, cu, cv
        for v in get_neighbors(adj[ru]):
            if v != rv:
                k = (min(v, ru), max(v, ru))
                affected.add(k)
        for v in get_neighbors(adj[rv]):
            if v != ru:
                k = (min(v, rv), max(v, rv))
                affected.add(k)
        for v in get_neighbors(adj[cu]):
            if v != cv:
                k = (min(v, cu), max(v, cu))
                affected.add(k)
        for v in get_neighbors(adj[cv]):
            if v != cu:
                k = (min(v, cv), max(v, cv))
                affected.add(k)
                
        # Save old book sizes
        old_books = {k: book_sizes.get(k, 0) for k in affected}
        
        # Apply swap
        del book_sizes[rkey]
        edge_set.remove(rkey)
        adj[ru] &= ~(1 << rv)
        adj[rv] &= ~(1 << ru)
        
        edge_set.add(ckey)
        adj[cu] |= (1 << cv)
        adj[cv] |= (1 << cu)
        current_edges[remove_idx] = [cu, cv]
        
        # Recompute book sizes for affected
        new_max_affected = 0
        for k in affected:
            if k in edge_set:
                u, v = k
                bs = (adj[u] & adj[v]).bit_count()
                book_sizes[k] = bs
                if bs > new_max_affected:
                    new_max_affected = bs
            else:
                if k in book_sizes:
                    del book_sizes[k]
        
        # Determine new global max
        was_max_edge = old_books.get(rkey, 0) == current_max
        if was_max_edge:
            new_max = max(book_sizes.values()) if book_sizes else 0
        else:
            new_max = max(current_max, new_max_affected)
            
        delta = new_max - current_max
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_edges = [list(e) for e in current_edges]
        else:
            # Revert
            edge_set.remove(ckey)
            adj[cu] &= ~(1 << cv)
            adj[cv] &= ~(1 << cu)
            if ckey in book_sizes:
                del book_sizes[ckey]
            
            edge_set.add(rkey)
            adj[ru] |= (1 << rv)
            adj[rv] |= (1 << ru)
            
            for k in affected:
                if k in old_books:
                    if k in edge_set:
                        book_sizes[k] = old_books[k]
                    else:
                        if k in book_sizes:
                            del book_sizes[k]
                else:
                    if k in book_sizes:
                        del book_sizes[k]
                        
            current_edges[remove_idx] = [ru, rv]
        
        T *= alpha
        
    return best_edges, best_max

def construct():
    start_time = time.time()
    n = 40
    target_edges = 401
    deadline = start_time + 148.0  # Safety margin before 150s kill
    
    best_edges = None
    best_score = float('inf')
    
    # Phase 1: Try greedy constructions with different seeds
    greedy_seeds = [42, 123, 456, 789, 1111, 2222]
    for seed in greedy_seeds:
        if time.time() > deadline - 30.0:
            break
        init_edges = _greedy_construct(n, target_edges, seed)
        init_score = _compute_book_sizes(init_edges, n)
        
        # Run SA to refine
        time_limit = min(15.0, deadline - time.time() - 5.0)
        if time_limit < 2.0:
            continue
            
        edges, score = _simulate_annealing(n, target_edges, time_limit=time_limit, seed=seed, initial_edges=init_edges)
        
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Phase 2: Try structural inits if time remains
    if time.time() < deadline - 10.0:
        configs = [
            (9, 31),
            (10, 30),
            (8, 32),
        ]
        seeds = [42, 123, 456]
        
        for a_size, b_size in configs:
            if time.time() > deadline - 10.0:
                break
            for seed in seeds:
                if time.time() > deadline - 10.0:
                    break
                init_edges = _build_structural_init(a_size, b_size, seed)
                init_score = _compute_book_sizes(init_edges, n)
                
                time_limit = min(10.0, deadline - time.time() - 5.0)
                if time_limit < 2.0:
                    continue
                    
                edges, score = _simulate_annealing(n, target_edges, time_limit=time_limit, seed=seed, initial_edges=init_edges)
                
                if score < best_score:
                    best_score = score
                    best_edges = edges
    
    # Phase 3: Final polish with best solution found
    if best_edges is not None and time.time() < deadline - 3.0:
        time_limit = min(5.0, deadline - time.time() - 1.0)
        if time_limit > 1.0:
            edges, score = _simulate_annealing(n, target_edges, time_limit=time_limit, seed=999, initial_edges=best_edges)
            if score < best_score:
                best_score = score
                best_edges = edges
    
    if best_edges is None:
        # Fallback
        best_edges = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return best_edges
