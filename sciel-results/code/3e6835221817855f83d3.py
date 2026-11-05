"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools

def _score(edges, n=16):
    """Compute the largest triangle degree sum."""
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    max_sum = 0
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            common = adj[u] & adj[v]
            for w in common:
                if w > v:
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
    return max_sum

def _build_adj_deg(edges, n=16):
    adj = [set() for _ in range(n)]
    deg = [0] * n
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        edge_set.add((u, v))
    return adj, deg, edge_set

def _score_from_adj_deg(adj, deg, n=16):
    max_sum = 0
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            common = adj[u] & adj[v]
            for w in common:
                if w > v:
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
    return max_sum

def _random_graph(n=16, target_edges=65, seed=None):
    if seed is not None:
        random.seed(seed)
    all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(all_possible)
    edges = [list(e) for e in all_possible[:target_edges]]
    return edges

def _local_search(edges, n=16, target_edges=65, time_limit=10, seed=None):
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    
    adj, deg, edge_set = _build_adj_deg(edges, n)
    current_score = _score_from_adj_deg(adj, deg, n)
    best_score = current_score
    best_edges = [list(e) for e in edges]
    
    all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    while time.time() - start_time < time_limit:
        # Pick a random edge to remove
        if not edges:
            break
        idx = random.randint(0, len(edges)-1)
        u, v = edges[idx]
        if u > v:
            u, v = v, u
        removed = (u, v)
        
        # Try a few random non-edges to add
        for _ in range(30):
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            if (a, b) in edge_set or (a, b) == removed:
                continue
            
            # Simulate the swap
            new_edge_set = set(edge_set)
            new_edge_set.remove(removed)
            new_edge_set.add((a, b))
            
            # Check if this creates a valid simple graph (it should, since we're swapping)
            # Update adj and deg incrementally would be faster, but let's just recompute for simplicity
            # Actually, let's try a quick check: does the new graph have the same number of edges? Yes.
            # Let's just recompute score.
            
            new_edges = [list(e) for e in new_edge_set]
            new_adj, new_deg, _ = _build_adj_deg(new_edges, n)
            new_score = _score_from_adj_deg(new_adj, new_deg, n)
            
            if new_score < current_score:
                # Accept
                edges = new_edges
                edge_set = new_edge_set
                adj = new_adj
                deg = new_deg
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = [e[:] for e in edges]
                break
            elif new_score == current_score and random.random() < 0.1:
                # Occasionally accept neutral moves to escape local minima
                edges = new_edges
                edge_set = new_edge_set
                adj = new_adj
                deg = new_deg
                break
    
    return best_edges, best_score

def construct():
    random.seed(42)
    n = 16
    target_edges = 65
    
    best_score = float('inf')
    best_edges = None
    
    # Start 1: The seed construction (K_{8,8} + one internal edge)
    edges_seed = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    score_seed = _score(edges_seed, n)
    if score_seed < best_score:
        best_score = score_seed
        best_edges = [e[:] for e in edges_seed]
    
    # Try multiple random starts with local search
    for attempt in range(10):
        if time.time() > 110:
            break
        edges_init = _random_graph(n, target_edges, seed=100 + attempt)
        try:
            edges_opt, score_opt = _local_search(
                edges_init, n, target_edges, 
                time_limit=min(10, 110 - time.time()),
                seed=200 + attempt
            )
            if score_opt < best_score:
                best_score = score_opt
                best_edges = [e[:] for e in edges_opt]
        except Exception:
            continue
    
    # Also try local search from the seed
    if time.time() < 110:
        try:
            edges_opt, score_opt = _local_search(
                best_edges, n, target_edges,
                time_limit=min(10, 110 - time.time()),
                seed=300
            )
            if score_opt < best_score:
                best_score = score_opt
                best_edges = [e[:] for e in edges_opt]
        except Exception:
            pass
    
    if best_edges is None:
        best_edges = edges_seed
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
