"""
K4-free graph without independent 6-set, maximizing n.
Uses a greedy + local search heuristic to find large graphs.
"""
import time
import random
import itertools

def _has_k4(adj, n):
    """Check if graph has a K4. adj is list of sets."""
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                continue
            for k in range(j + 1, n):
                if k not in adj[i] or k not in adj[j]:
                    continue
                # i,j,k form a triangle; check if any common neighbor l > k
                common = adj[i] & adj[j] & adj[k]
                for l in common:
                    if l > k:
                        return True
    return False

def _has_independent_6(adj, n):
    """Check if graph has an independent set of size 6."""
    # Independent set in G = clique in complement
    comp = [set(range(n)) - {i} - adj[i] for i in range(n)]
    
    # Find clique of size 6 in comp
    # Use a simple backtracking with pruning
    def find_clique(vertices, size):
        if size == 0:
            return True
        if not vertices:
            return False
        # Pruning: if len(vertices) < size, impossible
        if len(vertices) < size:
            return False
        # Try each vertex
        v_list = list(vertices)
        for idx, v in enumerate(v_list):
            # Pruning: if remaining vertices (including v) < size, break
            if len(v_list) - idx < size:
                break
            new_vertices = vertices - {v} & comp[v]
            if find_clique(new_vertices, size - 1):
                return True
            vertices = vertices - {v}
        return False
    
    return find_clique(set(range(n)), 6)

def _is_valid(adj, n):
    return not _has_k4(adj, n) and not _has_independent_6(adj, n)

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _greedy_build(n, seed=42):
    """
    Greedy construction: add vertices one by one, choosing edges to avoid K4 and independent 6.
    Actually, let's start with an empty graph and add edges, or start with complete and remove.
    Better: start with random graph and repair.
    """
    pass

def _local_search(edges, n, max_iter=5000, seed=42):
    """
    Local search: start from a random graph, try to add edges (to reduce independent sets)
    while avoiding K4. Or remove edges to avoid K4 while maintaining no independent 6.
    
    Strategy: 
    1. Start with a random graph with edge probability p.
    2. While invalid:
       - If has K4: remove an edge from a K4.
       - If has independent 6: add an edge within the independent 6.
    3. Local search: try flipping edges to improve "margin" or just try to reach larger n.
    """
    rng = random.Random(seed)
    
    # Initial edge probability: need to balance K4 (favor sparser) and independent 6 (favor denser).
    # For n=20, p=0.5 might work?
    p = 0.5
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i].add(j)
                adj[j].add(i)
    
    # Repair phase
    for _ in range(10000):
        if _has_k4(adj, n):
            # Find a K4 and remove an edge
            k4 = _find_k4(adj, n)
            if k4 is None:
                break
            # Remove one edge from K4
            u, v = k4[0], k4[1]
            adj[u].remove(v)
            adj[v].remove(u)
        elif _has_independent_6(adj, n):
            # Find an independent 6 and add an edge
            indep6 = _find_independent_6(adj, n)
            if indep6 is None:
                break
            # Add an edge between two vertices in indep6
            u, v = rng.sample(indep6, 2)
            adj[u].add(v)
            adj[v].add(u)
        else:
            break
    
    # Local search phase
    # Try to flip edges to maintain validity and perhaps allow growing n later.
    # For now, just return if valid.
    if _is_valid(adj, n):
        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if j in adj[i]]
        return edges
    
    return None

def _find_k4(adj, n):
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                continue
            for k in range(j + 1, n):
                if k not in adj[i] or k not in adj[j]:
                    continue
                common = adj[i] & adj[j] & adj[k]
                for l in common:
                    if l > k:
                        return [i, j, k, l]
    return None

def _find_independent_6(adj, n):
    comp = [set(range(n)) - {i} - adj[i] for i in range(n)]
    
    def find_clique(vertices, size):
        if size == 0:
            return []
        if not vertices:
            return None
        if len(vertices) < size:
            return None
        v_list = list(vertices)
        for idx, v in enumerate(v_list):
            if len(v_list) - idx < size:
                break
            new_vertices = vertices - {v} & comp[v]
            res = find_clique(new_vertices, size - 1)
            if res is not None:
                return [v] + res
            vertices = vertices - {v}
        return None
    
    return find_clique(set(range(n)), 6)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_n = 0
    best_edges = None
    
    # Try increasing n
    # Exoo's graph is 35. We aim for 35 or higher.
    # Let's try to find a valid graph for n=35 first, then 36, etc.
    # But local search is slow. Let's use a heuristic.
    
    # Known: Paley graph of order 17 is K4-free and alpha=5? No, alpha(Paley 17) = 5?
    # Actually, let's just run local search for n=35.
    
    for n in [35, 36, 37]:
        if time.time() > deadline - 30:
            break
            
        found = False
        for seed in range(100):
            if time.time() > deadline - 10:
                break
            edges = _local_search(None, n, max_iter=200, seed=seed)
            if edges is not None:
                # Verify
                adj = _build_adj(edges, n)
                if _is_valid(adj, n):
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    found = True
                    break
        
        if found:
            # Try to grow?
            # If we found 35, can we try 36?
            # The loop above already tries 36 if 35 succeeded? No, it breaks on found.
            # Let's restructure: try n from 35 up to 40.
            pass
    
    # If we didn't find 35, try smaller?
    # The seed is 17. We should at least beat 17.
    if best_n < 17:
        # Fallback to Paley 17
        squares = {x * x % 17 for x in range(1, 17)}
        best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
        best_n = 17
    
    return {'n': best_n, 'edges': best_edges}
