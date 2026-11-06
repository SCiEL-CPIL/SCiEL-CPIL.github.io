# EVOLVE-BLOCK-START
import time
import heapq
from random import Random

def _build_graph(n, p, seed):
    rng = Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch(adj, deadline):
    """
    Enumerate all maximal cliques using Bron-Kerbosch with pivot.
    """
    n = len(adj)
    # Use sets for R, P, X
    maximal_cliques = []
    
    def expand(R, P, X):
        if time.time() > deadline:
            return
        if not P and not X:
            maximal_cliques.append(R)
            return
        if not P:
            return
            
        # Pivot selection
        # Choose v in P U X that maximizes |P ∩ N(v)|
        pivot = None
        max_count = -1
        # Iterate over P U X
        # To avoid creating a new set P|X every time, we can just iterate P then X
        # But we need to check if v is in P or X.
        # Actually, we just need to pick a pivot from P U X.
        # Let's just pick the first one? No, heuristic is better.
        # Let's try picking v in P with max degree in P.
        for v in P:
            count = len(P & adj[v])
            if count > max_count:
                max_count = count
                pivot = v
                
        if pivot is None:
            # If P is empty, we returned earlier.
            # So pivot should be found.
            return
            
        # Recurse on P \ N(pivot)
        # We need to iterate over vertices in P that are NOT neighbors of pivot
        # We create a list to avoid modifying set while iterating
        p_not_neighbors = [v for v in P if v not in adj[pivot]]
        
        for v in p_not_neighbors:
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            expand(new_R, new_P, new_X)
            P.remove(v)
            X.add(v)
            
    expand(set(), set(range(n)), set())
    return maximal_cliques

def _greedy_set_cover(adj, deadline):
    n = len(adj)
    edges = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.add((u, v))
                
    if not edges:
        return []
        
    # Enumerate maximal cliques
    cliques = _bron_kerbosch(adj, deadline)
    
    # Filter to size >= 2
    cliques = [c for c in cliques if len(c) >= 2]
    
    if not cliques:
        return [[u, v] for u, v in edges]
        
    # Precompute edges for each clique
    clique_edges = []
    for c in cliques:
        c_edges = set()
        c_list = sorted(c)
        for i in range(len(c_list)):
            for j in range(i + 1, len(c_list)):
                c_edges.add((c_list[i], c_list[j]))
        clique_edges.append(c_edges)
        
    uncovered = set(edges)
    selected_cliques = []
    
    # Priority queue: (-count, index)
    # We don't store the clique itself to save memory, just index
    heap = []
    counts = [len(ce) for ce in clique_edges]
    for i in range(len(cliques)):
        if counts[i] > 0:
            heapq.heappush(heap, (-counts[i], i))
            
    while uncovered and time.time() < deadline:
        if not heap:
            break
            
        neg_count, idx = heapq.heappop(heap)
        
        # Get valid edges (intersection with uncovered)
        c_edges = clique_edges[idx]
        valid_edges = c_edges & uncovered
        
        if not valid_edges:
            continue
            
        # Select this clique
        selected_cliques.append(cliques[idx])
        
        # Remove covered edges
        uncovered -= valid_edges
        
        # Update heap for other cliques
        # Iterate through all cliques to see if they are affected
        # This is O(M) per selection, total O(M^2)
        for j in range(len(cliques)):
            if j == idx:
                continue
            # Check if clique j shares any edge with clique idx
            # If so, its count might have decreased
            # We can check if (clique_edges[j] & c_edges) is not empty
            if clique_edges[j] & c_edges:
                new_count = len(clique_edges[j] & uncovered)
                if new_count > 0:
                    heapq.heappush(heap, (-new_count, j))
                    
    # Cover remaining edges
    for e in uncovered:
        if time.time() > deadline:
            break
        selected_cliques.append([e[0], e[1]])
        
    return selected_cliques

def _solve_instance(n, p, seed, deadline):
    adj = _build_graph(n, p, seed)
    cliques = _greedy_set_cover(adj, deadline)
    return cliques

def construct():
    start = time.time()
    deadline = start + 140
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            adj = _build_graph(n, p, seed)
            cliques = []
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        cliques.append([u, v])
            results.append(cliques)
            continue
        
        try:
            cliques = _solve_instance(n, p, seed, deadline)
            results.append(cliques)
        except Exception:
            adj = _build_graph(n, p, seed)
            cliques = []
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        cliques.append([u, v])
            results.append(cliques)
            
    return results
# EVOLVE-BLOCK-END
