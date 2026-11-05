"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _generate_graph(n, p, seed):
    """Generate a random graph using the same RNG as the seed program."""
    rng = random.Random(seed)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    # Build adjacency as sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj, edges

def _bron_kerbosch_maximal_cliques(adj, n):
    """
    Enumerate all maximal cliques using Bron-Kerbosch with pivot.
    Returns list of frozensets of vertices.
    """
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P ∪ X that maximizes |N(u) ∩ P|
        P_union_X = P | X
        if not P_union_X:
            return
        
        # Pick pivot to minimize branching
        best_pivot = None
        max_count = -1
        for u in P_union_X:
            count = len(adj[u] & P)
            if count > max_count:
                max_count = count
                best_pivot = u
        
        if best_pivot is None:
            return
        
        # Branch on vertices in P \ N(pivot)
        candidates = P - adj[best_pivot]
        for v in list(candidates):
            R_new = R | {v}
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R_new, P_new, X_new)
            P.remove(v)
            X.add(v)
    
    bk(set(), set(range(n)), set())
    return cliques

def _solve_one_instance(n, p, seed, time_limit=10.0):
    """
    Find a small vertex set that meets every maximal clique of size >= 2.
    This is a hitting set problem: find minimum set S such that for every
    maximal clique C with |C| >= 2, S ∩ C != empty.
    """
    adj, edges = _generate_graph(n, p, seed)
    
    # Find all maximal cliques of size >= 2
    all_cliques = _bron_kerbosch_maximal_cliques(adj, n)
    relevant_cliques = [c for c in all_cliques if len(c) >= 2]
    
    if not relevant_cliques:
        return []
    
    # Greedy set cover: iteratively pick the vertex that covers the most uncovered cliques
    uncovered = set(range(len(relevant_cliques)))
    selected = []
    
    # Precompute for each vertex, which cliques it belongs to
    vertex_to_cliques = {v: set() for v in range(n)}
    for i, clique in enumerate(relevant_cliques):
        for v in clique:
            vertex_to_cliques[v].add(i)
    
    while uncovered:
        # Find vertex that covers the most uncovered cliques
        best_vertex = None
        best_count = -1
        for v in range(n):
            count = len(vertex_to_cliques[v] & uncovered)
            if count > best_count:
                best_count = count
                best_vertex = v
        
        if best_vertex is None or best_count <= 0:
            break
        
        selected.append(best_vertex)
        uncovered -= vertex_to_cliques[best_vertex]
    
    # Local search: try to improve by swapping or removing
    # Try to remove any vertex if the set still covers all cliques
    changed = True
    while changed:
        changed = False
        for i, v in enumerate(list(selected)):
            candidate = selected[:i] + selected[i+1:]
            # Check if candidate still covers all relevant cliques
            covers_all = all(any(x in candidate for x in clique) for clique in relevant_cliques)
            if covers_all:
                selected = candidate
                changed = True
    
    # Try to swap each selected vertex with an unselected one if it reduces size or maintains
    # Actually, since we want to minimize size, let's try to see if we can do better with local search
    # Try adding a vertex and removing two? That's more complex. Let's stick with simple local search.
    
    return sorted(selected)

def construct():
    """
    Construct vertex lists for the three fixed random graph instances.
    Each list must meet every maximal clique with at least two vertices.
    """
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        if time.time() > deadline - 10.0:
            # Not enough time, return what we have plus empty lists for remaining
            results.append([])
            continue
        
        try:
            result = _solve_one_instance(n, p, seed, time_limit=10.0)
            results.append(result)
        except Exception:
            # Fallback: all vertices
            results.append(list(range(n)))
    
    return results
# EVOLVE-BLOCK-END
