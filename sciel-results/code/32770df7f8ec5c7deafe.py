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
    
    # Local search: try to improve by removing vertices
    # First pass: greedy removal - remove vertices if possible
    changed = True
    while changed:
        changed = False
        for i in range(len(selected)):
            v = selected[i]
            # Check if removing v still covers all cliques
            selected_set = set(selected)
            selected_set.discard(v)
            covers_all = all(any(x in selected_set for x in clique) for clique in relevant_cliques)
            if covers_all:
                selected = [x for x in selected if x != v]
                changed = True
                break  # Restart after removal
    
    # Second pass: 2-for-1 improvement
    # Try to replace two vertices with one that covers all their uniquely-covered cliques
    improved = True
    while improved:
        improved = False
        selected_set = set(selected)
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                v1, v2 = selected[i], selected[j]
                # Find cliques uniquely covered by v1 or v2 (not covered by other selected vertices)
                others = selected_set - {v1, v2}
                unique_to_v1 = vertex_to_cliques[v1] - set().union(*[vertex_to_cliques[v] for v in others]) if others else vertex_to_cliques[v1]
                unique_to_v2 = vertex_to_cliques[v2] - set().union(*[vertex_to_cliques[v] for v in others]) if others else vertex_to_cliques[v2]
                must_cover = unique_to_v1 | unique_to_v2
                
                # Find a vertex w not in selected that covers all must_cover cliques
                for w in range(n):
                    if w in selected_set:
                        continue
                    if must_cover <= vertex_to_cliques[w]:
                        # Replace v1 and v2 with w
                        selected = [x for x in selected if x not in (v1, v2)]
                        selected.append(w)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    # Third pass: more aggressive removal
    changed = True
    while changed:
        changed = False
        for i in range(len(selected)):
            v = selected[i]
            selected_set = set(selected)
            selected_set.discard(v)
            covers_all = all(any(x in selected_set for x in clique) for clique in relevant_cliques)
            if covers_all:
                selected = [x for x in selected if x != v]
                changed = True
                break
    
    return sorted(selected)

def construct():
    """
    Construct vertex lists for the three fixed random graph instances.
    Each list must meet every maximal clique with at least two vertices.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    # Allocate time proportional to n^2 (clique enumeration cost)
    total_n2 = sum(n * n for n, p, seed in instances)
    time_budget = 130.0  # Leave 10s safety margin
    
    for idx, (n, p, seed) in enumerate(instances):
        remaining_time = deadline - time.time()
        if remaining_time < 5.0:
            # Not enough time, return empty list for remaining
            for _ in range(idx, len(instances)):
                results.append([])
            break
        
        # Allocate time for this instance based on n^2
        n2 = n * n
        allocated = min(remaining_time - 5.0, time_budget * n2 / total_n2)
        allocated = max(allocated, 3.0)  # Minimum 3 seconds
        
        try:
            result = _solve_one_instance(n, p, seed, time_limit=allocated)
            results.append(result)
        except Exception:
            # Fallback: all vertices
            results.append(list(range(n)))
    
    return results
# EVOLVE-BLOCK-END
