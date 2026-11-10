"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from random import Random
from collections import defaultdict

def _random_graph(n, p, seed):
    """Generate a random graph using the same RNG sequence as the seed program."""
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _bron_kerbosch_maximal_cliques(n, adj):
    """
    Enumerate all maximal cliques using Bron-Kerbosch with pivot.
    adj: list of sets, adj[i] is the set of neighbors of i.
    Returns list of frozensets (each a clique).
    """
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        if P:
            u = max(P, key=lambda v: len(adj[v] & P))
            candidates = P - adj[u]
        elif X:
            u = max(X, key=lambda v: len(adj[v] & X))
            candidates = P
            return
        else:
            return
        
        for v in list(candidates):
            R.add(v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R, P_new, X_new)
            R.remove(v)
            P.remove(v)
            X.add(v)
    
    initial_P = set(range(n))
    bk(set(), initial_P, set())
    return cliques

def _solve_set_cover(vertices, cliques, n):
    """
    Solve the minimum hitting set: select minimum vertices to hit all cliques.
    Uses greedy set cover + iterative local search for reduction.
    """
    if not cliques:
        return set()
    
    # Map each vertex to the cliques it covers
    vertex_to_cliques = defaultdict(list)
    for ci, clique in enumerate(cliques):
        for v in clique:
            vertex_to_cliques[v].append(ci)
    
    # Precompute coverage as sets for fast intersection
    coverage = {v: set(vertex_to_cliques.get(v, [])) for v in range(n)}
    
    # Greedy set cover
    selected = set()
    uncovered = set(range(len(cliques)))
    
    while uncovered:
        best_v = None
        best_count = -1
        for v in range(n):
            if v in selected:
                continue
            count = len(coverage[v] & uncovered)
            if count > best_count:
                best_count = count
                best_v = v
        
        if best_v is None:
            break
        
        selected.add(best_v)
        for ci in coverage[best_v]:
            if ci in uncovered:
                uncovered.remove(ci)
    
    # Local search: repeatedly try to remove redundant vertices
    # Maintain clique_cover_count[ci] = number of selected vertices covering clique ci
    clique_cover_count = defaultdict(int)
    for v in selected:
        for ci in coverage[v]:
            clique_cover_count[ci] += 1
    
    changed = True
    while changed:
        changed = False
        for v in list(selected):
            # Check if all cliques covered by v are also covered by other selected vertices
            redundant = True
            for ci in coverage[v]:
                if clique_cover_count[ci] <= 1:
                    redundant = False
                    break
            if redundant:
                selected.remove(v)
                for ci in coverage[v]:
                    clique_cover_count[ci] -= 1
                changed = True
    
    return selected

def construct():
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = _random_graph(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all maximal cliques of size >= 2
        cliques = _bron_kerbosch_maximal_cliques(n, adj)
        cliques = [c for c in cliques if len(c) >= 2]
        
        # Solve set cover
        if not cliques:
            out.append([])
        else:
            selected = _solve_set_cover(range(n), cliques, n)
            out.append(sorted(selected))
    
    return out
# EVOLVE-BLOCK-END
