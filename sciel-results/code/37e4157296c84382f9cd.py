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
    Returns list of sets (each a clique).
    """
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(set(R))
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        if P:
            u = max(P, key=lambda v: len(adj[v] & P))
            candidates = P - adj[u]
        elif X:
            u = max(X, key=lambda v: len(adj[v] & X))
            candidates = P  # P is empty, so this loop won't run
            # Actually if P is empty, we already returned above
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

def _solve_set_cover_greedy_with_lagrangian(vertices, cliques, n, max_iter=500):
    """
    Solve the set cover problem: select minimum vertices to cover all cliques.
    Uses a greedy approach guided by Lagrangian relaxation / subgradient-like weighting.
    
    vertices: list of vertex indices
    cliques: list of sets of vertex indices (each clique must be hit)
    n: number of vertices
    """
    if not cliques:
        return set()
    
    # Map each vertex to the cliques it covers
    vertex_to_cliques = defaultdict(set)
    for ci, clique in enumerate(cliques):
        for v in clique:
            vertex_to_cliques[v].add(ci)
    
    # Initialize weights for cliques (Lagrangian multipliers)
    # Start with uniform weights
    weights = np.ones(len(cliques))
    
    # We'll do a hybrid: iterative greedy with adaptive weights
    # At each step, pick vertex with best (weight_sum / cost) ratio
    # where weight_sum is sum of weights of uncovered cliques it covers
    
    selected = set()
    uncovered = set(range(len(cliques)))
    
    # Precompute coverage
    coverage = {v: list(vertex_to_cliques.get(v, [])) for v in range(n)}
    
    # Iterative greedy with subgradient-style weight updates
    for iteration in range(max_iter):
        if not uncovered:
            break
        
        # Compute marginal utility for each vertex
        best_v = None
        best_ratio = -1
        for v in range(n):
            if v in selected:
                continue
            # Sum of weights of uncovered cliques that v covers
            w_sum = 0.0
            for ci in coverage[v]:
                if ci in uncovered:
                    w_sum += weights[ci]
            if w_sum > 0:
                # Cost is 1 for each vertex
                ratio = w_sum / 1.0
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_v = v
        
        if best_v is None:
            # No vertex covers any uncovered clique? Shouldn't happen if problem is feasible
            break
        
        # Select this vertex
        selected.add(best_v)
        
        # Remove covered cliques from uncovered
        newly_covered = []
        for ci in coverage[best_v]:
            if ci in uncovered:
                newly_covered.append(ci)
                uncovered.remove(ci)
        
        if not uncovered:
            break
        
        # Subgradient update: increase weights of remaining uncovered cliques
        # to encourage covering them in next iterations
        # Use a simple step: add a constant to uncovered cliques
        step = 0.1
        for ci in uncovered:
            weights[ci] += step
        
        # Occasionally renormalize to prevent numerical issues
        if iteration % 50 == 0:
            weights = weights / (np.sum(weights) + 1e-10)
    
    # Fallback: if there are still uncovered cliques, use pure greedy
    while uncovered:
        best_v = None
        best_count = -1
        for v in range(n):
            if v in selected:
                continue
            count = 0
            for ci in coverage[v]:
                if ci in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_v = v
        if best_v is None:
            break
        selected.add(best_v)
        for ci in coverage[best_v]:
            if ci in uncovered:
                uncovered.remove(ci)
        if not uncovered:
            break
    
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
            selected = _solve_set_cover_greedy_with_lagrangian(range(n), cliques, n)
            out.append(sorted(selected))
    
    return out
# EVOLVE-BLOCK-END
