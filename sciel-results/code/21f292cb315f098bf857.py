"""Clique partition: Set Partitioning via ILP with Greedy Fallback."""
import time
import numpy as np
from random import Random
from collections import defaultdict

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_SCIPY_MILP = True
except ImportError:
    HAS_SCIPY_MILP = False

def _random_graph(n, p, seed):
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _find_maximal_cliques(n, adj):
    """Bron-Kerbosch algorithm to find all maximal cliques."""
    cliques = []
    
    def bron_kerbosch(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(tuple(sorted(R)))
            return
        
        # Choose pivot u from P union X to minimize branching
        # Heuristic: pick vertex with max degree in P
        if P:
            u = max(P, key=lambda v: len(adj[v] & P))
        else:
            u = max(X, key=lambda v: len(adj[v] & P)) if X else None
            
        # Branch on vertices in P \ N(u)
        candidates = P - adj[u] if u is not None else P
        
        for v in list(candidates):
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
            
    bron_kerbosch(set(), set(range(n)), set())
    return cliques

def _solve_instance(n, p, seed, deadline):
    edges = _random_graph(n, p, seed)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Map edge to index
    edge_list = list(edges)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    m = len(edge_list)
    
    # Find maximal cliques
    cliques = _find_maximal_cliques(n, adj)
    
    # Filter cliques to only those that are valid (size >= 2) and cover at least one edge
    # Also, we might have duplicate cliques? Bron-Kerbosch returns unique maximal cliques.
    # However, not all subsets of a maximal clique are maximal.
    # The ILP formulation $\sum x_C = 1$ allows selecting non-maximal cliques if they are in the set.
    # But we only generated maximal ones.
    # Is it possible that the optimal partition uses a non-maximal clique?
    # Yes. E.g., if two maximal cliques overlap heavily, it might be better to split them.
    # However, any non-maximal clique is a subset of a maximal clique.
    # If we select a maximal clique, we cover all its edges.
    # If the optimal solution uses a subset, it implies we are "wasting" some edges in the maximal clique that could have been covered by another clique?
    # No, the constraint is equality.
    # If we restrict to maximal cliques, we might not find the optimal solution if the optimal solution requires a non-maximal clique.
    # Example: Triangle 1-2-3 and Triangle 1-2-4. Edge 1-2 is shared.
    # Maximal cliques: {1,2,3}, {1,2,4}.
    # We cannot select both.
    # If we select {1,2,3}, edges (1,2), (1,3), (2,3) are covered.
    # Edge (1,4), (2,4) must be covered by other cliques.
    # If we could select {1,2}, it covers (1,2). Then {1,3,?} no, 3 is not connected to 4.
    # Actually, using non-maximal cliques increases the number of cliques, which we want to minimize.
    # So we prefer larger cliques.
    # It is generally safe to assume that an optimal partition consists of maximal cliques?
    # Counter-example:
    # Graph: Square 1-2-3-4-1 with diagonal 1-3.
    # Triangles: {1,2,3}, {1,3,4}.
    # Maximal cliques: {1,2,3}, {1,3,4}.
    # Edge 1-3 is in both.
    # If we pick {1,2,3}, we cover 1-2, 2-3, 1-3.
    # Remaining: 3-4, 4-1. These form a clique {3,4,1}? No, 1-4 is edge, 3-4 is edge, 1-3 is edge.
    # So {1,3,4} is a clique.
    # If we pick {1,3,4}, we cover 1-3, 3-4, 4-1.
    # Remaining: 1-2, 2-3.
    # We can pick {1,2,3} again? No, 1-3 is used.
    # So we must pick {1,2} and {2,3}? Or {1,2,3} is not available because 1-3 is used.
    # So we pick {1,2} and {2,3}. Total cliques: {1,3,4}, {1,2}, {2,3}. Count = 3.
    # If we picked {1,2,3} first:
    # Remaining: 3-4, 4-1.
    # Pick {3,4,1}? 1-3 used. So {3,4} and {4,1}? Or {1,3,4} not available.
    # Pick {3,4} and {1,4}. Count = 3.
    # Same.
    
    # What if we had a non-maximal clique {1,3}?
    # If we pick {1,3}, then {1,2,3} is not available (1-3 used).
    # We would pick {1,2} and {2,3} for the rest of that triangle?
    # This seems worse.
    
    # So restricting to maximal cliques is a good heuristic, but might miss the global optimum.
    # However, for dense graphs, maximal cliques are usually large enough.
    
    if not cliques:
        return [list(e) for e in edges]
    
    # If ILP is available and problem size is manageable
    if HAS_SCIPY_MILP and len(cliques) < 2000 and time.time() < deadline - 10:
        try:
            return _solve_ilp(cliques, edges, edge_to_idx, n, deadline)
        except Exception:
            pass
            
    # Fallback: Greedy Set Cover
    return _solve_greedy(cliques, edges, edge_to_idx, n, deadline)

def _solve_ilp(cliques, edges, edge_to_idx, n, deadline):
    m = len(edges)
    k = len(cliques)
    
    # Objective: Minimize sum(x_i)
    c = np.ones(k)
    
    # Constraints: For each edge, sum of x_i for cliques containing edge = 1
    # A matrix of size m x k
    # A[i, j] = 1 if edge i is in clique j
    
    # Sparse matrix construction
    # Since scipy milp accepts sparse, let's build it
    from scipy.sparse import lil_matrix
    A = lil_matrix((m, k))
    
    for j, clique in enumerate(cliques):
        clique_set = set(clique)
        for u in clique:
            for v in clique:
                if u < v:
                    e = (u, v)
                    if e in edge_to_idx:
                        i = edge_to_idx[e]
                        A[i, j] = 1
    
    A = A.tocsr()
    
    # Constraints: A @ x = 1
    lb = np.ones(m)
    ub = np.ones(m)
    
    constraints = LinearConstraint(A, lb, ub)
    bounds = Bounds(0, 1)
    
    res = milp(c, constraints=constraints, integrality=np.ones(k), bounds=bounds, options={"time_limit": 60})
    
    if res.status != 0:
        raise RuntimeError("ILP failed")
        
    selected_cliques = []
    for j in range(k):
        if res.x[j] > 0.5:
            selected_cliques.append(list(cliques[j]))
            
    # Verify coverage
    covered_edges = set()
    for c in selected_cliques:
        for i in range(len(c)):
            for j in range(i+1, len(c)):
                e = (min(c[i], c[j]), max(c[i], c[j]))
                if e in covered_edges:
                    raise RuntimeError("Conflict detected")
                covered_edges.add(e)
                
    if len(covered_edges) != m:
        # Add missing edges as 2-cliques
        for e in edges:
            if e not in covered_edges:
                selected_cliques.append(list(e))
                
    return selected_cliques

def _solve_greedy(cliques, edges, edge_to_idx, n, deadline):
    remaining_edges = set(edges)
    selected_cliques = []
    
    # Sort cliques by size descending
    cliques_sorted = sorted(cliques, key=len, reverse=True)
    
    # We can improve this by prioritizing cliques that cover many "rare" edges
    # For now, simple greedy
    
    for clique in cliques_sorted:
        if time.time() > deadline:
            break
        clique_edges = set()
        valid = True
        for i in range(len(clique)):
            for j in range(i+1, len(clique)):
                e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                if e in remaining_edges:
                    clique_edges.add(e)
                else:
                    # If any edge is already covered, this clique is not disjoint
                    # We can still use it if we remove the covered edges? No, it must be a clique in the original graph.
                    # But if we select it, we cover its edges. If some are already covered, it's a conflict.
                    # So we skip if any edge is covered.
                    valid = False
                    break
            if not valid:
                break
        
        if valid and len(clique_edges) > 0:
            selected_cliques.append(clique)
            remaining_edges -= clique_edges
            
    # Add remaining edges
    for e in remaining_edges:
        selected_cliques.append(list(e))
        
    return selected_cliques

def construct():
    deadline = time.time() + 100
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        cliques = _solve_instance(n, p, seed, deadline)
        result.append(cliques)
        if time.time() > deadline - 10:
            edges = _random_graph(n, p, seed)
            result.append([list(e) for e in edges])
            break
    return result
