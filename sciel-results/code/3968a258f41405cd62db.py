"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import Counter
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _generate_edges(n, p, seed):
    """Generate edges for a random graph using the same RNG as the verifier."""
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_vertex_cover_milp(n, edges):
    """
    Solve Minimum Vertex Cover using MILP.
    Returns a list of vertices in the cover.
    """
    if not edges:
        return []
    
    # Variables x_v for v in 0..n-1
    # Objective: min sum(x_v)
    c = np.ones(n)
    
    # Constraints: x_u + x_v >= 1 for each edge (u, v)
    # Number of constraints = number of edges
    m = len(edges)
    A = np.zeros((m, n))
    lower = np.ones(m)
    upper = np.inf * np.ones(m)
    
    for i, (u, v) in enumerate(edges):
        A[i, u] = 1
        A[i, v] = 1
        
    constraints = LinearConstraint(A, lower, upper)
    integrality = np.ones(n) # All variables integer (binary)
    bounds = Bounds(lb=0, ub=1)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.status == 0: # Optimal
            # res.x contains the solution
            # Round to nearest integer to be safe
            x = np.round(res.x).astype(int)
            return [i for i in range(n) if x[i] == 1]
        else:
            # If MILP fails, fall back to a simple heuristic
            return _heuristic_vertex_cover(n, edges)
    except Exception:
        return _heuristic_vertex_cover(n, edges)

def _heuristic_vertex_cover(n, edges):
    """
    Fallback heuristic: Greedy Vertex Cover.
    Repeatedly pick the vertex with highest degree in the remaining graph.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    selected = set()
    remaining_edges = set(edges)
    
    while remaining_edges:
        # Find vertex with max degree in remaining graph
        deg = {v: 0 for v in range(n)}
        for u, v in remaining_edges:
            deg[u] += 1
            deg[v] += 1
            
        max_deg = -1
        best_v = -1
        for v in range(n):
            if deg[v] > max_deg:
                max_deg = deg[v]
                best_v = v
                
        if best_v == -1:
            break
            
        selected.add(best_v)
        
        # Remove all edges incident to best_v
        new_remaining = set()
        for u, v in remaining_edges:
            if u != best_v and v != best_v:
                new_remaining.add((u, v))
        remaining_edges = new_remaining
        
    return sorted(selected)

def construct():
    deadline = time.time() + 120  # Safety margin before 150s kill
    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5:
            # Not enough time, return what we have
            break
        
        edges = _generate_edges(n, p, seed)
        
        # Solve Minimum Vertex Cover
        # This is equivalent to hitting all maximal cliques of size >= 2
        cover = _solve_vertex_cover_milp(n, edges)
        
        # Verification step (optional but safe)
        # Check if cover hits all edges
        cover_set = set(cover)
        valid = True
        for u, v in edges:
            if u not in cover_set and v not in cover_set:
                valid = False
                break
                
        if not valid:
            # Fallback if MILP result was invalid (should not happen)
            cover = _heuristic_vertex_cover(n, edges)
            
        results.append(cover)
    
    # If we ran out of time, fill in with trivial solutions
    while len(results) < 3:
        n, p, seed = ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))[len(results)]
        edges = _generate_edges(n, p, seed)
        # Trivial: all vertices
        results.append(list(range(n)))
        
    return results
# EVOLVE-BLOCK-END
