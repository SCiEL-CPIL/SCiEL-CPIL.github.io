import time
import sys
import numpy as np
from itertools import combinations
from scipy.optimize import linprog

def _make_instance():
    from random import Random
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    return instance

_DEADLINE = 150.0
_T0 = None

def _elapsed():
    return time.time() - _T0

def _time_up(margin=5.0):
    return _elapsed() > _DEADLINE - margin

def construct():
    global _T0
    _T0 = time.time()
    instance = _make_instance()
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if _time_up():
            for remaining_spec in specs[specs.index(spec):]:
                results.append([list(e) for e in instance(*remaining_spec)])
            break
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = _solve_split(n, k, edges)
        results.append(cliques)
    return results

def _solve_split(n, k, edges):
    I_vertices = list(range(k, n))
    edge_set = set()
    for (a, b) in edges:
        edge_set.add((a, b) if a < b else (b, a))
        
    N_in_C = {}
    for v in I_vertices:
        neigh = set()
        for u in range(k):
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edge_set:
                neigh.add(u)
        N_in_C[v] = neigh

    # Strategy: 
    # 1. Select a set of "big" vertices V_big such that their N(v) induced edge sets are disjoint.
    # 2. For v in V_big, use clique {v} U N(v).
    # 3. For v not in V_big, use singletons {v, u}.
    # 4. Cover remaining C-C edges with greedy clique cover.

    V_big = []
    covered_edges = set()
    
    # Sort by degree descending to pick "big" ones first? Or random?
    # Greedy selection: pick v if E(N(v)) is disjoint from covered_edges.
    for v in sorted(I_vertices, key=lambda x: len(N_in_C[x]), reverse=True):
        new_edges = set()
        Nv = N_in_C[v]
        for u1, u2 in combinations(Nv, 2):
            e = (u1, u2) if u1 < u2 else (u2, u1)
            new_edges.add(e)
            
        if not (new_edges & covered_edges):
            V_big.append(v)
            covered_edges.update(new_edges)

    cliques = []
    
    # Add big cliques
    for v in V_big:
        cliques.append([v] + list(N_in_C[v]))
        
    # Add singletons for others
    for v in I_vertices:
        if v not in V_big:
            for u in N_in_C[v]:
                cliques.append([u, v])
                
    # Cover remaining C-C edges
    C_vertices = list(range(k))
    all_C_edges = set()
    for u1, u2 in combinations(C_vertices, 2):
        all_C_edges.add((u1, u2))
        
    remaining_edges = all_C_edges - covered_edges
    
    # Greedy clique cover for remaining edges
    rem_graph = {u: set() for u in C_vertices}
    for e in remaining_edges:
        rem_graph[e[0]].add(e[1])
        rem_graph[e[1]].add(e[0])
        
    # Find maximal cliques in rem_graph
    # Simple greedy: repeatedly find a clique that covers the most uncovered edges
    uncovered = set(remaining_edges)
    while uncovered:
        if not uncovered:
            break
        # Find vertex with highest degree in uncovered subgraph
        # Or just find an edge, extend to clique
        u, v = next(iter(uncovered))
        # Extend to clique
        clique = {u, v}
        # Try to add more vertices
        for w in C_vertices:
            if w not in clique:
                if all((min(w, x), max(w, x)) in uncovered or (min(w, x), max(w, x)) in covered_edges for x in clique):
                    # Wait, we can only use edges that are in remaining_edges?
                    # No, the clique must be a valid clique in the ORIGINAL graph.
                    # All vertices in C are pairwise adjacent in original graph.
                    # So ANY subset of C is a clique.
                    # We just need to pick a subset that covers edges in `uncovered`.
                    # To maximize coverage, we want the largest subset of C that contains u and v and whose internal edges are in `uncovered`?
                    # No, the clique covers ALL its internal edges.
                    # If we pick a clique S, it covers E(S).
                    # We want E(S) to be a subset of `uncovered`? No, E(S) must be disjoint from already covered.
                    # Since we are covering `uncovered`, E(S) must be a subset of `uncovered`.
                    # So S must be a clique in the graph (C, uncovered).
                    
                    # Check if w can be added to clique S={u,v} such that all edges in {u,v,w} are in `uncovered`.
                    if (min(u,w), max(u,w)) in uncovered and (min(v,w), max(v,w)) in uncovered:
                        clique.add(w)
                        
        # Remove covered edges
        for e in combinations(clique, 2):
            ee = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
            if ee in uncovered:
                uncovered.remove(ee)
                
        cliques.append(list(clique))

    return cliques

if __name__ == "__main__":
    _T0 = time.time()
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total cliques: {total}")
    print(f"Time: {time.time() - _T0:.2f}s")
