# EVOLVE-BLOCK-START
import time
from random import Random

def _random_graph(n, p, seed):
    rng = Random(seed)
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    return edges

def _is_clique(vertices, adj):
    verts = sorted(vertices)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            u, v = verts[i], verts[j]
            if v not in adj[u]:
                return False
    return True

def _bron_kerbosch(adj, n):
    """Find all maximal cliques using Bron-Kerbosch with pivot."""
    cliques = []
    # adj[u] is a set of neighbors
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Choose pivot u from P ∪ X with max |P ∩ adj[u]|
        union = P | X
        if not union:
            return
        pivot = max(union, key=lambda u: len(P & adj[u]))
        candidates = [v for v in P if v not in adj[pivot]]
        for v in candidates:
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bk(new_R, new_P, new_X)
            P = P - {v}
            X = X | {v}
    bk(set(), set(range(n)), set())
    return cliques

def _solve_instance(n, p, seed, deadline):
    edges = _random_graph(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find maximal cliques
    maximal_cliques = _bron_kerbosch(adj, n)
    
    # Sort by size descending (larger cliques cover more edges)
    def clique_edges(c):
        return len(c) * (len(c) - 1) // 2
    
    maximal_cliques.sort(key=clique_edges, reverse=True)
    
    # Greedy selection: pick cliques that don't share edges with already-selected
    selected = []
    covered_edges = set()
    
    for clique in maximal_cliques:
        if time.time() > deadline:
            break
        # Check if any edge in this clique is already covered
        verts = sorted(clique)
        clique_edge_set = set()
        conflict = False
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                e = (verts[i], verts[j])
                if e in edges:
                    if e in covered_edges:
                        conflict = True
                        break
                    clique_edge_set.add(e)
                else:
                    # Shouldn't happen for a valid clique, but skip
                    conflict = True
                    break
            if conflict:
                break
        if not conflict and len(clique) >= 2:
            selected.append(list(verts))
            covered_edges.update(clique_edge_set)
    
    # Add remaining edges as 2-cliques
    for e in edges:
        if e not in covered_edges:
            selected.append(list(e))
    
    return selected

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Fallback: return edge list as 2-cliques
            edges = _random_graph(n, p, seed)
            results.append([list(e) for e in edges])
        else:
            result = _solve_instance(n, p, seed, deadline)
            results.append(result)
    return results
# EVOLVE-BLOCK-END
