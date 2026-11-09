"""Clique partition via Bron-Kerbosch maximal cliques + greedy set cover with local improvement."""
# EVOLVE-BLOCK-START
import time
from random import Random

_START = time.time()
_LIMIT = 115.0  # safety margin under 120s


def _deadline_ok():
    return time.time() - _START < _LIMIT


def _random_edges(n, p, seed):
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _bron_kerbosch(adj, max_clique_size=6):
    """Find all maximal cliques using Bron-Kerbosch with pivoting."""
    n = len(adj)
    adj_list = [set() for _ in range(n)]
    for u in range(n):
        adj_list[u] = adj[u]
    
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(tuple(sorted(R)))
            return
        if not _deadline_ok():
            return
        
        # Pivot selection: choose vertex with max |P ∩ N(u)|
        # For efficiency, pick from P ∪ X
        candidates = P | X
        if not candidates:
            return
        
        # Choose pivot with max degree into P
        pivot = max(candidates, key=lambda v: len(adj_list[v] & P))
        # Iterate over P \ N(pivot)
        for v in list(P - adj_list[pivot]):
            if not _deadline_ok():
                return
            new_R = R | {v}
            if len(new_R) > max_clique_size:
                continue
            new_P = P & adj_list[v]
            new_X = X & adj_list[v]
            bk(new_R, new_P, new_X)
            P.remove(v)
            X.add(v)
    
    bk(set(), set(range(n)), set())
    return cliques


def _clique_edges(clique):
    """Return set of edges in a clique."""
    edges = set()
    for i in range(len(clique)):
        for j in range(i + 1, len(clique)):
            e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
            edges.add(e)
    return edges


def _greedy_set_cover(remaining_edges, candidate_cliques, adj, n):
    """Greedy set cover: repeatedly pick clique covering most remaining edges."""
    if not remaining_edges:
        return []
    
    # Precompute edge sets for each candidate clique
    clique_edge_sets = []
    for cl in candidate_cliques:
        eset = _clique_edges(cl)
        if eset <= remaining_edges:
            clique_edge_sets.append((cl, eset))
    
    chosen = []
    remaining = set(remaining_edges)
    
    while remaining:
        if not _deadline_ok():
            break
        
        best = None
        best_cov = 0
        for cl, eset in clique_edge_sets:
            cov = len(eset & remaining)
            if cov > best_cov:
                best_cov = cov
                best = (cl, eset)
        
        if best is None or best_cov == 0:
            # No clique covers any remaining edge; add single-edge cliques
            for e in list(remaining):
                chosen.append([e[0], e[1]])
            remaining.clear()
            break
        
        cl, eset = best
        chosen.append(list(cl))
        remaining -= eset
        # Remove cliques that no longer fully fit
        clique_edge_sets = [(c, es) for c, es in clique_edge_sets if es <= remaining]
    
    # Fallback for any leftover
    for e in remaining:
        chosen.append([e[0], e[1]])
    
    return chosen


def _local_improvement(chosen, adj, n, max_iters=50):
    """Try to merge two cliques into one if their union is a clique."""
    if not _deadline_ok():
        return chosen
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if not _deadline_ok():
            break
        
        new_chosen = list(chosen)
        for i in range(len(new_chosen)):
            if not _deadline_ok():
                break
            for j in range(i + 1, len(new_chosen)):
                if not _deadline_ok():
                    break
                cl_i = set(new_chosen[i])
                cl_j = set(new_chosen[j])
                union = cl_i | cl_j
                # Check if union is a clique
                is_clique = True
                for u in union:
                    for v in union:
                        if u == v:
                            continue
                        if v not in adj[u]:
                            is_clique = False
                            break
                    if not is_clique:
                        break
                if is_clique and len(union) >= 2:
                    # Verify edges: every edge in union clique must be in chosen
                    union_edges = set()
                    for a in union:
                        for b in union:
                            if a < b:
                                union_edges.add((a, b))
                    # Check all these edges are covered by cliques i and j
                    ci_edges = _clique_edges(tuple(sorted(cl_i)))
                    cj_edges = _clique_edges(tuple(sorted(cl_j)))
                    if union_edges == ci_edges | cj_edges:
                        new_chosen[i] = list(sorted(union))
                        new_chosen.pop(j)
                        improved = True
                        break
            if improved:
                break
        if improved:
            chosen = new_chosen
    
    return chosen


def _solve_one(n, p, seed):
    edges = _random_edges(n, p, seed)
    edge_set = set(edges)
    adj = _adjacency(n, edges)
    
    # Find all maximal cliques up to size 6
    cliques = _bron_kerbosch(adj, max_clique_size=6)
    
    # Filter: only keep cliques whose all edges are in the graph
    valid_cliques = []
    for cl in cliques:
        eset = _clique_edges(cl)
        if eset <= edge_set:
            valid_cliques.append(cl)
    
    # Greedy set cover
    chosen = _greedy_set_cover(edge_set, valid_cliques, adj, n)
    
    # Local improvement
    chosen = _local_improvement(chosen, adj, n)
    
    return chosen


def construct():
    res1 = _solve_one(32, 0.6, 601)
    if not _deadline_ok():
        # Fallback: single-edge cliques
        edges1 = _random_edges(32, 0.6, 601)
        res1 = [[e[0], e[1]] for e in edges1]
    
    res2 = _solve_one(64, 0.55, 602)
    if not _deadline_ok():
        edges2 = _random_edges(64, 0.55, 602)
        res2 = [[e[0], e[1]] for e in edges2]
    
    return [res1, res2]
# EVOLVE-BLOCK-END
