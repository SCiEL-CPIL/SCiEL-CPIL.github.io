"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

_START = time.time()
_LIMIT = 110.0  # safety margin under 150s wall-clock

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency matrix for fast lookup
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Use a set of frozensets for edges to track remaining edges
    remaining_edges = set()
    for u, v in edges:
        if u < v:
            remaining_edges.add(frozenset((u, v)))
        else:
            remaining_edges.add(frozenset((v, u)))
    
    if not remaining_edges:
        return []
    
    partition = []
    
    def get_clique_edges(clique):
        return set(frozenset((min(a, b), max(a, b))) for a, b in combinations(clique, 2))
    
    # Greedy algorithm: repeatedly pick the edge that appears in the most cliques
    # that can still be formed from remaining edges, then expand to a maximal clique
    # and remove all edges in that clique.
    
    # To speed up, we'll use a heuristic: pick the edge (u,v) such that the number of
    # common neighbors of u and v (in the remaining graph) is maximized.
    # Then expand {u,v} to a maximal clique using those common neighbors.
    
    while remaining_edges:
        if not _deadline_ok():
            # Fallback: add remaining edges as 2-cliques
            for e in remaining_edges:
                partition.append(list(e))
            break
        
        # Find the edge with the maximum number of common neighbors in the remaining graph
        best_edge = None
        best_common_count = -1
        
        # To avoid O(E^2) which might be too slow for n=64 (E ~ 1100),
        # we can sample or use a heuristic. But n is small enough.
        # Let's iterate over remaining edges.
        for e in remaining_edges:
            u, v = tuple(e)
            # Count common neighbors of u and v that are connected to both
            # in the remaining graph
            common = 0
            for w in range(n):
                if w == u or w == v:
                    continue
                # Check if (u,w) and (v,w) are in remaining edges
                if frozenset((u, w)) in remaining_edges and frozenset((v, w)) in remaining_edges:
                    common += 1
            if common > best_common_count:
                best_common_count = common
                best_edge = e
        
        if best_edge is None:
            break
            
        u, v = tuple(best_edge)
        
        # Expand {u, v} to a maximal clique
        clique = [u, v]
        
        # Find all common neighbors of u and v in the remaining graph
        candidates = []
        for w in range(n):
            if w == u or w == v:
                continue
            if frozenset((u, w)) in remaining_edges and frozenset((v, w)) in remaining_edges:
                candidates.append(w)
        
        # Greedily add candidates to the clique
        # We want to add candidates that are connected to all current clique members
        # and preferably to other candidates
        while candidates:
            # Find the candidate with the highest degree in the subgraph induced by candidates
            # and also connected to all clique members
            valid_candidates = []
            for w in candidates:
                is_connected = True
                for c in clique:
                    if frozenset((c, w)) not in remaining_edges:
                        is_connected = False
                        break
                if is_connected:
                    valid_candidates.append(w)
            
            if not valid_candidates:
                break
            
            # Pick the one with the most connections to other valid candidates
            best_v = -1
            best_deg = -1
            for w in valid_candidates:
                deg = 0
                for x in valid_candidates:
                    if x != w and frozenset((w, x)) in remaining_edges:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_v = w
            
            if best_v == -1:
                break
            
            clique.append(best_v)
            candidates.remove(best_v)
            
            # Filter candidates: must be connected to best_v
            candidates = [w for w in candidates if frozenset((w, best_v)) in remaining_edges]
        
        # Add this clique to the partition
        partition.append(clique)
        
        # Remove all edges in this clique from remaining_edges
        clique_edges = get_clique_edges(clique)
        remaining_edges -= clique_edges
    
    return partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
# EVOLVE-BLOCK-END
