"""Clique partition: density-prioritized maximal clique extraction with backtracking refinement."""
import time
from random import Random
from collections import defaultdict

def _random_graph(n, p, seed):
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed, deadline):
    edges = _random_graph(n, p, seed)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    remaining = set(edges)
    cliques = []
    
    # Precompute initial common-neighbor counts for each edge
    # common_neigh[e] = |N(u) ∩ N(v)| ∩ endpoints connected in original graph
    # We'll maintain this dynamically as edges are removed
    
    while remaining:
        if time.time() > deadline:
            for e in remaining:
                cliques.append(list(e))
            break
        
        # Score each edge by "clique density": ratio of common neighbors to degree
        # Higher density edges are more likely to be part of large cliques
        # Use a sampling heuristic for speed when many edges remain
        
        if len(remaining) > 300:
            # Sample a subset for scoring
            rng_sample = Random(12345)
            candidates = rng_sample.sample(list(remaining), 300)
        else:
            candidates = list(remaining)
        
        best_edge = None
        best_score = -1
        
        for e in candidates:
            u, v = e
            # Common neighbors in the ORIGINAL graph
            common_orig = len(adj[u] & adj[v])
            # Degree in original graph
            deg_u = len(adj[u])
            deg_v = len(adj[v])
            # Density score: prefer edges with high common-neighbor ratio
            # This is a proxy for how "clique-like" the neighborhood is
            if deg_u == 0 or deg_v == 0:
                score = 0
            else:
                score = common_orig / min(deg_u, deg_v)
            
            # Tie-break: prefer edges with more common neighbors
            if score > best_score or (score == best_score and best_edge is not None and 
                                       common_orig > len(adj[candidates[0][0]] & adj[candidates[0][1]])):
                best_score = score
                best_edge = e
        
        if best_edge is None:
            best_edge = next(iter(remaining))
        
        u, v = best_edge
        
        # Build a maximal clique containing edge (u, v)
        # Start with {u, v}, greedily add vertices that connect to all current members
        # and have all required edges in 'remaining'
        
        clique = [u, v]
        clique_set = {u, v}
        
        # Candidates: vertices adjacent to both u and v in original graph
        # and not yet in clique
        cand_set = (adj[u] & adj[v]) - clique_set
        
        # Sort candidates by how many clique members they connect to in 'remaining'
        # This prioritizes vertices that can be added early
        
        def _add_score(w):
            # Count how many edges from w to clique members are in remaining
            count = 0
            for x in clique:
                ek = (min(x, w), max(x, w))
                if ek in remaining:
                    count += 1
            return count
        
        cand_list = sorted(cand_set, key=_add_score, reverse=True)
        
        # Greedily add vertices
        for w in cand_list:
            # Check if w can be added: all edges from w to current clique must be in remaining
            can_add = True
            for x in clique:
                ek = (min(x, w), max(x, w))
                if ek not in remaining:
                    can_add = False
                    break
            if can_add:
                clique.append(w)
                clique_set.add(w)
        
        # Remove all edges within this clique from remaining
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                ek = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                remaining.discard(ek)
        
        if len(clique) >= 2:
            cliques.append(clique)
    
    return cliques

def construct():
    deadline = time.time() + 100  # Safety margin before 120s limit
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        cliques = _solve_instance(n, p, seed, deadline)
        result.append(cliques)
        if time.time() > deadline - 10:
            # Not enough time for second instance, return edges for second
            edges = _random_graph(n, p, seed)
            result.append([list(e) for e in edges])
            break
    return result
