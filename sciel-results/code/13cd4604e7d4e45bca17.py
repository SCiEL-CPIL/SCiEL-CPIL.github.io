"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from random import Random

def _generate_graph(n, p, seed):
    """Generate graph edges matching the verifier's random_graph."""
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _find_large_clique(start_v, rem_adj, remaining_edges, n):
    """Find a large clique containing start_v using only remaining edges.
    Uses greedy with candidate ordering by mutual connectivity."""
    clique = {start_v}
    # Candidates: neighbors of start_v that still have edges to start_v
    candidates = set(rem_adj[start_v])
    
    # Greedily add candidates, ordering by how well connected they are
    # to the current clique and to each other
    while candidates:
        best_candidate = None
        best_score = -1
        for u in candidates:
            # Check if u can be added (connected to all clique members)
            can_add = True
            for w in clique:
                e = (min(u, w), max(u, w))
                if e not in remaining_edges:
                    can_add = False
                    break
            if not can_add:
                continue
            # Score: number of connections u has to other candidates that are also addable
            # Plus connections to clique (all should be there)
            score = 0
            for v in candidates:
                if v == u:
                    continue
                e = (min(u, v), max(u, v))
                if e in remaining_edges:
                    score += 1
            # Also add bonus for degree in remaining graph
            score += len(rem_adj[u]) * 0.1
            if score > best_score:
                best_score = score
                best_candidate = u
        
        if best_candidate is None:
            break
        
        clique.add(best_candidate)
        candidates.discard(best_candidate)
    
    return clique

def _solve_instance(n, p, seed, deadline):
    """Find a clique partition of the graph minimizing number of cliques."""
    edges = _generate_graph(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency as sets for fast lookup
    rem_adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        edge_set.add((u, v))
    
    remaining_edges = set(edge_set)
    cliques = []
    
    # Track remaining degree for each vertex
    rem_deg = [0] * n
    for u in range(n):
        rem_deg[u] = len(rem_adj[u])
    
    # Greedy: repeatedly find a large clique from remaining edges
    while remaining_edges:
        if time.time() > deadline:
            break
        
        # Pick the vertex with highest remaining degree
        best_v = -1
        best_deg = -1
        for v in range(n):
            if rem_deg[v] > best_deg:
                best_deg = rem_deg[v]
                best_v = v
        
        if best_v < 0:
            break
        
        # Find a large clique containing best_v
        clique = _find_large_clique(best_v, rem_adj, remaining_edges, n)
        
        # Convert clique to sorted list
        clique_list = sorted(clique)
        if len(clique_list) >= 2:
            cliques.append(clique_list)
            
            # Remove all edges within this clique from remaining
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    u, v = clique_list[i], clique_list[j]
                    e = (u, v)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                        rem_adj[u].discard(v)
                        rem_adj[v].discard(u)
                        rem_deg[u] -= 1
                        rem_deg[v] -= 1
        else:
            # Shouldn't happen, but handle single vertex case
            break
    
    # Handle any remaining edges that couldn't be covered
    for e in remaining_edges:
        cliques.append([e[0], e[1]])
    
    # Local improvement: try to merge cliques or reassign
    # For each pair of cliques, check if their union is a clique
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_cliques = []
        i = 0
        while i < len(cliques):
            c1 = set(cliques[i])
            j = i + 1
            merged = False
            while j < len(cliques):
                c2 = set(cliques[j])
                # Check if c1 ∪ c2 is a clique
                union = c1 | c2
                is_clique = True
                for a in union:
                    for b in union:
                        if a >= b:
                            continue
                        e = (a, b)
                        if e not in edge_set:
                            is_clique = False
                            break
                    if not is_clique:
                        break
                if is_clique and len(union) >= 2:
                    new_cliques.append(sorted(union))
                    merged = True
                    j += 1
                    # Reset c1 to the new union for further merging
                    c1 = union
                else:
                    j += 1
            if not merged:
                new_cliques.append(cliques[i])
            i += 1
        
        if len(new_cliques) < len(cliques):
            cliques = new_cliques
            improved = True
    
    return cliques

def construct():
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Fallback: each edge as its own clique
            edges = _generate_graph(n, p, seed)
            result.append([[u, v] for u, v in edges])
        else:
            cliques = _solve_instance(n, p, seed, deadline)
            result.append(cliques)
    
    return result
# EVOLVE-BLOCK-END
