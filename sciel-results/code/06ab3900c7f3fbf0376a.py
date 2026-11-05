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

def _solve_instance(n, p, seed, deadline):
    """Find a clique partition of the graph minimizing number of cliques."""
    edges = _generate_graph(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency as sets for fast lookup
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
    
    remaining_edges = set(edge_set)
    cliques = []
    
    # Track remaining degree for each vertex
    rem_deg = [0] * n
    for u in range(n):
        rem_deg[u] = len(adj[u])
    
    # Greedy: repeatedly find a large clique from remaining edges
    # Priority: pick vertex with highest remaining degree, then expand
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
        
        # Find the largest clique containing best_v using only remaining edges
        # Start with {best_v}, try to add neighbors that are all mutually connected
        clique = {best_v}
        # Candidates: neighbors of best_v that still have edges to best_v
        candidates = [u for u in adj[best_v] if (min(best_v, u), max(best_v, u)) in remaining_edges]
        
        # Greedily add candidates that are connected to all current clique members
        # Sort by how many connections they have to current clique + each other
        changed = True
        while changed:
            changed = False
            # Try to add each candidate
            for u in candidates[:]:
                if u in clique:
                    continue
                # Check if u is connected to all members of clique via remaining edges
                can_add = True
                for w in clique:
                    e = (min(u, w), max(u, w))
                    if e not in remaining_edges:
                        can_add = False
                        break
                if can_add:
                    clique.add(u)
                    candidates.remove(u)
                    changed = True
        
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
                    # Update remaining degrees
                    rem_deg[u] -= 1
                    rem_deg[v] -= 1
        else:
            # Shouldn't happen, but handle single vertex case
            break
    
    # Handle any remaining edges that couldn't be covered (shouldn't happen if valid)
    # Each remaining edge is itself a clique of size 2
    for e in remaining_edges:
        cliques.append([e[0], e[1]])
    
    return cliques

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
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
