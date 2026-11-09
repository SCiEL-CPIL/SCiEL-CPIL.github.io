"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from collections import defaultdict

_START = time.time()
_LIMIT = 140.0  # safety margin under 150s wall-clock

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

def _find_maximal_cliques(n, adj, edges_set):
    """
    Find all maximal cliques in the graph.
    Uses a simplified Bron-Kerbosch with pivoting.
    Returns a list of sets of vertices.
    """
    maximal_cliques = []
    
    # Adjacency lists for faster intersection
    adj_lists = [set() for _ in range(n)]
    for u, v in edges_set:
        adj_lists[u].add(v)
        adj_lists[v].add(u)
        
    def bron_kerbosch(R, P, X):
        if not _deadline_ok():
            return
        if not P and not X:
            if len(R) >= 2: # Only keep cliques of size >= 2
                maximal_cliques.append(frozenset(R))
            return
            
        # Pivot selection: choose u in P ∪ X with max |P ∩ N(u)|
        # This reduces the number of recursive calls
        pivot_set = P.union(X)
        if not pivot_set:
            return
            
        max_inter_size = -1
        pivot = None
        for u in pivot_set:
            inter_size = len(P.intersection(adj_lists[u]))
            if inter_size > max_inter_size:
                max_inter_size = inter_size
                pivot = u
                
        # Iterate over P \ N(pivot)
        # We must iterate over a static copy because P changes
        candidates = list(P.difference(adj_lists[pivot]))
        
        for v in candidates:
            new_R = R.union({v})
            new_P = P.intersection(adj_lists[v])
            new_X = X.intersection(adj_lists[v])
            
            bron_kerbosch(new_R, new_P, new_X)
            
            P.remove(v)
            X.add(v)
            
    # Start with R={}, P=all vertices, X={}
    # Optimization: We can prune P to only vertices with degree > 0
    initial_P = frozenset(range(n))
    # Bron-Kerbosch on full set
    bron_kerbosch(frozenset(), initial_P, frozenset())
    
    return maximal_cliques

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency matrix and set
    adj = np.zeros((n, n), dtype=np.uint8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
        
    edges_set = set(edges)
    
    # 1. Enumerate Maximal Cliques
    # For n=64, this can be expensive, but random graphs have few maximal cliques.
    # We add a timeout check inside the recursion.
    maximal_cliques = _find_maximal_cliques(n, adj, edges_set)
    
    # If enumeration failed or took too long, fallback to simple edges
    if not maximal_cliques:
        return [[u, v] for u, v in edges]
        
    # Filter cliques: We only care about cliques that are subsets of edges in G.
    # By construction of Bron-Kerbosch on G, they are.
    # However, we need to ensure we can cover ALL edges.
    # The set of all maximal cliques covers all edges? 
    # Yes, every edge belongs to at least one maximal clique.
    
    # 2. Greedy Set Cover
    # We want to select a subset of maximal_cliques such that every edge in edges_set is covered.
    # "Covered" means the edge is inside the clique.
    # Minimize the number of cliques selected.
    
    # Precompute edges for each clique
    clique_edges = []
    for c in maximal_cliques:
        c_edges = set()
        verts = list(c)
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                u, v = verts[i], verts[j]
                if u > v: u, v = v, u
                # Check if edge exists (should always be true for maximal cliques of G)
                if (u, v) in edges_set:
                    c_edges.add((u, v))
        if c_edges:
            clique_edges.append(c_edges)
            
    # Map edge to index in edges list for fast tracking
    # We track uncovered edges
    uncovered_edges = set(edges)
    
    selected_cliques = []
    
    # Sort cliques by size descending initially? Or just iterate greedily.
    # Greedy: Pick clique that covers most uncovered edges.
    # To break ties, prefer larger cliques? Or smaller? 
    # Actually, covering more edges is the primary goal.
    
    # Optimization: Instead of iterating all cliques every time (O(C * E)),
    # we can maintain a count of how many uncovered edges each clique covers.
    # But updating this after each selection is O(E_clique).
    
    # Let's just do a simple greedy pass, but efficiently.
    # For n=64, E~1100. C (max cliques) is usually small (<100).
    # So O(C * E) per step is fine.
    
    while uncovered_edges and _deadline_ok():
        best_clique_idx = -1
        best_count = 0
        
        for i, c_edges in enumerate(clique_edges):
            if not c_edges: continue
            # Count how many edges in c_edges are in uncovered_edges
            # Intersection size
            count = len(c_edges.intersection(uncovered_edges))
            if count > best_count:
                best_count = count
                best_clique_idx = i
                
        if best_clique_idx == -1:
            # Should not happen if maximal cliques cover all edges
            # Fallback: pick an arbitrary uncovered edge as a clique of size 2
            e = next(iter(uncovered_edges))
            selected_cliques.append([e[0], e[1]])
            uncovered_edges.remove(e)
            continue
            
        # Select the best clique
        c_verts = maximal_cliques[best_clique_idx]
        selected_cliques.append(list(c_verts))
        
        # Remove covered edges from uncovered set
        c_edges = clique_edges[best_clique_idx]
        uncovered_edges -= c_edges
        
    # If we timed out, we might have remaining edges.
    # We must ensure a valid partition.
    # The problem requires EVERY edge to be in EXACTLY ONE clique.
    # My greedy set cover selects cliques. 
    # Wait, does greedy set cover ensure disjointness?
    # NO. Set cover allows overlapping sets.
    # If I select Clique A and Clique B, and they share an edge, that edge is covered twice.
    # The problem requires a PARTITION of edges.
    
    # So, standard Set Cover is NOT a valid solution directly because of the "exactly one" constraint.
    # However, if I select a set of cliques, I can "fix" the overlaps?
    # No, if I have a partition, I can't just add a clique that overlaps.
    
    # Let's rethink.
    # The "Greedy Peeling" (current) ensures disjointness by removing edges.
    # The "Set Cover" approach finds a set of cliques that covers all edges, but they might overlap.
    # If they overlap, it's invalid.
    
    # How to make Set Cover work for Edge Partition?
    # We can view it as: Find a set of cliques C1...Ck such that their edge sets are disjoint and union is E.
    # This is exactly the problem.
    
    # Greedy Set Cover with Disjointness Constraint:
    # 1. Start with all edges uncovered.
    # 2. Find a maximal clique K in the "remaining" graph?
    #    This is what the current code does.
    
    # Is there a better way?
    # What if we enumerate ALL cliques (not just maximal), and use a greedy algorithm that picks the clique covering the most uncovered edges, then REMOVES those edges?
    # This is effectively "Greedy Peeling" but with a better choice of clique.
    # Current code picks "first edge, expand to maximal".
    # This new approach picks "clique covering max uncovered edges".
    
    # Let's implement this "Max Coverage Greedy Peeling".
    # We need the list of all cliques? Or just maximal?
    # If we pick a maximal clique, it covers the most edges among its sub-cliques.
    # So considering only maximal cliques is sufficient for the "max coverage" step?
    # Yes, because any sub-clique of a maximal clique covers a subset of edges that the maximal clique covers.
    # So if a sub-clique is the best choice, the maximal clique containing it is at least as good (covers more).
    # BUT, picking the maximal clique might "waste" edges that could have been covered by other cliques?
    # No, because we remove edges.
    # If we pick Maximal Clique M, we remove E(M).
    # If we picked Sub-clique S, we would remove E(S).
    # Since E(S) subset E(M), picking M removes more edges.
    # Is it always better to remove more edges?
    # Yes, because we want to minimize the NUMBER of cliques.
    # Covering more edges with one clique is strictly better for the count, unless it prevents forming larger cliques later?
    # Wait.
    # Example: Triangle 1-2-3 and 1-2-4.
    # Edges: (1,2), (2,3), (1,3), (1,4), (2,4).
    # Max cliques: {1,2,3}, {1,2,4}.
    # If we pick {1,2,3}, we cover (1,2), (2,3), (1,3).
    # Remaining: (1,4), (2,4). These form clique {1,2,4}? No, (1,2) is gone.
    # Remaining edges form path 1-4-2. Not a clique of size 3.
    # We need 2 more cliques: {1,4}, {2,4}. Total 3 cliques.
    
    # If we picked {1,2,4} first, same result.
    
    # What if we picked {1,2}?
    # Cover (1,2).
    # Remaining: (2,3), (1,3), (1,4), (2,4).
    # Max cliques in remaining: {1,3}? No, (1,3) is edge. {2,4}? Edge.
    # Actually {1,3} and {2,4} are disjoint edges.
    # Can we form {1,3,4}? No (3,4) not edge.
    # So we pick {1,3}, {2,4}.
    # Total 3 cliques.
    
    # It seems picking maximal is good.
    
    # Let's implement "Greedy Peeling with Max Coverage from Maximal Cliques".
    
    # Re-initialize state
    uncovered_edges = set(edges)
    selected_cliques = []
    
    # Precomputed maximal cliques and their edges
    # clique_edges is list of sets of edges.
    
    # We need to be careful: As edges are removed, a "maximal clique" in the original graph might not be a clique in the remaining graph?
    # No, a clique in G is a clique in any subgraph of G.
    # But it might not be MAXIMAL in the remaining graph.
    # However, we only require it to be a CLIQUE.
    # And we require it to cover only UNCOVERED edges?
    # No, we remove the edges it covers.
    # So if we pick Clique C, we remove E(C) ∩ Uncovered.
    # But C must be a clique in the ORIGINAL graph (which it is).
    # Does C need to be a clique in the REMAINING graph?
    # The problem statement: "Every listed set must be a clique... in the graph".
    # It doesn't say "in the remaining graph". It says "in the graph".
    # So any clique in G is a valid candidate at any step.
    
    # So the algorithm is:
    # 1. Enumerate all maximal cliques in G.
    # 2. While uncovered edges exist:
    #    a. Find maximal clique C that maximizes |E(C) ∩ Uncovered|.
    #    b. If max is 0, break (shouldn't happen).
    #    c. Add C to partition.
    #    d. Remove E(C) from Uncovered.
    
    # This is structurally different from "Pick random edge, expand".
    # It is "Pick best global clique, remove its edges".
    
    # Complexity:
    # Enumerating max cliques: O(3^(n/3)) worst case, but fast for random graphs.
    # Greedy loop: O(E / avg_clique_size) iterations.
    # Each iteration: O(C * E) to find best clique.
    # C ~ 100, E ~ 1000. 100k ops per iteration.
    # Iterations ~ 1000 / 5 = 200.
    # Total ~ 20M ops. Very fast.
    
    # Let's restart the solve logic with this new strategy.
    
    uncovered_edges = set(edges)
    selected_cliques = []
    
    # We already have maximal_cliques and clique_edges from above.
    # But we need to handle the case where enumeration was slow or failed.
    # If maximal_cliques is empty, we fallback.
    
    if not maximal_cliques:
        return [[u, v] for u, v in edges]
        
    # Optimization: Sort cliques by size descending?
    # Not necessary for the greedy logic, but might help tie-breaking.
    
    while uncovered_edges and _deadline_ok():
        best_clique_idx = -1
        best_count = 0
        
        for i, c_edges in enumerate(clique_edges):
            if not c_edges: continue
            count = len(c_edges.intersection(uncovered_edges))
            if count > best_count:
                best_count = count
                best_clique_idx = i
                
        if best_clique_idx == -1:
            # Fallback
            e = next(iter(uncovered_edges))
            selected_cliques.append([e[0], e[1]])
            uncovered_edges.remove(e)
            continue
            
        c_verts = maximal_cliques[best_clique_idx]
        selected_cliques.append(list(c_verts))
        
        c_edges = clique_edges[best_clique_idx]
        uncovered_edges -= c_edges
        
    # Final check: if any edges remain (due to timeout), add them as edges
    if uncovered_edges:
        for e in uncovered_edges:
            selected_cliques.append([e[0], e[1]])
            
    return selected_cliques

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
# EVOLVE-BLOCK-END
