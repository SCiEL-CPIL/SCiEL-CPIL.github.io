"""Clique partition of split graphs using minimum set cover heuristic."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Build adjacency: clique vertices are 0..k-1, independent set is k..n-1
        # For clique vertices, they're all connected to each other.
        # For independent set vertices, they connect only to clique vertices (with probability p).
        
        # Represent edges as sets
        edge_set = set()
        for e in edges:
            edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        
        # Build adjacency for all vertices
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Identify cross-edges: edges between clique (0..k-1) and independent set (k..n-1)
        cross_edges = set()
        clique_edges = set()
        for (u, v) in edge_set:
            if u < k and v >= k:
                cross_edges.add((u, v))
            elif u >= k and v < k:
                cross_edges.add((v, u))
            else:
                # Both in clique or both in independent set
                # Independent set has no edges, so these must be clique edges
                if u < k and v < k:
                    clique_edges.add((u, v))
                # Note: No edges within independent set by definition of split graph
        
        # Candidate cliques are of the form {i} U S where i is in clique, S subset of N(i) in independent set
        # However, we must cover ALL edges, including clique edges.
        # A clique {i} U S covers:
        # 1. Edges (i, s) for s in S (cross edges)
        # 2. Edges (i, j) for j in clique \ {i} (clique edges)
        # 3. Edges (s1, s2) for s1, s2 in S -- but there are NO such edges in split graphs.
        # So each candidate clique covers a specific set of cross edges and a specific set of clique edges.
        
        # To minimize total cliques, we need to cover all cross edges and all clique edges.
        # Let's formulate as set cover.
        # Universe U = cross_edges U clique_edges
        # Candidate sets C = { C_i,S : i in [0,k), S subset of N_i }
        # This is too many candidates. We need a smarter heuristic.
        
        # Alternative view:
        # Every clique must contain at least one clique vertex (since independent set is empty of edges).
        # Let's say we assign each independent vertex v to one or more clique vertices.
        # Actually, a clique is defined by its intersection with the clique part (call it I) and independent part (call it S).
        # Since I must be a clique in the clique part (which is complete), any subset of clique vertices works.
        # But to cover cross edges (u, v), the clique must contain u and v.
        # Since v is independent, it can only be in cliques with clique vertices.
        
        # Let's use the Set Cover approach on Cross Edges primarily, as they are the bottleneck.
        # Clique edges can be covered by cliques that also cover cross edges, or by dedicated clique-only cliques.
        # However, a clique {i} U S covers clique edges (i, j) for all j in clique.
        
        # Let's generate candidate cliques.
        # For each clique vertex i, let N_i be the set of independent neighbors.
        # Any subset S of N_i forms a clique with i.
        # But we don't need to consider ALL subsets.
        # In a greedy set cover, we want cliques that cover many uncovered edges.
        
        # Let's try a different greedy strategy:
        # 1. Start with all cross edges and clique edges uncovered.
        # 2. Repeatedly pick a clique that covers the most uncovered edges.
        #    To do this efficiently, we can iterate over possible "centers" i (clique vertices).
        #    For a fixed i, the best S to choose is the set of all independent vertices v such that
        #    (i, v) is uncovered. Why? Because adding v to the clique covers edge (i, v) and doesn't
        #    introduce any new independent-independent edges (which don't exist).
        #    So for a fixed i, the maximal clique centered at i is {i} U N_i_uncovered.
        #    This clique covers:
        #    - All cross edges (i, v) for v in N_i_uncovered.
        #    - All clique edges (i, j) for j in [0, k) \ {i}.
        
        # So the candidate sets are simply: for each i in [0, k), C_i = {i} U {v in independent | (i,v) is uncovered}.
        # Wait, this is not quite right. If we pick C_i, we cover all edges incident to i that are currently uncovered.
        # But we might want to pick a subset S to save some edges for other cliques?
        # No, in set cover, if an edge is covered, it's covered. There's no penalty for covering "too much".
        # The only cost is the number of cliques.
        # So for a fixed i, the best choice is to include ALL possible independent neighbors that have uncovered edges to i.
        # This maximizes the number of cross edges covered by this clique.
        
        # What about clique edges?
        # A clique C_i covers all clique edges incident to i.
        # So if we pick C_i, we cover all edges (i, j) for j != i.
        
        # Algorithm:
        # 1. Uncovered = set of all edges.
        # 2. While Uncovered is not empty:
        #    a. For each i in [0, k), calculate the set of edges covered by C_i = {i} U N_i_uncovered.
        #       Edges covered:
        #       - (i, v) for v in N_i_uncovered
        #       - (i, j) for all j in [0, k) \ {i} such that (i, j) is in Uncovered
        #    b. Pick i* that maximizes the number of uncovered edges covered.
        #    c. Add C_i* to the list of cliques.
        #    d. Remove all edges covered by C_i* from Uncovered.
        #    e. If no i covers any edge, something is wrong (should not happen if Uncovered is not empty).
        
        # Let's implement this.
        
        uncovered = set(edge_set)
        cliques = []
        
        # Precompute neighbors for speed
        # adj[i] for i < k contains all neighbors.
        # For i < k, N_i = adj[i] & {k..n-1}
        
        while uncovered:
            if time.time() > deadline - 5:
                # Fallback: just list each edge as a clique
                for (u, v) in uncovered:
                    cliques.append([u, v])
                break
            
            best_i = -1
            best_count = -1
            best_clique = None
            
            for i in range(k):
                # Candidate clique centered at i
                # Includes i and all independent neighbors v such that (i, v) is uncovered
                S = set()
                for v in adj[i]:
                    if v >= k:
                        if (i, v) in uncovered:
                            S.add(v)
                
                # Count uncovered edges covered by this clique
                count = 0
                # Cross edges
                for v in S:
                    # Edge (i, v) is covered
                    count += 1
                
                # Clique edges
                # All edges (i, j) for j in [0, k) \ {i}
                for j in range(k):
                    if j == i:
                        continue
                    if (i, j) in uncovered:
                        count += 1
                
                if count > best_count:
                    best_count = count
                    best_i = i
                    best_clique = [i] + list(S)
            
            if best_i == -1 or best_count == 0:
                # Fallback: take any edge
                u, v = next(iter(uncovered))
                cliques.append([u, v])
                uncovered.remove((u, v))
                # Also remove if reversed
                if (v, u) in uncovered:
                    uncovered.remove((v, u))
            else:
                cliques.append(best_clique)
                # Remove covered edges
                clique_set = set(best_clique)
                new_uncovered = set()
                for (u, v) in uncovered:
                    if u in clique_set and v in clique_set:
                        continue
                    new_uncovered.add((u, v))
                uncovered = new_uncovered
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 15:
            # Emergency fallback
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue
        
        edges = instance(*spec)
        cliques = partition_split_graph(spec[0], spec[1], edges)
        
        # Convert to lists of Python ints
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
        
        results.append(final_cliques)
    
    return results
# EVOLVE-BLOCK-END
