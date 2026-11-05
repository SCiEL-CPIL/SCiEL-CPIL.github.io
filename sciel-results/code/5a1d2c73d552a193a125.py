"""Rewritten program for clique partition of dense random graphs."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def greedy_clique_partition(edges, adj, n, deadline):
        """
        Greedy set-cover style approach:
        1. Generate candidate cliques using a fast heuristic (greedy expansion from high-degree vertices).
        2. Repeatedly select the clique that covers the most uncovered edges.
        3. Remove covered edges and repeat until all edges are covered.
        """
        remaining_edges = set(edges)
        cliques = []
        
        if not remaining_edges:
            return cliques

        # Precompute vertex degrees in the original graph
        degrees = [len(adj[v]) for v in range(n)]
        
        # Generate candidate cliques
        # We'll generate candidates by starting from each vertex and greedily expanding
        candidate_cliques = []
        
        # Sort vertices by degree (descending) to prioritize high-degree vertices
        vertices_sorted = sorted(range(n), key=lambda v: degrees[v], reverse=True)
        
        for start_v in vertices_sorted:
            if time.time() > deadline:
                break
            
            # Greedy expansion from start_v
            clique = {start_v}
            # Candidates that can be added: neighbors of all vertices in clique
            candidates = set(adj[start_v])
            
            while True:
                if time.time() > deadline:
                    break
                
                # Filter candidates: all edges from candidate to all clique members must be in remaining_edges
                valid_candidates = []
                for c in candidates:
                    valid = True
                    for m in clique:
                        e = (min(c, m), max(c, m))
                        if e not in remaining_edges:
                            valid = False
                            break
                    if valid:
                        valid_candidates.append(c)
                
                if not valid_candidates:
                    break
                
                # Pick the candidate that maximizes the number of edges it shares with other valid candidates
                best_cand = None
                best_cand_score = -1
                for c in valid_candidates:
                    score = 0
                    for other in valid_candidates:
                        if other != c:
                            e = (min(c, other), max(c, other))
                            if e in remaining_edges:
                                score += 1
                    if score > best_cand_score:
                        best_cand_score = score
                        best_cand = c
                
                if best_cand is not None:
                    clique.add(best_cand)
                    # Update candidates: intersection with neighbors of best_cand
                    candidates = candidates & adj[best_cand]
                    candidates -= clique
                else:
                    break
            
            if len(clique) >= 2:
                candidate_cliques.append(frozenset(clique))
        
        # Now do greedy set cover: repeatedly pick the clique covering most uncovered edges
        # To speed this up, we can limit the number of candidate cliques we consider
        # and use a heap or sorted list
        
        # For efficiency, let's just iterate and pick the best each time
        # This is O(|candidates| * |edges_in_clique|) per iteration, which could be slow
        
        # Alternative: just use the candidate cliques in order of size (larger first)
        # and pick the first one that covers at least one uncovered edge
        
        # Sort candidates by size descending
        candidate_cliques.sort(key=len, reverse=True)
        
        # Remove duplicates
        seen = set()
        unique_candidates = []
        for c in candidate_cliques:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        
        # Greedy selection
        while remaining_edges:
            if time.time() > deadline:
                break
            
            best_clique = None
            best_coverage = 0
            
            # Try each candidate clique and count how many uncovered edges it covers
            for clique in unique_candidates:
                if len(clique) < 2:
                    continue
                
                # Count uncovered edges in this clique
                coverage = 0
                clique_list = sorted(clique)
                for a, b in combinations(clique_list, 2):
                    if (a, b) in remaining_edges:
                        coverage += 1
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_clique = clique
                
                # Early termination if we found a clique covering all remaining edges
                if best_coverage >= len(remaining_edges):
                    break
            
            if best_clique is None or best_coverage == 0:
                # No candidate clique covers any remaining edge; take a single edge
                e = next(iter(remaining_edges))
                clique = frozenset(e)
                cliques.append(sorted(clique))
                remaining_edges.discard(e)
                continue
            
            # Add this clique to the partition
            cliques.append(sorted(best_clique))
            
            # Remove all edges in this clique from remaining
            clique_list = sorted(best_clique)
            for a, b in combinations(clique_list, 2):
                remaining_edges.discard((a, b))
        
        # Handle any remaining edges
        for e in remaining_edges:
            cliques.append(sorted(e))
        
        return cliques

    def solve_instance(n, p, seed):
        edges, adj = random_graph(n, p, seed)
        return greedy_clique_partition(edges, adj, n, deadline)

    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
