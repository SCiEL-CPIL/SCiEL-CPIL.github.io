"""Rewritten program for clique partition of split graphs."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, p, seed):
        """
        Structure-aware clique partition for split graphs.
        """
        edges = instance(n, k, p, seed)
        
        # Build adjacency for cross edges
        # N[i] = set of C-vertices adjacent to I-vertex i
        N = {i: set() for i in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)
        
        # C-edges
        C_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                C_edges.add((u, v))
        
        # Track which C-edges are covered by cross-cliques
        covered_C_edges = set()
        cliques = []
        
        # Process I-vertices in order of decreasing |N(i)|
        I_vertices = sorted(range(k, n), key=lambda i: len(N[i]), reverse=True)
        
        for i in I_vertices:
            if time.time() > deadline:
                break
            if not N[i]:
                continue
            
            # Greedily partition N[i] into groups, avoiding conflicts with
            # already-covered C-edges
            remaining_N = set(N[i])
            
            while remaining_N:
                if time.time() > deadline:
                    break
                
                # Find the largest subset S of remaining_N such that no edge
                # within S is already covered by another cross-clique
                S = set()
                # Sort to ensure determinism
                for u in sorted(remaining_N):
                    can_add = True
                    for v in S:
                        e = (min(u, v), max(u, v))
                        if e in covered_C_edges:
                            can_add = False
                            break
                    if can_add:
                        S.add(u)
                
                if not S:
                    # Shouldn't happen if remaining_N is non-empty, but just in case
                    S = {next(iter(remaining_N))}
                
                # Create clique {i} ∪ S
                clique = sorted([i] + list(S))
                cliques.append(clique)
                
                # Mark C-edges within S as covered
                S_list = sorted(S)
                for a_idx in range(len(S_list)):
                    for b_idx in range(a_idx + 1, len(S_list)):
                        e = (S_list[a_idx], S_list[b_idx])
                        covered_C_edges.add(e)
                
                remaining_N -= S
        
        # Now cover remaining C-edges with C-only cliques
        remaining_C_edges = C_edges - covered_C_edges
        
        if remaining_C_edges:
            # Build adjacency for remaining C-edges
            remaining_adj = defaultdict(set)
            for u, v in remaining_C_edges:
                remaining_adj[u].add(v)
                remaining_adj[v].add(u)
            
            while remaining_C_edges:
                if time.time() > deadline:
                    break
                
                # Find a maximal clique in the remaining graph
                # Pick a vertex with highest degree
                if not remaining_adj:
                    break
                
                # Find vertex with max degree in remaining graph
                max_deg = 0
                start_v = None
                for v in remaining_adj:
                    if len(remaining_adj[v]) > max_deg:
                        max_deg = len(remaining_adj[v])
                        start_v = v
                
                if start_v is None:
                    break
                
                # Greedily expand clique starting from start_v
                current_clique = {start_v}
                candidates = set(remaining_adj[start_v])
                
                while candidates:
                    # Pick a candidate that is connected to all in current_clique
                    # Since it's a subgraph of a clique, we just need to check
                    # if it's connected to all in current_clique in the remaining graph
                    # Actually, if we maintain the property that current_clique is a clique
                    # in the remaining graph, we can just pick any candidate
                    # that is adjacent to all vertices in current_clique.
                    
                    valid_candidate = None
                    for c in candidates:
                        # Check if c is adjacent to all in current_clique
                        is_adj_to_all = True
                        for v in current_clique:
                            if v not in remaining_adj[c]:
                                is_adj_to_all = False
                                break
                        if is_adj_to_all:
                            valid_candidate = c
                            break
                    
                    if valid_candidate is None:
                        break
                    
                    current_clique.add(valid_candidate)
                    candidates.discard(valid_candidate)
                    
                    # Update candidates: remove those not adjacent to valid_candidate
                    # Actually, we can just re-evaluate or filter
                    # For efficiency, filter candidates
                    new_candidates = set()
                    for c in candidates:
                        if c in remaining_adj[valid_candidate]:
                            # Also must be adjacent to all others in current_clique?
                            # Since c was in candidates, it was adjacent to start_v.
                            # We need it to be adjacent to all in current_clique.
                            # The check in the loop above handles this for the next iteration.
                            # So we can just keep it for now, or filter now.
                            # Filtering now is safer.
                            is_adj_to_all = True
                            for v in current_clique:
                                if v not in remaining_adj[c]:
                                    is_adj_to_all = False
                                    break
                            if is_adj_to_all:
                                new_candidates.add(c)
                    candidates = new_candidates
                
                # Add the clique
                if len(current_clique) >= 2:
                    clique = sorted(current_clique)
                    cliques.append(clique)
                    
                    # Remove edges in this clique from remaining_C_edges
                    clique_list = sorted(current_clique)
                    for a_idx in range(len(clique_list)):
                        for b_idx in range(a_idx + 1, len(clique_list)):
                            e = (clique_list[a_idx], clique_list[b_idx])
                            if e in remaining_C_edges:
                                remaining_C_edges.discard(e)
                                # Update adj
                                remaining_adj[clique_list[a_idx]].discard(clique_list[b_idx])
                                remaining_adj[clique_list[b_idx]].discard(clique_list[a_idx])
                    
                    # Clean up empty adj entries
                    for v in list(remaining_adj.keys()):
                        if not remaining_adj[v]:
                            del remaining_adj[v]
                else:
                    # Fallback: if clique size < 2, just take one edge
                    # This shouldn't happen if remaining_C_edges is not empty
                    e = next(iter(remaining_C_edges))
                    cliques.append(list(e))
                    remaining_C_edges.discard(e)
                    remaining_adj[e[0]].discard(e[1])
                    remaining_adj[e[1]].discard(e[0])
                    if not remaining_adj[e[0]]:
                        del remaining_adj[e[0]]
                    if not remaining_adj[e[1]]:
                        del remaining_adj[e[1]]

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Not enough time, use seed program result as fallback
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = partition_split_graph(*spec)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
