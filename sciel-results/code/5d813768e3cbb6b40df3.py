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
            
            # Use Bron-Kerbosch with pivot to find maximum clique
            best_clique = set()
            
            def bronk(R, P, X):
                nonlocal best_clique
                if time.time() > deadline:
                    return
                if not P and not X:
                    if len(R) > len(best_clique):
                        best_clique = set(R)
                    return
                # Choose pivot u from P ∪ X to minimize branching
                if P:
                    u = max(P, key=lambda v: len(P & remaining_adj[v]) + len(X & remaining_adj[v]))
                elif X:
                    u = max(X, key=lambda v: len(P & remaining_adj[v]) + len(X & remaining_adj[v]))
                else:
                    return
                
                # Branch on P \ N(u)
                for v in list(P - remaining_adj[u]):
                    bronk(R | {v}, P & remaining_adj[v], X & remaining_adj[v])
                    P.discard(v)
                    X.add(v)
            
            # Find maximum clique in remaining graph
            all_vertices = set(remaining_adj.keys())
            bronk(set(), set(all_vertices), set())
            
            # Process found cliques
            while remaining_C_edges:
                if time.time() > deadline:
                    break
                
                # Rebuild adj from remaining edges
                remaining_adj = defaultdict(set)
                for u, v in remaining_C_edges:
                    remaining_adj[u].add(v)
                    remaining_adj[v].add(u)
                
                # Find maximum clique
                best_clique = set()
                all_vertices = set(remaining_adj.keys())
                if not all_vertices:
                    break
                
                # Use Bron-Kerbosch with pivot
                def bronk_local(R, P, X):
                    nonlocal best_clique
                    if time.time() > deadline:
                        return
                    if not P and not X:
                        if len(R) > len(best_clique):
                            best_clique = set(R)
                        return
                    if P:
                        u = max(P, key=lambda v: len(P & remaining_adj[v]) + len(X & remaining_adj[v]))
                    elif X:
                        u = max(X, key=lambda v: len(P & remaining_adj[v]) + len(X & remaining_adj[v]))
                    else:
                        return
                    
                    for v in list(P - remaining_adj[u]):
                        bronk_local(R | {v}, P & remaining_adj[v], X & remaining_adj[v])
                        P.discard(v)
                        X.add(v)
                
                bronk_local(set(), set(all_vertices), set())
                
                if len(best_clique) < 2:
                    # Fallback: take one edge
                    e = next(iter(remaining_C_edges))
                    cliques.append(sorted([e[0], e[1]]))
                    remaining_C_edges.discard(e)
                    continue
                
                clique = sorted(best_clique)
                cliques.append(clique)
                
                # Remove edges in this clique from remaining_C_edges
                for a_idx in range(len(clique)):
                    for b_idx in range(a_idx + 1, len(clique)):
                        e = (clique[a_idx], clique[b_idx])
                        if e in remaining_C_edges:
                            remaining_C_edges.discard(e)

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
