"""Clique partitions of three fixed split (chordal) graphs."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = t0 + 110.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Identify cross edges
        cross_edges = []
        for u in range(k):
            for v in range(k, n):
                if (u, v) in edges:
                    cross_edges.append((u, v))
        
        # Compute neighborhoods of I vertices in C
        neigh = defaultdict(set)
        for v in range(k, n):
            for u in range(k):
                if (u, v) in edges:
                    neigh[v].add(u)
        
        # Build graph H of coverable internal edges
        # Edge (u, u') in H if there exists v in I such that u, u' in N(v)
        # We need a maximum matching in H.
        # H vertices are 0..k-1
        # H edges are pairs (u, u')
        
        # For matching, we can use a simple greedy or Hopcroft-Karp.
        # Given k <= 30, a simple DFS-based augmenting path algorithm is fast enough.
        
        # First, determine which internal edges exist in H
        # And for each, store a witness v
        coverable_edges = {} # (u, u') -> v
        for v in range(k, n):
            ns = list(neigh[v])
            for i in range(len(ns)):
                for j in range(i+1, len(ns)):
                    u1 = ns[i]
                    u2 = ns[j]
                    if u1 > u2:
                        u1, u2 = u2, u1
                    # Only store the first witness, or any witness
                    # We just need to know it's coverable
                    if (u1, u2) not in coverable_edges:
                        coverable_edges[(u1, u2)] = v
        
        # Build adjacency list for H
        adj_H = defaultdict(list)
        for (u1, u2), v in coverable_edges.items():
            adj_H[u1].append(u2)
            adj_H[u2].append(u1)
            
        # Maximum Bipartite Matching? No, H is not necessarily bipartite.
        # H is a graph on k vertices. We need Maximum Matching in a general graph.
        # For k=30, we can use a simple recursive backtracking or a library.
        # Since no external libs for matching are guaranteed (networkx not mentioned, only stdlib/num/scipy),
        # and scipy doesn't have general graph matching directly in easy-to-use API for this size,
        # we can use a simple greedy + local improvement or just greedy.
        # Greedy matching is O(V^2).
        # Let's try a simple greedy matching first. It's not guaranteed optimal but likely good.
        # Or we can implement a basic augmenting path for general graphs (Blossom algorithm is complex).
        # Given the small size, a simple randomized restart greedy might work.
        # Or just standard greedy: sort edges, pick if both vertices free.
        
        # Let's use a simple deterministic greedy:
        # Iterate through all possible edges in H (sorted for determinism)
        # If both endpoints are unmatched, match them.
        
        matched = [False] * k
        matching = []
        
        # Sort edges by some heuristic? Or just iterate.
        # To maximize matching, we should prefer edges that are "rare"?
        # For now, simple iteration.
        for u in range(k):
            if matched[u]:
                continue
            for v in adj_H[u]:
                if v > u and not matched[v]:
                    matching.append((u, v))
                    matched[u] = True
                    matched[v] = True
                    break
        
        # Create cliques from matching
        cliques = []
        covered_cross = set()
        
        for u, v in matching:
            # Find a witness w for (u, v)
            w = coverable_edges.get((u, v))
            if w is None:
                # Should not happen
                continue
            cliques.append([u, v, w])
            # Mark cross edges (u, w) and (v, w) as covered
            covered_cross.add((u, w))
            covered_cross.add((v, w))
            
        # Add remaining cross edges as size-2 cliques
        for u, v in cross_edges:
            if (u, v) not in covered_cross:
                cliques.append([u, v])
                
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            break
        cliques = partition_cliques(*spec)
        # Ensure types and validity
        cleaned = []
        for c in cliques:
            cl = [int(x) for x in c]
            if len(cl) >= 2 and len(set(cl)) == len(cl):
                cleaned.append(cl)
        result.append(cleaned)

    return result
# EVOLVE-BLOCK-END
