import time
from random import Random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_one(n, k, p, seed):
        # Build adjacency
        adj = [set() for _ in range(n)]
        cross = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if (u, v) in set(instance(n, k, p, seed)):
                    adj[u].add(v)
                    adj[v].add(u)
                    cross[u].add(v)
                    cross[v].add(u)
        # Add K clique edges
        for u in range(k):
            for v in range(u + 1, k):
                adj[u].add(v)
                adj[v].add(u)

        # For each I-vertex v, we want to form as many size-3 cliques as possible
        # Each size-3 clique is {v, u1, u2} where u1, u2 in N_K(v) and u1-u2 is an edge in K
        # Since K is complete, any pair in N_K(v) works.
        # We want to partition N_K(v) into pairs (each becomes a 3-clique with v)
        # and leftovers (each becomes a 2-clique with v).
        # But we also want to "use up" K-edges efficiently.
        # Actually, the K-edges inside N_K(v) don't need to be explicitly listed as K-edges
        # if they're covered by the 3-clique.
        
        # Strategy: For each v in I, greedily pair up neighbors in K.
        # This uses up pairs of K-vertices, creating 3-cliques.
        # The remaining K-edges that are NOT covered by any 3-clique need to be covered separately.
        
        # Let's think about which K-edges are "covered" by a 3-clique.
        # A 3-clique {v, u1, u2} covers edge (u1, u2).
        # So we want to cover as many K-edges as possible with these 3-cliques.
        
        # For each v, N_K(v) is a subset of K. The number of pairs we can form is floor(|N_K(v)|/2).
        # Each pair covers one K-edge.
        
        # We need to assign K-edges to be covered by specific v's.
        # This is a matching-like problem on the bipartite graph between I-vertices and K-edges,
        # where v is connected to edge (u1,u2) if both u1,u2 in N_K(v).
        
        # Greedy approach:
        # 1. For each v, compute N_K(v).
        # 2. Sort I-vertices by |N_K(v)| descending.
        # 3. For each v, greedily pick unused K-edges where both endpoints are in N_K(v).
        #    Prefer edges that are "rare" (few other v's can cover them).
        
        # Simpler: just pair up neighbors arbitrarily, and track which K-edges are used.
        # Then cover remaining K-edges with additional cliques.
        
        cliques = []
        used_k_edges = set()
        
        # Process I-vertices
        for v in range(k, n):
            nk = sorted(cross[v])
            # Pair up
            i = 0
            while i + 1 < len(nk):
                u1, u2 = nk[i], nk[i+1]
                edge = (min(u1,u2), max(u1,u2))
                if edge not in used_k_edges:
                    cliques.append([v, u1, u2])
                    used_k_edges.add(edge)
                i += 2
            # Leftover singletons
            if i < len(nk):
                cliques.append([v, nk[i]])
        
        # Now cover remaining K-edges
        # Build list of uncovered K-edges
        uncovered = []
        for u in range(k):
            for v in range(u+1, k):
                if (u, v) not in used_k_edges:
                    uncovered.append((u, v))
        
        # Cover uncovered edges with cliques.
        # Greedy: repeatedly find largest clique of uncovered edges.
        # Since these are edges in K (which is complete), any set of edges that form a clique
        # in the line graph... actually we just need sets of edges that are vertex-disjoint?
        # No, a clique in the original graph means a set of vertices that are all pairwise adjacent.
        # The edges in our clique must all be within that vertex set.
        # So if we pick a vertex set S, we can cover all edges within S at once.
        # We want to cover all uncovered edges with minimum number of vertex sets (cliques).
        
        # This is equivalent to: cover the uncovered edges with minimum number of cliques.
        # Greedy: find the largest set of vertices whose internal edges are all uncovered.
        
        remaining = set(uncovered)
        while remaining:
            # Find the densest small clique
            # Try all pairs, triples, etc.
            best_clique = None
            best_count = 0
            # Greedy: pick a vertex, then try to extend
            # Simpler: just pick the edge with max degree, then extend
            # Even simpler: random sampling of subsets
            
            # Let's just greedily build a clique:
            # Start with a random uncovered edge
            # Try to add vertices that are connected to all current vertices
            import random
            rng2 = random.Random(42)
            
            # Find vertex with max degree in remaining
            deg = defaultdict(int)
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            
            if not deg:
                break
                
            start_v = max(deg, key=lambda x: deg[x])
            current_clique = [start_v]
            
            # Try to add vertices
            changed = True
            while changed:
                changed = False
                candidates = [u for u in range(k) if u not in current_clique]
                for c in candidates:
                    # Check if c is connected to all in current_clique via remaining edges
                    ok = True
                    for u in current_clique:
                        e = (min(c, u), max(c, u))
                        if e not in remaining:
                            ok = False
                            break
                    if ok:
                        current_clique.append(c)
                        changed = True
                        break
            
            # Now remove all edges within current_clique from remaining
            for u in range(len(current_clique)):
                for v in range(u+1, len(current_clique)):
                    e = (min(current_clique[u], current_clique[v]), max(current_clique[u], current_clique[v]))
                    if e in remaining:
                        remaining.discard(e)
            
            if len(current_clique) >= 2:
                cliques.append(current_clique)
            
            if time.time() > deadline:
                # Emergency: add remaining edges as 2-cliques
                for u, v in remaining:
                    cliques.append([u, v])
                break
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for n, k, p, seed in specs:
        if time.time() > deadline:
            # Should not happen for these small instances
            break
        cliques = solve_one(n, k, p, seed)
        results.append(cliques)
    
    return results
