"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency for K and I
        # K is 0..k-1, I is k..n-1
        # K is a clique, so all pairs in K are edges.
        # I is independent, so no pairs in I are edges.
        
        # Cross edges
        cross_adj = defaultdict(set) # v in I -> set of u in K
        for u, v in edges:
            if u < k and v >= k:
                cross_adj[v].add(u)
        
        # Initial available K-edges
        all_k_edges = set((u, w) for u in range(k) for w in range(u+1, k))
        
        best_cliques = None
        best_count = float('inf')
        
        # Try multiple randomized orders
        i_vertices = list(range(k, n))
        
        # We will run several trials
        # Number of trials depends on time available
        # Let's say 500 trials per instance if time permits
        
        for trial in range(500):
            if time.time() > deadline - 5:
                break
                
            current_k_edges = set(all_k_edges)
            cliques = []
            
            # Shuffle I vertices
            random.shuffle(i_vertices)
            
            covered_cross = set()
            
            for v in i_vertices:
                nv = cross_adj[v]
                if not nv:
                    continue
                
                # Find maximal clique in H_v (subgraph of K induced by nv, using current_k_edges)
                # Greedy: pick vertex with max degree in H_v
                local_deg = defaultdict(int)
                for u in nv:
                    for w in nv:
                        if u < w and (u, w) in current_k_edges:
                            local_deg[u] += 1
                            local_deg[w] += 1
                
                if not local_deg:
                    # No K-edges available in nv
                    # Cover cross edges as 2-cliques
                    for u in nv:
                        cliques.append([u, v])
                        covered_cross.add((u, v))
                    continue
                
                max_v = max(local_deg, key=local_deg.get)
                clique_v = [max_v]
                candidates = [u for u in nv if u != max_v]
                candidates.sort(key=lambda x: local_deg.get(x, 0), reverse=True)
                
                for cand in candidates:
                    ok = True
                    for cv in clique_v:
                        if (min(cv, cand), max(cv, cand)) not in current_k_edges:
                            ok = False
                            break
                    if ok:
                        clique_v.append(cand)
                
                if len(clique_v) >= 2:
                    cliques.append(sorted(clique_v + [v]))
                    for u in clique_v:
                        covered_cross.add((u, v))
                    # Remove K-edges
                    for i in range(len(clique_v)):
                        for j in range(i + 1, len(clique_v)):
                            u, w = clique_v[i], clique_v[j]
                            e = (min(u, w), max(u, w))
                            if e in current_k_edges:
                                current_k_edges.remove(e)
                else:
                    # len(clique_v) == 1
                    # This covers 1 cross edge and 0 K-edges
                    # Equivalent to 2-clique
                    cliques.append([clique_v[0], v])
                    covered_cross.add((clique_v[0], v))
            
            # Cover remaining K-edges
            # Greedy clique partition
            rem_k_edges = current_k_edges
            while rem_k_edges:
                if time.time() > deadline - 2:
                    # Fallback
                    for e in rem_k_edges:
                        cliques.append([e[0], e[1]])
                    rem_k_edges.clear()
                    break
                    
                # Find maximal clique
                degrees = defaultdict(int)
                for u, w in rem_k_edges:
                    degrees[u] += 1
                    degrees[w] += 1
                
                if not degrees:
                    break
                    
                max_v = max(degrees, key=degrees.get)
                neighbors = set()
                for u, w in rem_k_edges:
                    if u == max_v:
                        neighbors.add(w)
                    elif w == max_v:
                        neighbors.add(u)
                
                candidates = list(neighbors)
                candidates.sort(key=lambda x: degrees.get(x, 0), reverse=True)
                
                clique = [max_v]
                for cand in candidates:
                    ok = True
                    for cv in clique:
                        if (min(cv, cand), max(cv, cand)) not in rem_k_edges:
                            ok = False
                            break
                    if ok:
                        clique.append(cand)
                
                cliques.append(sorted(clique))
                
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, w = clique[i], clique[j]
                        e = (min(u, w), max(u, w))
                        if e in rem_k_edges:
                            rem_k_edges.remove(e)
            
            # Cover remaining cross edges
            for v in i_vertices:
                for u in cross_adj[v]:
                    if (u, v) not in covered_cross:
                        cliques.append([u, v])
            
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = [list(c) for c in cliques]
        
        # Clean up types
        clean_cliques = []
        for c in best_cliques:
            clean_cliques.append([int(x) for x in c])
        
        return clean_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
