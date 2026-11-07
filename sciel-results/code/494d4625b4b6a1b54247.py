import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(n, k, p, seed):
        rng = random.Random(seed)
        # Build adjacency
        # K: 0..k-1, I: k..n-1
        adj_I = [set() for _ in range(n - k)]  # adj_I[i] is set of K neighbors of vertex k+i
        
        # Generate cross edges
        for i in range(n - k):
            v = k + i
            for u in range(k):
                if rng.random() < p:
                    adj_I[i].add(u)
        
        # Note: K is a clique, so all K-K edges exist.
        
        used_K_edges = set()
        cliques = []
        
        # Process I vertices in order of decreasing degree
        i_indices = sorted(range(n - k), key=lambda i: len(adj_I[i]), reverse=True)
        
        for i in i_indices:
            v = k + i
            U = set(adj_I[i])
            while U:
                # Find largest subset S of U such that all pairs in S are not in used_K_edges
                # Greedy: build S by adding vertices that don't conflict with current S
                S = set()
                candidates = list(U)
                # Sort candidates by degree in the "available" graph (higher degree first)
                def avail_deg(u):
                    count = 0
                    for w in candidates:
                        if u == w:
                            continue
                        if (min(u, w), max(u, w)) not in used_K_edges:
                            count += 1
                    return count
                
                candidates.sort(key=avail_deg, reverse=True)
                
                for u in candidates:
                    if u not in U:
                        continue
                    # Check if u can be added to S
                    can_add = True
                    for w in S:
                        if (min(u, w), max(u, w)) in used_K_edges:
                            can_add = False
                            break
                    if can_add:
                        S.add(u)
                
                if S:
                    # Add clique {v} U S
                    clique = [v] + list(S)
                    cliques.append(clique)
                    # Update used_K_edges
                    for u, w in combinations(S, 2):
                        used_K_edges.add((min(u, w), max(u, w)))
                    U -= S
                else:
                    # Should not happen, as single vertex is always valid
                    u = next(iter(U))
                    clique = [v, u]
                    cliques.append(clique)
                    U.remove(u)
        
        # Cover remaining K edges
        # Build list of remaining edges
        rem_edges = []
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in used_K_edges:
                    rem_edges.append((u, w))
        
        if rem_edges:
            # Graph G_rem on K
            # Find min edge clique cover
            # Greedy: repeatedly find max clique in G_rem, add it, remove edges.
            # Represent G_rem by adjacency set
            adj_rem = [set() for _ in range(k)]
            for u, w in rem_edges:
                adj_rem[u].add(w)
                adj_rem[w].add(u)
            
            remaining = set(rem_edges)
            
            while remaining:
                # Find max clique in G_rem
                # Vertices involved in remaining edges
                verts = set()
                for u, w in remaining:
                    verts.add(u)
                    verts.add(w)
                
                # Greedy max clique: pick vertex with max degree, restrict to neighbors
                C = set()
                candidates = set(verts)
                
                while candidates:
                    best_v = None
                    max_deg = -1
                    for v in candidates:
                        deg = len(adj_rem[v] & candidates)
                        if deg > max_deg:
                            max_deg = deg
                            best_v = v
                    
                    if best_v is None:
                        break
                    
                    # Check compatibility with C
                    can_add = True
                    for u in C:
                        if u not in adj_rem[best_v]:
                            can_add = False
                            break
                    
                    if can_add:
                        C.add(best_v)
                        # Restrict candidates to neighbors of best_v
                        candidates = candidates & adj_rem[best_v]
                    else:
                        candidates.remove(best_v)
                
                if not C:
                    break
                    
                clique = list(C)
                cliques.append(clique)
                
                # Remove edges in C from remaining
                for u, w in combinations(C, 2):
                    if (min(u, w), max(u, w)) in remaining:
                        remaining.remove((min(u, w), max(u, w)))
                
                # Update adj_rem to remove edges in C
                for u in C:
                    for w in C:
                        if u != w:
                            if w in adj_rem[u]:
                                adj_rem[u].remove(w)
                                adj_rem[w].remove(u)

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline:
            break
        cliques = clique_partition(n, k, p, seed)
        results.append(cliques)

    while len(results) < 3:
        results.append([])

    return results
