"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin under 150s limit
    
    # Use bitset representation for fast C4 checking
    # G is C4-free iff for every pair of non-adjacent vertices (u,v) in G,
    # |N_G(u) ∩ N_G(v)| <= 1
    # Equivalently, in the complement H: for every edge (u,v) in H,
    # |N_G(u) ∩ N_G(v)| <= 1, where N_G(x) = all vertices except x and N_H(x)
    
    # Strategy: Start from a known C4-free structure and use local search
    # to maximize n while maintaining min degree >= n-39
    
    best_result = {'n': 39, 'edges': []}
    
    # Try n=46 first with multiple restarts using different seeds
    for target_n in [46, 45, 44]:
        if time.time() > deadline:
            break
        min_deg_req = target_n - 39
        
        for seed in range(20):
            if time.time() > deadline:
                break
            
            random.seed(seed * 1000 + target_n)
            
            # Represent adjacency as list of sets for fast intersection
            # adj[v] = set of neighbors of v in G
            adj = [set() for _ in range(target_n)]
            
            # Phase 1: Build initial graph by adding edges greedily
            # Prioritize edges that help low-degree vertices
            for _ in range(5000):
                if time.time() > deadline:
                    break
                # Pick a random pair, prefer pairs involving low-degree vertices
                degs = [len(adj[v]) for v in range(target_n)]
                # Weight by (min_deg_req - deg[v]) for vertices below target
                weights = []
                for v in range(target_n):
                    w = max(1, min_deg_req - degs[v] + 1)
                    weights.append(w)
                
                # Pick two vertices weighted by need
                v = random.choices(range(target_n), weights=weights)[0]
                u = random.choices(range(target_n), weights=weights)[0]
                
                if u == v or u in adj[v]:
                    continue
                
                # Check if adding (u,v) creates a C4
                # C4 exists if |N(u) ∩ N(v)| >= 2
                common = adj[u] & adj[v]
                if len(common) < 2:
                    adj[u].add(v)
                    adj[v].add(u)
            
            # Phase 2: Local search - try to increase min degree
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                degs = [len(adj[v]) for v in range(target_n)]
                if min(degs) >= min_deg_req:
                    break
                
                # Find a low-degree vertex
                low_v = min(range(target_n), key=lambda v: degs[v])
                if degs[low_v] >= min_deg_req:
                    break
                
                # Try to add an edge from low_v to some non-neighbor
                candidates = [w for w in range(target_n) if w != low_v and w not in adj[low_v]]
                random.shuffle(candidates)
                
                improved = False
                for w in candidates:
                    common = adj[low_v] & adj[w]
                    if len(common) < 2:
                        adj[low_v].add(w)
                        adj[w].add(low_v)
                        degs[low_v] += 1
                        degs[w] += 1
                        improved = True
                        break
                
                if not improved:
                    # Try removing an edge that's not helping and adding a different one
                    # Find an edge (a,b) where both have degree > min_deg_req
                    # and removing it allows adding two new edges
                    removable = [(a, b) for a in range(target_n) for b in adj[a] 
                                if a < b and degs[a] > min_deg_req and degs[b] > min_deg_req]
                    if removable:
                        a, b = random.choice(removable)
                        # Remove edge
                        adj[a].remove(b)
                        adj[b].remove(a)
                        degs[a] -= 1
                        degs[b] -= 1
                        
                        # Try to add edges from a and b to low-degree vertices
                        for x in [a, b]:
                            if degs[x] >= min_deg_req:
                                continue
                            cands = [w for w in range(target_n) if w != x and w not in adj[x]]
                            random.shuffle(cands)
                            for w in cands:
                                common = adj[x] & adj[w]
                                if len(common) < 2:
                                    adj[x].add(w)
                                    adj[w].add(x)
                                    degs[x] += 1
                                    degs[w] += 1
                                    break
            
            # Verify and check if valid
            degs = [len(adj[v]) for v in range(target_n)]
            if min(degs) >= min_deg_req:
                # Full C4 verification
                is_c4_free = True
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if v not in adj[u]:  # non-adjacent in G
                            common = adj[u] & adj[v]
                            if len(common) >= 2:
                                is_c4_free = False
                                break
                    if not is_c4_free:
                        break
                
                if is_c4_free:
                    edges = []
                    for u in range(target_n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    best_result = {'n': target_n, 'edges': edges}
                    if target_n == 46:
                        return best_result
                    break  # Got this n, try higher
    
    return best_result
# EVOLVE-BLOCK-END
