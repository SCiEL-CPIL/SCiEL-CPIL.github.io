"""C4-free graph with high minimum degree via incremental simulated annealing."""
import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 142.0

    def time_left():
        return DEADLINE - (time.time() - start_time)

    best_result = {'n': 39, 'edges': []}

    for n in range(47, 38, -1):
        if time_left() < 18:
            break

        found_valid = False
        for restart in range(10):
            if time_left() < 10:
                break

            # State: adj matrix of G (1=edge in G), and track:
            # - deg[v]: degree of v in G
            # - common[u,v]: number of common neighbors of u,v in G (for u != v)
            # - c4_count: total number of "excess" common neighbors = sum_{u<v} max(0, common[u,v]-1)
            #   Actually let's track: for each pair, whether common > 1, and by how much.
            
            # Better: track c4_pairs = number of pairs (u,v) with common[u,v] >= 2
            # and total_excess = sum of (common[u,v] - 1) over pairs with common >= 2
            
            # When toggling edge (u,v):
            # For each w not in {u,v}:
            #   If we add edge (u,v) to G: common[u,w] += adj[v,w], common[v,w] += adj[u,w]
            #   If we remove edge (u,v) from G: common[u,w] -= adj[v,w], common[v,w] -= adj[u,w]
            #   Also common[u,v] changes: if we add edge, common[u,v] stays same (it's about other neighbors)
            #   Wait, common[u,v] = |N(u) ∩ N(v)|. Toggling edge (u,v) doesn't change N(u) ∩ N(v)
            #   because u is not in N(u) and v is not in N(v) (simple graph, no self-loops).
            #   Actually N(u) includes v if edge (u,v) exists. So if we add edge (u,v),
            #   then v ∈ N(u) and u ∈ N(v), but this doesn't affect common[u,v] since
            #   common[u,v] = |{w : w~u and w~v}| and w ranges over all vertices including u,v?
            #   No, w must be a common NEIGHBOR, so w ~ u and w ~ v. u ~ u is false, v ~ v is false.
            #   So common[u,v] = |{w ≠ u,v : w~u and w~v}|. Toggling (u,v) doesn't change this.
            #   
            #   But toggling (u,v) DOES change common[u,w] for w ≠ u,v:
            #   common[u,w] = |{x : x~u and x~w}|. If we add edge (u,v), then v becomes a neighbor of u.
            #   So if v~w, then v becomes a new common neighbor of u and w.
            #   Similarly for common[v,w].
            
            # Let me track:
            # adj: n x n boolean, adj[i,j] = 1 iff edge in G
            # deg: array of degrees
            # common: n x n matrix, common[i,j] = number of common neighbors of i and j
            # c4_excess: total excess = sum_{i<j} max(0, common[i,j] - 1)
            # min_deg: minimum degree
            
            # For the energy: c4_excess + penalty * max(0, target_md - min_deg)
            
            # Initialize
            adj = np.zeros((n, n), dtype=np.int8)
            # Start with dense graph, then remove edges to make H
            adj[:] = 1
            np.fill_diagonal(adj, 0)
            
            # Remove random edges to make each vertex have degree >= n-39
            # i.e., each vertex loses at most 38 edges
            for v in range(n):
                others = list(range(n))
                others.remove(v)
                random.shuffle(others)
                to_remove = others[:min(38, n-1)]
                for u in to_remove:
                    if adj[v, u]:
                        adj[v, u] = 0
                        adj[u, v] = 0
            
            # Compute initial state
            deg = adj.sum(axis=1).astype(np.int32)
            common = (adj @ adj.T).astype(np.int32)
            # common[i,i] = deg[i], but we only care about i != j
            np.fill_diagonal(common, 0)
            
            # c4_excess
            c4_excess = 0
            for i in range(n):
                for j in range(i+1, n):
                    if common[i,j] > 1:
                        c4_excess += (int(common[i,j]) - 1)
            
            min_deg = int(np.min(deg))
            target_md = n - 39
            deg_deficit = max(0, target_md - min_deg)
            curr_e = c4_excess + 10000 * deg_deficit
            
            best_e = curr_e
            best_adj = adj.copy()
            best_deg = deg.copy()
            
            T = 5.0
            cooling = 0.99995
            max_it = 500000
            
            for it in range(max_it):
                if time_left() < 6:
                    break
                if it % 5000 == 0 and it > 0:
                    T = max(0.005, T * cooling)
                
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if v == u:
                    continue
                
                if adj[u, v] == 1:
                    # Remove edge (u,v) from G
                    # For each w != u,v:
                    #   common[u,w] -= adj[v,w]  (if v~w, v was a common neighbor of u,w)
                    #   common[v,w] -= adj[u,w]
                    #   common[w,u] -= adj[v,w]
                    #   common[w,v] -= adj[u,w]
                    # deg[u] -= 1, deg[v] -= 1
                    # c4_excess changes based on how common[u,w] and common[v,w] cross threshold 1
                    
                    old_c4 = c4_excess
                    old_min = min_deg
                    
                    # Update c4_excess for pairs involving u or v
                    # Pairs (u,w) for w != u,v: common[u,w] changes by -adj[v,w]
                    # Pairs (v,w) for w != u,v: common[v,w] changes by -adj[u,w]
                    
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        # Pair (u,w): common[u,w] -= adj[v,w]
                        if adj[v, w]:
                            old_c = common[u, w]
                            new_c = old_c - 1
                            common[u, w] = new_c
                            common[w, u] = new_c
                            # Update c4_excess contribution
                            old_contrib = max(0, old_c - 1)
                            new_contrib = max(0, new_c - 1)
                            c4_excess += (new_contrib - old_contrib)
                        
                        # Pair (v,w): common[v,w] -= adj[u,w]
                        if adj[u, w]:
                            old_c = common[v, w]
                            new_c = old_c - 1
                            common[v, w] = new_c
                            common[w, v] = new_c
                            old_contrib = max(0, old_c - 1)
                            new_contrib = max(0, new_c - 1)
                            c4_excess += (new_contrib - old_contrib)
                    
                    # Update degrees
                    deg[u] -= 1
                    deg[v] -= 1
                    adj[u, v] = 0
                    adj[v, u] = 0
                    
                    new_min = int(np.min(deg))
                    new_deficit = max(0, target_md - new_min)
                    new_e = c4_excess + 10000 * new_deficit
                    delta = new_e - curr_e
                    
                    if delta <= 0 or random.random() < math.exp(-delta / max(T, 0.01)):
                        curr_e = new_e
                        min_deg = new_min
                        if new_e < best_e:
                            best_e = new_e
                            best_adj = adj.copy()
                    else:
                        # Revert
                        adj[u, v] = 1
                        adj[v, u] = 1
                        deg[u] += 1
                        deg[v] += 1
                        for w in range(n):
                            if w == u or w == v:
                                continue
                            if adj[v, w]:
                                common[u, w] += 1
                                common[w, u] += 1
                            if adj[u, w]:
                                common[v, w] += 1
                                common[w, v] += 1
                        c4_excess = old_c4
                        min_deg = old_min
                else:
                    # Add edge (u,v) to G
                    old_c4 = c4_excess
                    old_min = min_deg
                    
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if adj[v, w]:
                            old_c = common[u, w]
                            new_c = old_c + 1
                            common[u, w] = new_c
                            common[w, u] = new_c
                            old_contrib = max(0, old_c - 1)
                            new_contrib = max(0, new_c - 1)
                            c4_excess += (new_contrib - old_contrib)
                        
                        if adj[u, w]:
                            old_c = common[v, w]
                            new_c = old_c + 1
                            common[v, w] = new_c
                            common[w, v] = new_c
                            old_contrib = max(0, old_c - 1)
                            new_contrib = max(0, new_c - 1)
                            c4_excess += (new_contrib - old_contrib)
                    
                    deg[u] += 1
                    deg[v] += 1
                    adj[u, v] = 1
                    adj[v, u] = 1
                    
                    new_min = int(np.min(deg))
                    new_deficit = max(0, target_md - new_min)
                    new_e = c4_excess + 10000 * new_deficit
                    delta = new_e - curr_e
                    
                    if delta <= 0 or random.random() < math.exp(-delta / max(T, 0.01)):
                        curr_e = new_e
                        min_deg = new_min
                        if new_e < best_e:
                            best_e = new_e
                            best_adj = adj.copy()
                    else:
                        adj[u, v] = 0
                        adj[v, u] = 0
                        deg[u] -= 1
                        deg[v] -= 1
                        for w in range(n):
                            if w == u or w == v:
                                continue
                            if adj[v, w]:
                                common[u, w] -= 1
                                common[w, u] -= 1
                            if adj[u, w]:
                                common[v, w] -= 1
                                common[w, v] -= 1
                        c4_excess = old_c4
                        min_deg = old_min
            
            if best_e == 0:
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if best_adj[i, j] == 1:
                            edges.append([i, j])
                if n > best_result['n']:
                    best_result = {'n': n, 'edges': edges}
                found_valid = True
                break

        if found_valid:
            break

    return best_result
