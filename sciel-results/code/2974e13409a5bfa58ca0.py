import time
import random
import itertools
from collections import defaultdict

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s
    n = 20
    best_edges = None
    best_count = 0
    
    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges
    
    def build_common_neighbors(adj):
        """Build common neighbor count matrix: cn[u][v] = |N(u) ∩ N(v)|"""
        cn = [[0] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                common = len(adj[u] & adj[v])
                cn[u][v] = common
                cn[v][u] = common
        return cn
    
    def greedy_build(seed_val):
        """Build a K2,3-free graph greedily with incremental CN tracking."""
        rng = random.Random(seed_val)
        adj = [set() for _ in range(n)]
        cn = [[0] * n for _ in range(n)]
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if v in adj[u]:
                continue
            
            # Check if adding (u,v) would create K2,3
            # Adding (u,v): for each w != u,v, if w in N(u) then cn[v][w] += 1 (v gains neighbor u's neighbor w... wait)
            # Actually: after adding (u,v), N(u) gains v, N(v) gains u.
            # For pair (u, w): cn[u][w] increases by 1 iff v in N(w) (i.e., w in N(v))
            # For pair (v, w): cn[v][w] increases by 1 iff u in N(w) (i.e., w in N(u))
            # So we need to check: for all w in N(u), cn[v][w] would become cn[v][w]+1; need cn[v][w]+1 < 3, i.e., cn[v][w] <= 1
            # And for all w in N(v), cn[u][w] would become cn[u][w]+1; need cn[u][w]+1 < 3, i.e., cn[u][w] <= 1
            
            ok = True
            for w in adj[u]:
                if cn[v][w] >= 2:  # would become 3
                    ok = False
                    break
            if ok:
                for w in adj[v]:
                    if cn[u][w] >= 2:  # would become 3
                        ok = False
                        break
            
            if not ok:
                continue
            
            # Add the edge and update CN matrix
            adj[u].add(v)
            adj[v].add(u)
            
            # Update CN: for each w in N(u) (excluding v), cn[v][w] += 1
            for w in adj[u]:
                if w != v:
                    cn[v][w] += 1
                    cn[w][v] += 1
            # For each w in N(v) (excluding u), cn[u][w] += 1
            for w in adj[v]:
                if w != u:
                    cn[u][w] += 1
                    cn[w][u] += 1
            # cn[u][v] stays the same (common neighbors of u and v don't change when they become adjacent)
        
        return adj, cn
    
    def local_search(adj_input, cn_input, seed_val, max_iter=5000):
        """Local search with incremental CN tracking."""
        rng = random.Random(seed_val)
        current_adj = [set(s) for s in adj_input]
        cn = [row[:] for row in cn_input]
        current_edges = sum(len(s) for s in current_adj) // 2
        best_local = current_edges
        best_local_adj = [set(s) for s in current_adj]
        best_local_cn = [row[:] for row in cn]
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            action = rng.random()
            
            if action < 0.5:
                # Try to add a random non-edge
                non_edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if v not in current_adj[u]:
                            non_edges.append((u, v))
                if not non_edges:
                    continue
                u, v = rng.choice(non_edges)
                
                # Check validity
                ok = True
                for w in current_adj[u]:
                    if cn[v][w] >= 2:
                        ok = False
                        break
                if ok:
                    for w in current_adj[v]:
                        if cn[u][w] >= 2:
                            ok = False
                            break
                
                if ok:
                    # Add edge
                    current_adj[u].add(v)
                    current_adj[v].add(u)
                    for w in current_adj[u]:
                        if w != v:
                            cn[v][w] += 1
                            cn[w][v] += 1
                    for w in current_adj[v]:
                        if w != u:
                            cn[u][w] += 1
                            cn[w][u] += 1
                    current_edges += 1
                    if current_edges > best_local:
                        best_local = current_edges
                        best_local_adj = [set(s) for s in current_adj]
                        best_local_cn = [row[:] for row in cn]
                # else: skip
                
            elif action < 0.8:
                # Remove edge (u1,v1), add edge (u2,v2)
                edges_list = []
                for u in range(n):
                    for v in current_adj[u]:
                        if u < v:
                            edges_list.append((u, v))
                if not edges_list:
                    continue
                u1, v1 = rng.choice(edges_list)
                
                # Temporarily remove (u1,v1)
                current_adj[u1].discard(v1)
                current_adj[v1].discard(u1)
                # Undo CN updates
                for w in current_adj[u1]:
                    cn[v1][w] -= 1
                    cn[w][v1] -= 1
                for w in current_adj[v1]:
                    cn[u1][w] -= 1
                    cn[w][u1] -= 1
                current_edges -= 1
                
                # Find a non-edge to add
                non_edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if v not in current_adj[u]:
                            non_edges.append((u, v))
                if not non_edges:
                    # Re-add (u1,v1)
                    current_adj[u1].add(v1)
                    current_adj[v1].add(u1)
                    for w in current_adj[u1]:
                        if w != v1:
                            cn[v1][w] += 1
                            cn[w][v1] += 1
                    for w in current_adj[v1]:
                        if w != u1:
                            cn[u1][w] += 1
                            cn[w][u1] += 1
                    current_edges += 1
                    continue
                
                u2, v2 = rng.choice(non_edges)
                
                # Check validity of adding (u2,v2)
                ok = True
                for w in current_adj[u2]:
                    if cn[v2][w] >= 2:
                        ok = False
                        break
                if ok:
                    for w in current_adj[v2]:
                        if cn[u2][w] >= 2:
                            ok = False
                            break
                
                if ok:
                    # Add (u2,v2)
                    current_adj[u2].add(v2)
                    current_adj[v2].add(u2)
                    for w in current_adj[u2]:
                        if w != v2:
                            cn[v2][w] += 1
                            cn[w][v2] += 1
                    for w in current_adj[v2]:
                        if w != u2:
                            cn[u2][w] += 1
                            cn[w][u2] += 1
                    current_edges += 1
                    if current_edges > best_local:
                        best_local = current_edges
                        best_local_adj = [set(s) for s in current_adj]
                        best_local_cn = [row[:] for row in cn]
                else:
                    # Re-add (u1,v1)
                    current_adj[u1].add(v1)
                    current_adj[v1].add(u1)
                    for w in current_adj[u1]:
                        if w != v1:
                            cn[v1][w] += 1
                            cn[w][v1] += 1
                    for w in current_adj[v1]:
                        if w != u1:
                            cn[u1][w] += 1
                            cn[w][u1] += 1
                    current_edges += 1
            
            else:
                # Remove one edge, then try to add up to 2 edges
                edges_list = []
                for u in range(n):
                    for v in current_adj[u]:
                        if u < v:
                            edges_list.append((u, v))
                if not edges_list:
                    continue
                u1, v1 = rng.choice(edges_list)
                
                # Remove (u1,v1)
                current_adj[u1].discard(v1)
                current_adj[v1].discard(u1)
                for w in current_adj[u1]:
                    cn[v1][w] -= 1
                    cn[w][v1] -= 1
                for w in current_adj[v1]:
                    cn[u1][w] -= 1
                    cn[w][u1] -= 1
                current_edges -= 1
                
                # Try to add up to 2 non-edges
                added = 0
                for _ in range(2):
                    non_edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if v not in current_adj[u]:
                                non_edges.append((u, v))
                    if not non_edges:
                        break
                    u2, v2 = rng.choice(non_edges)
                    ok = True
                    for w in current_adj[u2]:
                        if cn[v2][w] >= 2:
                            ok = False
                            break
                    if ok:
                        for w in current_adj[v2]:
                            if cn[u2][w] >= 2:
                                ok = False
                                break
                    if ok:
                        current_adj[u2].add(v2)
                        current_adj[v2].add(u2)
                        for w in current_adj[u2]:
                            if w != v2:
                                cn[v2][w] += 1
                                cn[w][v2] += 1
                        for w in current_adj[v2]:
                            if w != u2:
                                cn[u2][w] += 1
                                cn[w][u2] += 1
                        current_edges += 1
                        added += 1
                    else:
                        break
                
                if current_edges > best_local:
                    best_local = current_edges
                    best_local_adj = [set(s) for s in current_adj]
                    best_local_cn = [row[:] for row in cn]
        
        return best_local_adj, best_local_cn, best_local
    
    # Phase 1: Greedy builds with multiple seeds
    for seed in range(80):
        if time.time() > deadline:
            break
        adj, cn = greedy_build(seed)
        edges = get_edge_list(adj)
        cnt = len(edges)
        if cnt > best_count:
            best_count = cnt
            best_edges = edges
    
    # Phase 2: Local search from best
    if best_edges is not None:
        # Reconstruct adjacency from best_edges
        adj = [set() for _ in range(n)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)
        cn = build_common_neighbors(adj)
        
        for ls_seed in range(50):
            if time.time() > deadline:
                break
            best_adj, best_cn, best_cnt = local_search(adj, cn, ls_seed, max_iter=5000)
            if best_cnt > best_count:
                best_count = best_cnt
                best_edges = get_edge_list(best_adj)
                adj = [set(s) for s in best_adj]
                cn = [row[:] for row in best_cn]
    
    # Phase 3: More greedy + local search
    for seed in range(80, 300):
        if time.time() > deadline:
            break
        adj, cn = greedy_build(seed)
        best_adj, best_cn, best_cnt = local_search(adj, cn, seed, max_iter=3000)
        if best_cnt > best_count:
            best_count = best_cnt
            best_edges = get_edge_list(best_adj)
            adj = [set(s) for s in best_adj]
            cn = [row[:] for row in best_cn]
    
    if best_edges is None:
        # Fallback: 5 disjoint K4's
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
    
    return best_edges
