import random
import time
import numpy as np

def construct():
    deadline = time.time() + 115
    n = 16
    target_edges = 65
    
    def compute_score(adj, deg):
        """Compute the maximum degree sum over all triangles."""
        max_sum = 0
        worst_tri = None
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    continue
                for w in range(v + 1, n):
                    if adj[u, w] and adj[v, w]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                            worst_tri = (u, v, w)
        return max_sum, worst_tri
    
    def try_swap(current_adj, current_deg, current_score, e_remove, e_add):
        """Try a single edge swap and return new state if score improves."""
        u1, v1 = e_remove
        u2, v2 = e_add
        
        if not current_adj[u1, v1] or current_adj[u2, v2]:
            return None
        
        new_adj = current_adj.copy()
        new_deg = current_deg.copy()
        
        new_adj[u1, v1] = False
        new_adj[v1, u1] = False
        new_deg[u1] -= 1
        new_deg[v1] -= 1
        
        new_adj[u2, v2] = True
        new_adj[v2, u2] = True
        new_deg[u2] += 1
        new_deg[v2] += 1
        
        new_score, new_worst = compute_score(new_adj, new_deg)
        
        if new_score < current_score:
            return new_adj, new_deg, new_score, new_worst
        return None
    
    def soft_guided_search(adj, deg, score, worst_tri, deadline, max_iters=800):
        """Local search guided by degree-based soft partition heuristic."""
        current_adj = adj.copy()
        current_deg = deg.copy()
        current_score = score
        current_worst = worst_tri
        
        all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        T = 2.0
        cooling = 0.998
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            moved = False
            
            if current_worst is not None:
                u, v, w = current_worst
                tri_edges = [(u, v), (u, w), (v, w)]
                random.shuffle(tri_edges)
                
                for e1 in tri_edges:
                    if not current_adj[e1[0], e1[1]]:
                        continue
                    
                    # Rank candidate additions: prefer edges that minimize new triangle degree sums
                    candidates = []
                    for e2 in all_possible_edges:
                        if not current_adj[e2[0], e2[1]]:
                            a, b = e2
                            penalty = 0
                            for t in range(n):
                                if t != a and t != b and current_adj[a, t] and current_adj[b, t]:
                                    penalty += current_deg[t]
                            candidates.append((penalty, e2))
                    
                    candidates.sort(key=lambda x: x[0])
                    
                    for _, e2 in candidates[:20]:
                        result = try_swap(current_adj, current_deg, current_score, e1, e2)
                        if result is not None:
                            current_adj, current_deg, current_score, current_worst = result
                            moved = True
                            break
                    
                    if moved:
                        break
            
            if not moved:
                # Random move with degree-weighted guidance
                e1 = None
                e2 = None
                
                for _ in range(20):
                    high_deg_edges = [(e, current_deg[e[0]] + current_deg[e[1]]) 
                                     for e in all_possible_edges if current_adj[e[0], e[1]]]
                    if high_deg_edges:
                        weights = [w for _, w in high_deg_edges]
                        total_w = sum(weights)
                        if total_w > 0:
                            r = random.random() * total_w
                            cumsum = 0
                            for e, w in high_deg_edges:
                                cumsum += w
                                if cumsum >= r:
                                    e1 = e
                                    break
                            if e1 is None:
                                e1 = high_deg_edges[0][0]
                    
                    low_deg_nonedges = [(e, current_deg[e[0]] + current_deg[e[1]])
                                      for e in all_possible_edges if not current_adj[e[0], e[1]]]
                    if low_deg_nonedges:
                        weights = [w for _, w in low_deg_nonedges]
                        total_w = sum(weights)
                        if total_w > 0:
                            r = random.random() * total_w
                            cumsum = 0
                            for e, w in low_deg_nonedges:
                                cumsum += w
                                if cumsum >= r:
                                    e2 = e
                                    break
                            if e2 is None:
                                e2 = low_deg_nonedges[0][0]
                    
                    if e1 is not None and e2 is not None:
                        break
                
                if e1 is None or e2 is None:
                    T *= cooling
                    continue
                
                result = try_swap(current_adj, current_deg, current_score, e1, e2)
                if result is not None:
                    current_adj, current_deg, current_score, current_worst = result
                else:
                    u1, v1 = e1
                    u2, v2 = e2
                    new_adj = current_adj.copy()
                    new_deg = current_deg.copy()
                    new_adj[u1, v1] = False
                    new_adj[v1, u1] = False
                    new_deg[u1] -= 1
                    new_deg[v1] -= 1
                    new_adj[u2, v2] = True
                    new_adj[v2, u2] = True
                    new_deg[u2] += 1
                    new_deg[v2] += 1
                    new_score, new_worst = compute_score(new_adj, new_deg)
                    delta = new_score - current_score
                    if delta > 0 and random.random() < np.exp(-delta / T):
                        current_adj = new_adj
                        current_deg = new_deg
                        current_score = new_score
                        current_worst = new_worst
            
            T *= cooling
        
        return current_adj, current_deg, current_score, current_worst
    
    def exhaustive_polish(adj, deg, score, worst_tri, deadline):
        """Exhaustive polish: try ALL edge swaps involving worst triangle vertices."""
        current_adj = adj.copy()
        current_deg = deg.copy()
        current_score = score
        current_worst = worst_tri
        
        all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        improved = True
        passes = 0
        while improved and passes < 30:
            if time.time() > deadline - 3:
                break
            passes += 1
            improved = False
            
            if current_worst is None:
                break
                
            u, v, w = current_worst
            tri_vertices = set([u, v, w])
            
            candidate_removals = []
            for e in all_possible_edges:
                if current_adj[e[0], e[1]]:
                    if e in [(u, v), (u, w), (v, w)]:
                        candidate_removals.insert(0, e)
                    elif e[0] in tri_vertices or e[1] in tri_vertices:
                        candidate_removals.insert(0, e)
                    else:
                        candidate_removals.append(e)
            
            candidate_adds = []
            for e in all_possible_edges:
                if not current_adj[e[0], e[1]]:
                    if e[0] in tri_vertices or e[1] in tri_vertices:
                        candidate_adds.insert(0, e)
                    else:
                        candidate_adds.append(e)
            
            for e1 in candidate_removals[:40]:
                if time.time() > deadline - 3:
                    break
                for e2 in candidate_adds[:40]:
                    result = try_swap(current_adj, current_deg, current_score, e1, e2)
                    if result is not None:
                        current_adj, current_deg, current_score, current_worst = result
                        improved = True
                        break
                if improved:
                    break
        
        return current_adj, current_deg, current_score, current_worst
    
    def make_initial_graph(seed, split_size):
        """Create an initial graph with a given bipartition."""
        random.seed(seed)
        adj = np.zeros((n, n), dtype=bool)
        
        perm = list(range(n))
        random.shuffle(perm)
        A = set(perm[:split_size])
        B = set(perm[split_size:])
        
        for u in A:
            for v in B:
                adj[u, v] = True
                adj[v, u] = True
        
        edge_count = len(A) * len(B)
        remaining = target_edges - edge_count
        
        if remaining > 0:
            internal_edges = []
            for part in [A, B]:
                part_list = list(part)
                for i in range(len(part_list)):
                    for j in range(i + 1, len(part_list)):
                        u, v = part_list[i], part_list[j]
                        if not adj[u, v]:
                            internal_edges.append((u, v))
            random.shuffle(internal_edges)
            for u, v in internal_edges[:remaining]:
                adj[u, v] = True
                adj[v, u] = True
        elif remaining < 0:
            cross_edges = [(u, v) for u in A for v in B]
            random.shuffle(cross_edges)
            for u, v in cross_edges[:abs(remaining)]:
                adj[u, v] = False
                adj[v, u] = False
        
        deg = adj.sum(axis=1).astype(int)
        score, worst_tri = compute_score(adj, deg)
        return adj, deg, score, worst_tri
    
    best_score = float('inf')
    best_adj = None
    best_deg = None
    best_worst_tri = None
    
    for seed in range(100):
        if time.time() > deadline - 20:
            break
        
        if seed % 4 == 0:
            split = 8
        elif seed % 4 == 1:
            split = 7
        elif seed % 4 == 2:
            split = 9
        else:
            split = 8
        
        adj, deg, score, worst_tri = make_initial_graph(seed, split)
        
        if score < best_score:
            best_score = score
            best_adj = adj.copy()
            best_deg = deg.copy()
            best_worst_tri = worst_tri
        
        current_adj, current_deg, current_score, current_worst = soft_guided_search(
            adj, deg, score, worst_tri, deadline, max_iters=800
        )
        
        if current_score < best_score:
            best_score = current_score
            best_adj = current_adj.copy()
            best_deg = current_deg.copy()
            best_worst_tri = current_worst
    
    if best_adj is not None:
        polished_adj, polished_deg, polished_score, polished_worst = exhaustive_polish(
            best_adj, best_deg, best_score, best_worst_tri, deadline
        )
        
        if polished_score < best_score:
            best_score = polished_score
            best_adj = polished_adj.copy()
            best_deg = polished_deg.copy()
            best_worst_tri = polished_worst
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}
