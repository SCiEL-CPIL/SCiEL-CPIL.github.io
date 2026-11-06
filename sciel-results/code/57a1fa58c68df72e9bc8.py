import numpy as np
import time
import itertools

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit
    
    # Precompute triangle indices
    tri_list = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                tri_list.append((i, j, k))
    tri_arr = np.array(tri_list)
    I = tri_arr[:, 0]
    J = tri_arr[:, 1]
    K = tri_arr[:, 2]
    n_tri = len(tri_arr)
    
    A = list(range(8))
    B = list(range(8, 16))
    all_cross = [(u, v) for u in A for v in B]
    all_int_A = list(itertools.combinations(A, 2))
    all_int_B = list(itertools.combinations(B, 2))
    
    def get_score(adj):
        deg = adj.sum(axis=1)
        valid = (adj[I, J] == 1) & (adj[J, K] == 1) & (adj[K, I] == 1)
        if not np.any(valid):
            return 0
        sums = deg[I[valid]] + deg[J[valid]] + deg[K[valid]]
        return int(np.max(sums))
    
    def build_adj(removed_cross, int_A, int_B):
        adj = np.zeros((n, n), dtype=np.int8)
        for u in range(8):
            for v in range(8, 16):
                adj[u, v] = 1
                adj[v, u] = 1
        for u, v in removed_cross:
            adj[u, v] = 0
            adj[v, u] = 0
        for u, v in int_A:
            adj[u, v] = 1
            adj[v, u] = 1
        for u, v in int_B:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj
    
    # Phase 1: Systematic search over small perturbations of K_{8,8}
    best_adj = None
    best_score = 999
    
    # k=0: baseline (64 cross + 1 internal = 65)
    for u, v in all_int_A:
        adj = build_adj([], [(u, v)], [])
        if adj.sum() // 2 == target_edges:
            s = get_score(adj)
            if s < best_score:
                best_score = s
                best_adj = adj.copy()
    
    for u, v in all_int_B:
        adj = build_adj([], [], [(u, v)])
        if adj.sum() // 2 == target_edges:
            s = get_score(adj)
            if s < best_score:
                best_score = s
                best_adj = adj.copy()
    
    # k=1: remove 1 cross, add 2 internal
    if time.time() - start_time < deadline - 60:
        for rc in all_cross:
            for ia in all_int_A:
                for ib in all_int_B:
                    adj = build_adj([rc], [ia], [ib])
                    if adj.sum() // 2 == target_edges:
                        s = get_score(adj)
                        if s < best_score:
                            best_score = s
                            best_adj = adj.copy()
            for ia1, ia2 in itertools.combinations(all_int_A, 2):
                adj = build_adj([rc], [ia1, ia2], [])
                if adj.sum() // 2 == target_edges:
                    s = get_score(adj)
                    if s < best_score:
                        best_score = s
                        best_adj = adj.copy()
            for ib1, ib2 in itertools.combinations(all_int_B, 2):
                adj = build_adj([rc], [], [ib1, ib2])
                if adj.sum() // 2 == target_edges:
                    s = get_score(adj)
                    if s < best_score:
                        best_score = s
                        best_adj = adj.copy()
            if time.time() - start_time > deadline - 60:
                break
    
    # k=2: remove 2 cross, add 3 internal
    if time.time() - start_time < deadline - 40:
        for rc1, rc2 in itertools.combinations(all_cross, 2):
            removed = [rc1, rc2]
            for ia in all_int_A:
                for ib in all_int_B:
                    adj = build_adj(removed, [ia], [ib])
                    if adj.sum() // 2 == target_edges:
                        s = get_score(adj)
                        if s < best_score:
                            best_score = s
                            best_adj = adj.copy()
            for ia1, ia2 in itertools.combinations(all_int_A, 2):
                adj = build_adj(removed, [ia1, ia2], [])
                if adj.sum() // 2 == target_edges:
                    s = get_score(adj)
                    if s < best_score:
                        best_score = s
                        best_adj = adj.copy()
            for ib1, ib2 in itertools.combinations(all_int_B, 2):
                adj = build_adj(removed, [], [ib1, ib2])
                if adj.sum() // 2 == target_edges:
                    s = get_score(adj)
                    if s < best_score:
                        best_score = s
                        best_adj = adj.copy()
            if time.time() - start_time > deadline - 40:
                break
    
    # Phase 2: Simulated annealing with multiple restarts
    if best_adj is None:
        best_adj = build_adj([], [(0, 1)], [])
        best_score = get_score(best_adj)
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] == 1:
                    edges.append((i, j))
        return edges
    
    def get_non_edges(adj):
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] == 0:
                    non_edges.append((i, j))
        return non_edges
    
    # Multiple SA restarts with different parameters
    sa_configs = [
        {'T0': 5.0, 'cooling': 0.999, 'min_T': 0.1, 'steps': 50000},
        {'T0': 3.0, 'cooling': 0.9995, 'min_T': 0.2, 'steps': 40000},
        {'T0': 4.0, 'cooling': 0.9992, 'min_T': 0.15, 'steps': 45000},
    ]
    
    for cfg in sa_configs:
        if time.time() - start_time > deadline - 25:
            break
        
        current_adj = best_adj.copy()
        current_score = best_score
        T = cfg['T0']
        cooling = cfg['cooling']
        min_T = cfg['min_T']
        steps = cfg['steps']
        
        # Precompute current edge sets for efficiency
        cur_cross = [(u, v) for u, v in all_cross if current_adj[u, v] == 1]
        cur_cross_removed = [(u, v) for u, v in all_cross if current_adj[u, v] == 0]
        cur_int_A = [(a, b) for a, b in all_int_A if current_adj[a, b] == 1]
        cur_int_A_removed = [(a, b) for a, b in all_int_A if current_adj[a, b] == 0]
        cur_int_B = [(a, b) for a, b in all_int_B if current_adj[a, b] == 1]
        cur_int_B_removed = [(a, b) for a, b in all_int_B if current_adj[a, b] == 0]
        
        for step in range(steps):
            if time.time() - start_time > deadline - 25:
                break
            
            r = np.random.random()
            
            if r < 0.35:
                # Remove a cross edge, add an internal edge
                if not cur_cross:
                    continue
                u, v = cur_cross[np.random.randint(len(cur_cross))]
                
                int_options = cur_int_A_removed + cur_int_B_removed
                if not int_options:
                    continue
                x, y = int_options[np.random.randint(len(int_options))]
                
                current_adj[u, v] = 0
                current_adj[v, u] = 0
                current_adj[x, y] = 1
                current_adj[y, x] = 1
                
                new_score = get_score(current_adj)
                
                if new_score <= current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj.copy()
                    # Update edge sets
                    cur_cross.remove((u, v))
                    cur_cross_removed.append((u, v))
                    if (x, y) in cur_int_A_removed:
                        cur_int_A_removed.remove((x, y))
                        cur_int_A.append((x, y))
                    elif (x, y) in cur_int_B_removed:
                        cur_int_B_removed.remove((x, y))
                        cur_int_B.append((x, y))
                else:
                    delta = new_score - current_score
                    if np.random.random() < np.exp(-delta / max(T, 0.01)):
                        current_score = new_score
                        cur_cross.remove((u, v))
                        cur_cross_removed.append((u, v))
                        if (x, y) in cur_int_A_removed:
                            cur_int_A_removed.remove((x, y))
                            cur_int_A.append((x, y))
                        elif (x, y) in cur_int_B_removed:
                            cur_int_B_removed.remove((x, y))
                            cur_int_B.append((x, y))
                    else:
                        current_adj[u, v] = 1
                        current_adj[v, u] = 1
                        current_adj[x, y] = 0
                        current_adj[y, x] = 0
                        
            elif r < 0.65:
                # Remove an internal edge, add a cross edge
                int_edges = cur_int_A + cur_int_B
                if not int_edges:
                    continue
                x, y = int_edges[np.random.randint(len(int_edges))]
                
                if not cur_cross_removed:
                    continue
                u, v = cur_cross_removed[np.random.randint(len(cur_cross_removed))]
                
                current_adj[x, y] = 0
                current_adj[y, x] = 0
                current_adj[u, v] = 1
                current_adj[v, u] = 1
                
                new_score = get_score(current_adj)
                
                if new_score <= current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj.copy()
                    if (x, y) in cur_int_A:
                        cur_int_A.remove((x, y))
                        cur_int_A_removed.append((x, y))
                    elif (x, y) in cur_int_B:
                        cur_int_B.remove((x, y))
                        cur_int_B_removed.append((x, y))
                    cur_cross_removed.remove((u, v))
                    cur_cross.append((u, v))
                else:
                    delta = new_score - current_score
                    if np.random.random() < np.exp(-delta / max(T, 0.01)):
                        current_score = new_score
                        if (x, y) in cur_int_A:
                            cur_int_A.remove((x, y))
                            cur_int_A_removed.append((x, y))
                        elif (x, y) in cur_int_B:
                            cur_int_B.remove((x, y))
                            cur_int_B_removed.append((x, y))
                        cur_cross_removed.remove((u, v))
                        cur_cross.append((u, v))
                    else:
                        current_adj[x, y] = 1
                        current_adj[y, x] = 1
                        current_adj[u, v] = 0
                        current_adj[v, u] = 0
                        
            else:
                # General edge swap: remove any edge, add any non-edge
                edges = get_edges(current_adj)
                non_edges = get_non_edges(current_adj)
                if not edges or not non_edges:
                    continue
                u, v = edges[np.random.randint(len(edges))]
                x, y = non_edges[np.random.randint(len(non_edges))]
                
                current_adj[u, v] = 0
                current_adj[v, u] = 0
                current_adj[x, y] = 1
                current_adj[y, x] = 1
                
                new_score = get_score(current_adj)
                
                if new_score <= current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj.copy()
                    # Rebuild edge sets
                    cur_cross = [(a, b) for a, b in all_cross if current_adj[a, b] == 1]
                    cur_cross_removed = [(a, b) for a, b in all_cross if current_adj[a, b] == 0]
                    cur_int_A = [(a, b) for a, b in all_int_A if current_adj[a, b] == 1]
                    cur_int_A_removed = [(a, b) for a, b in all_int_A if current_adj[a, b] == 0]
                    cur_int_B = [(a, b) for a, b in all_int_B if current_adj[a, b] == 1]
                    cur_int_B_removed = [(a, b) for a, b in all_int_B if current_adj[a, b] == 0]
                else:
                    delta = new_score - current_score
                    if np.random.random() < np.exp(-delta / max(T, 0.01)):
                        current_score = new_score
                        cur_cross = [(a, b) for a, b in all_cross if current_adj[a, b] == 1]
                        cur_cross_removed = [(a, b) for a, b in all_cross if current_adj[a, b] == 0]
                        cur_int_A = [(a, b) for a, b in all_int_A if current_adj[a, b] == 1]
                        cur_int_A_removed = [(a, b) for a, b in all_int_A if current_adj[a, b] == 0]
                        cur_int_B = [(a, b) for a, b in all_int_B if current_adj[a, b] == 1]
                        cur_int_B_removed = [(a, b) for a, b in all_int_B if current_adj[a, b] == 0]
                    else:
                        current_adj[u, v] = 1
                        current_adj[v, u] = 1
                        current_adj[x, y] = 0
                        current_adj[y, x] = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
    
    # Phase 3: Thorough local search polish
    if time.time() - start_time < deadline - 3:
        improved = True
        polish_round = 0
        while improved and time.time() - start_time < deadline - 3:
            improved = False
            polish_round += 1
            edges = get_edges(best_adj)
            non_edges = get_non_edges(best_adj)
            
            # Try all single-edge swaps
            for u, v in edges:
                if time.time() - start_time > deadline - 3:
                    break
                for x, y in non_edges:
                    if x == u or x == v or y == u or y == v:
                        continue
                    
                    best_adj[u, v] = 0
                    best_adj[v, u] = 0
                    best_adj[x, y] = 1
                    best_adj[y, x] = 1
                    
                    new_score = get_score(best_adj)
                    if new_score < best_score:
                        best_score = new_score
                        improved = True
                        non_edges = get_non_edges(best_adj)
                    else:
                        best_adj[u, v] = 1
                        best_adj[v, u] = 1
                        best_adj[x, y] = 0
                        best_adj[y, x] = 0
            
            # If we improved, also try 2-edge swaps for more aggressive improvement
            if improved and time.time() - start_time < deadline - 3:
                edges = get_edges(best_adj)
                non_edges = get_non_edges(best_adj)
                for i in range(len(edges)):
                    if time.time() - start_time > deadline - 3:
                        break
                    u1, v1 = edges[i]
                    for j in range(i+1, len(edges)):
                        if time.time() - start_time > deadline - 3:
                            break
                        u2, v2 = edges[j]
                        # Check if removing these two edges is valid (no shared vertices issues for adding)
                        for k in range(len(non_edges)):
                            x1, y1 = non_edges[k]
                            if x1 in [u1, v1, u2, v2] or y1 in [u1, v1, u2, v2]:
                                continue
                            for l in range(k+1, len(non_edges)):
                                if time.time() - start_time > deadline - 3:
                                    break
                                x2, y2 = non_edges[l]
                                if x2 in [u1, v1, u2, v2, x1, y1] or y2 in [u1, v1, u2, v2, x1, y1]:
                                    continue
                                
                                best_adj[u1, v1] = 0
                                best_adj[v1, u1] = 0
                                best_adj[u2, v2] = 0
                                best_adj[v2, u2] = 0
                                best_adj[x1, y1] = 1
                                best_adj[y1, x1] = 1
                                best_adj[x2, y2] = 1
                                best_adj[y2, x2] = 1
                                
                                new_score = get_score(best_adj)
                                if new_score < best_score:
                                    best_score = new_score
                                    improved = True
                                else:
                                    best_adj[u1, v1] = 1
                                    best_adj[v1, u1] = 1
                                    best_adj[u2, v2] = 1
                                    best_adj[v2, u2] = 1
                                    best_adj[x1, y1] = 0
                                    best_adj[y1, x1] = 0
                                    best_adj[x2, y2] = 0
                                    best_adj[y2, x2] = 0
    
    # Return result
    edges_list = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] == 1:
                edges_list.append([i, j])
    
    return {'n': n, 'edges': edges_list}
