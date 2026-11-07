import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = P['n']
    edges = P['edges']
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
        
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)

    candidates = []
    seen_bicliques = set()
    
    for u, v in edge_set:
        L = {u}
        R = {v}
        changed = True
        while changed:
            changed = False
            for u_prime in range(n):
                if u_prime not in L:
                    if R.issubset(adj[u_prime]):
                        L.add(u_prime)
                        changed = True
            for v_prime in range(n):
                if v_prime not in R:
                    if L.issubset(adj[v_prime]):
                        R.add(v_prime)
                        changed = True
        
        biclique_key = (frozenset(L), frozenset(R))
        if biclique_key not in seen_bicliques:
            seen_bicliques.add(biclique_key)
            biclique_edges = set()
            for l in L:
                for r in R:
                    e = (l, r) if l < r else (r, l)
                    biclique_edges.add(e)
            candidates.append((L, R, biclique_edges))

    candidates.sort(key=lambda x: len(x[2]), reverse=True)
    
    edge_list = list(edge_set)
    M = len(edge_list)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    
    candidate_masks = []
    for L, R, b_edges in candidates:
        mask = 0
        for e in b_edges:
            idx = edge_to_idx[e]
            mask |= (1 << idx)
        candidate_masks.append(mask)
        
    initial_mask = (1 << M) - 1
    
    edge_to_candidates = {e: [] for e in edge_set}
    for i, (L, R, b_edges) in enumerate(candidates):
        for e in b_edges:
            edge_to_candidates[e].append(i)

    best_solution = None
    best_depth = float('inf')
    max_biclique_size = max(len(x[2]) for x in candidates) if candidates else 1

    def solve(uncovered_mask, current_depth, current_path):
        nonlocal best_solution, best_depth
        
        if time.time() > deadline:
            return
            
        if uncovered_mask == 0:
            if current_depth < best_depth:
                best_depth = current_depth
                best_solution = list(current_path)
            return
            
        # Lower bound
        remaining_edges = uncovered_mask.bit_count()
        lb = (remaining_edges + max_biclique_size - 1) // max_biclique_size
        if current_depth + lb >= best_depth:
            return
            
        # Find edge with fewest valid candidates
        min_count = float('inf')
        target_edge_idx = -1
        target_candidates = []
        
        # Iterate through uncovered edges
        # To speed up, we can check bits
        for i in range(M):
            if (uncovered_mask >> i) & 1:
                # Count valid candidates
                count = 0
                for c_idx in edge_to_candidates[edge_list[i]]:
                    if (candidate_masks[c_idx] & ~uncovered_mask) == 0:
                        count += 1
                        if count >= min_count:
                            break
                if count < min_count:
                    min_count = count
                    target_edge_idx = i
                    target_candidates = edge_to_candidates[edge_list[i]]
                    if min_count == 1:
                        break
                        
        if target_edge_idx == -1:
            return
            
        valid_candidates = []
        for c_idx in target_candidates:
            if (candidate_masks[c_idx] & ~uncovered_mask) == 0:
                valid_candidates.append(c_idx)
                
        if not valid_candidates:
            return

        valid_candidates.sort(key=lambda idx: candidate_masks[idx].bit_count(), reverse=True)
        
        for c_idx in valid_candidates:
            new_uncovered = uncovered_mask & ~candidate_masks[c_idx]
            current_path.append(c_idx)
            solve(new_uncovered, current_depth + 1, current_path)
            current_path.pop()
            
    solve(initial_mask, 0, [])
    
    if best_solution is None:
        # Fallback
        star_partition = []
        for u in range(n):
            neighbors = [v for v in adj[u]]
            if neighbors:
                star_partition.append([ [u], neighbors ])
        return star_partition

    result = []
    for idx in best_solution:
        L, R, _ = candidates[idx]
        result.append([list(L), list(R)])
        
    return result
