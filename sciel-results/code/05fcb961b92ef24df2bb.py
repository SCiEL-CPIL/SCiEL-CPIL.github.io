import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges_list = P['edges']
    
    edge_set = set()
    adj = [set() for _ in range(n)]
    for e in edges_list:
        a, b = e
        if a > b:
            a, b = b, a
        edge_set.add((a, b))
        adj[a].add(b)
        adj[b].add(a)
    
    all_edges = frozenset(edge_set)
    total_edges = len(all_edges)
    
    start_time = time.time()
    deadline = start_time + 75.0
    
    def expand_biclique(L_init, R_init):
        L = set(L_init)
        R = set(R_init)
        changed = True
        while changed:
            changed = False
            for x in range(n):
                if x not in L and x not in R:
                    if all((min(x, y), max(x, y)) in edge_set for y in R):
                        L.add(x)
                        changed = True
            for y in range(n):
                if y not in L and y not in R:
                    if all((min(x, y), max(x, y)) in edge_set for x in L):
                        R.add(y)
                        changed = True
        return L, R
    
    def get_edges(L, R):
        es = set()
        for l in L:
            for r in R:
                e = (min(l, r), max(l, r))
                if e in edge_set:
                    es.add(e)
        return frozenset(es)
    
    # Generate candidates
    candidates = []
    seen = set()
    
    for u, v in edge_set:
        for uu, vv in [(u, v), (v, u)]:
            L, R = expand_biclique({uu}, {vv})
            if L and R and L.isdisjoint(R):
                key = (frozenset(L), frozenset(R))
                if key not in seen:
                    seen.add(key)
                    es = get_edges(L, R)
                    if es and es <= all_edges:
                        candidates.append((frozenset(L), frozenset(R), es))
    
    # Add trimmed versions
    for L_f, R_f, es in list(candidates):
        for l in L_f:
            Ln = L_f - {l}
            if Ln:
                key = (frozenset(Ln), R_f)
                if key not in seen:
                    esn = get_edges(Ln, R_f)
                    if esn and esn <= all_edges:
                        seen.add(key)
                        candidates.append((frozenset(Ln), R_f, esn))
        for r in R_f:
            Rn = R_f - {r}
            if Rn:
                key = (L_f, frozenset(Rn))
                if key not in seen:
                    esn = get_edges(L_f, Rn)
                    if esn and esn <= all_edges:
                        seen.add(key)
                        candidates.append((L_f, frozenset(Rn), esn))
    
    # Deduplicate by edge set
    seen_es = set()
    unique = []
    for L, R, es in candidates:
        if es not in seen_es:
            seen_es.add(es)
            unique.append((L, R, es))
    candidates = unique
    
    # Build edge-to-candidate index
    edge_to_cands = {}
    for i, (L, R, es) in enumerate(candidates):
        for e in es:
            edge_to_cands.setdefault(e, []).append(i)
    
    cand_edges = [c[2] for c in candidates]
    cand_L = [c[0] for c in candidates]
    cand_R = [c[1] for c in candidates]
    num_cands = len(candidates)
    
    # Greedy initial solution
    def greedy_solve():
        rem = set(all_edges)
        sol = []
        while rem:
            best_i = -1
            best_cov = 0
            for i in range(num_cands):
                cov = len(cand_edges[i] & rem)
                if cov > best_cov:
                    best_cov = cov
                    best_i = i
            if best_i < 0 or best_cov == 0:
                e = next(iter(rem))
                u, v = e
                sol.append((frozenset({u}), frozenset({v}), frozenset({e})))
                rem.discard(e)
            else:
                sol.append(candidates[best_i])
                rem -= cand_edges[best_i]
        return sol
    
    greedy_sol = greedy_solve()
    best_count = len(greedy_sol)
    best_sol = greedy_sol
    
    # Branch and bound
    nodes_explored = 0
    
    def bb(remaining, chosen, depth):
        nonlocal best_count, best_sol, nodes_explored
        
        if time.time() > deadline:
            return
        nodes_explored += 1
        
        if not remaining:
            if depth < best_count:
                best_count = depth
                best_sol = list(chosen)
            return
        
        if depth >= best_count:
            return
        
        # Lower bound
        max_cov = 0
        for i in range(num_cands):
            c = len(cand_edges[i] & remaining)
            if c > max_cov:
                max_cov = c
            if max_cov >= len(remaining):
                break
        if max_cov == 0:
            lb = len(remaining)
        else:
            lb = (len(remaining) + max_cov - 1) // max_cov
        
        if depth + lb >= best_count:
            return
        
        # Pick most constrained edge
        min_cnt = float('inf')
        pick = None
        for e in remaining:
            cnt = 0
            for i in edge_to_cands.get(e, []):
                if cand_edges[i] <= remaining:
                    cnt += 1
            if cnt < min_cnt:
                min_cnt = cnt
                pick = e
            if cnt == 0:
                return
        
        if pick is None or min_cnt == 0:
            return
        
        # Try candidates covering pick, sorted by size desc
        cands_for_pick = [i for i in edge_to_cands.get(pick, []) if cand_edges[i] <= remaining]
        cands_for_pick.sort(key=lambda i: -len(cand_edges[i]))
        
        for i in cands_for_pick:
            chosen.append(i)
            bb(remaining - cand_edges[i], chosen, depth + 1)
            chosen.pop()
    
    bb(frozenset(all_edges), [], 0)
    
    # Build result
    if best_sol:
        result = []
        for item in best_sol:
            if isinstance(item, int):
                L = cand_L[item]
                R = cand_R[item]
            else:
                L, R, _ = item
            result.append([[int(x) for x in sorted(L)], [int(x) for x in sorted(R)]])
    else:
        result = [[int(u), ], [int(v)]] for u, v in all_edges]
    
    return result
