import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    orig_edge_set = set()
    for u, v in edges:
        if u > v: u, v = v, u
        orig_edge_set.add((u, v))
        
    host_adj = [set() for _ in range(n)]
    for u, v in orig_edge_set:
        host_adj[u].add(v)
        host_adj[v].add(u)

    best_count = float('inf')
    best_bicliques = []
    start_time = time.time()
    time_limit = 55.0

    def solve(rem_edges, current_bicliques, count):
        nonlocal best_count, best_bicliques
        
        if time.time() - start_time > time_limit:
            return
            
        if not rem_edges:
            if count < best_count:
                best_count = count
                best_bicliques = [b[:] for b in current_bicliques]
            return
            
        if count >= best_count:
            return
            
        degs = [0] * n
        for u, v in rem_edges:
            degs[u] += 1
            degs[v] += 1
            
        max_deg = max(degs)
        if max_deg == 0:
            return
            
        v = degs.index(max_deg)
        
        N_v = set()
        for u, w in rem_edges:
            if u == v: N_v.add(w)
            elif w == v: N_v.add(u)
            
        if not N_v:
            return

        u_min = min(N_v)
        
        candidates = [N_v, {u_min}]
        for u in N_v:
            if u != u_min:
                candidates.append({u})
                candidates.append({u_min, u})
                
        for _ in range(10):
            size = random.randint(1, len(N_v))
            R_1 = {u_min}
            others = list(N_v - {u_min})
            random.shuffle(others)
            R_1.update(others[:size-1])
            candidates.append(R_1)
            
        seen = set()
        unique_candidates = []
        for c in candidates:
            key = frozenset(c)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)
                
        candidates = unique_candidates
        candidates.sort(key=len, reverse=True)

        for R_1 in candidates:
            L_1 = {v}
            potential_L = set()
            for u in range(n):
                if u in R_1:
                    continue
                valid = True
                for r in R_1:
                    a, b = (u, r) if u < r else (r, u)
                    if (a, b) not in rem_edges:
                        valid = False
                        break
                if valid:
                    potential_L.add(u)
            
            L_1 = L_1.union(potential_L)
            
            edges_to_remove = set()
            for u in L_1:
                for r in R_1:
                    if u == r: continue
                    a, b = (u, r) if u < r else (r, u)
                    if (a, b) in rem_edges:
                        edges_to_remove.add((a, b))
            
            if not edges_to_remove:
                continue
                
            new_rem = rem_edges.difference(edges_to_remove)
            
            current_bicliques.append([list(L_1), list(R_1)])
            solve(new_rem, current_bicliques, count + 1)
            current_bicliques.pop()

    solve(orig_edge_set, [], 0)
    
    if best_bicliques is None or best_count == float('inf'):
        return [[ [u], [v] ] for u, v in orig_edge_set]
        
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
    return result
