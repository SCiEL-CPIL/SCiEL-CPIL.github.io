import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Heuristic: Order vertices by degree (all 4), so just 0..n-1
    # But maybe we can improve by ordering based on "clique" or something?
    # Let's stick to 0..n-1.
    
    def solve(k, time_limit):
        colors = [-1] * n
        pair_edges = {}
        
        def is_connected(u, v, edge_list):
            if not edge_list:
                return False
            
            # Build adjacency list for the subgraph
            sub_adj = {}
            for a, b in edge_list:
                if a not in sub_adj: sub_adj[a] = []
                if b not in sub_adj: sub_adj[b] = []
                sub_adj[a].append(b)
                sub_adj[b].append(a)
            
            # BFS
            if u not in sub_adj:
                return False
            if v not in sub_adj:
                return False
                
            visited = set([u])
            queue = [u]
            while queue:
                cur = queue.pop(0)
                if cur == v:
                    return True
                if cur in sub_adj:
                    for nei in sub_adj[cur]:
                        if nei not in visited:
                            visited.add(nei)
                            queue.append(nei)
            return False

        def backtrack(idx):
            if time.time() > start_time + time_limit:
                return False, "timeout"
            
            if idx == n:
                return True, "success"
            
            v = idx
            for c in range(k):
                valid = True
                affected_pairs = []
                
                for u in adj[v]:
                    if colors[u] != -1:
                        if colors[u] == c:
                            valid = False
                            break
                        c_u = colors[u]
                        c1, c2 = (c, c_u) if c < c_u else (c_u, c)
                        pair = (c1, c2)
                        
                        if pair not in pair_edges:
                            pair_edges[pair] = []
                            
                        if is_connected(u, v, pair_edges[pair]):
                            valid = False
                            break
                        else:
                            affected_pairs.append((pair, u, v))
                
                if not valid:
                    continue
                
                colors[v] = c
                for pair, u, vv in affected_pairs:
                    pair_edges[pair].append((u, vv))
                
                res, status = backtrack(idx + 1)
                if res:
                    return True, status
                
                colors[v] = -1
                for pair, u, vv in affected_pairs:
                    pair_edges[pair].pop()
                    
            return False, "failed"

        start_time = time.time()
        res, status = backtrack(0)
        if res:
            return colors
        return None

    # Try 3 colors
    res = solve(3, 20)
    if res is not None:
        return res
        
    # Try 4 colors
    res = solve(4, 20)
    if res is not None:
        return res
        
    # Try 5 colors
    res = solve(5, 20)
    if res is not None:
        return res
        
    return list(range(P['n']))
