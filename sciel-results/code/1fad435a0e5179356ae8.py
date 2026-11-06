"""Long path finder using systematic backtracking with pruning."""

import time

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def construct():
    adj = _build_adjacency()
    n = P['n']
    
    best_path = []
    best_len = 0
    
    # Pre-compute degrees for initial ordering
    degrees = [len(adj[i]) for i in range(n)]
    
    # Sort starting nodes: prefer low degree (endpoints of paths)
    start_order = sorted(range(n), key=lambda x: (degrees[x], x))
    
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer
    
    # Helper to get sorted candidates for an end node
    def get_sorted_candidates(end_node, visited):
        cands = []
        for v in adj[end_node]:
            if v not in visited:
                # Heuristic: fewest unvisited neighbors first
                unvisited_neighs = len(adj[v] - visited)
                cands.append((unvisited_neighs, v))
        cands.sort()
        return cands

    def dfs(path, visited):
        nonlocal best_path, best_len, start_time, time_limit
        
        # Check time
        if time.time() - start_time > time_limit:
            return
        
        curr_len = len(path)
        if curr_len > best_len:
            best_len = curr_len
            best_path = path[:]
            
        if curr_len == n:
            return

        # Get candidates for both ends
        start_cands = get_sorted_candidates(path[0], visited)
        end_cands = get_sorted_candidates(path[-1], visited)
        
        if curr_len == 1:
            # Only one end effectively
            for _, v in start_cands:
                path.append(v)
                visited.add(v)
                dfs(path, visited)
                visited.remove(v)
                path.pop()
                if time.time() - start_time > time_limit:
                    return
        else:
            # Try extending from start
            for _, v in start_cands:
                path.insert(0, v)
                visited.add(v)
                dfs(path, visited)
                visited.remove(v)
                path.pop(0)
                if time.time() - start_time > time_limit:
                    return
            
            # Try extending from end
            for _, v in end_cands:
                path.append(v)
                visited.add(v)
                dfs(path, visited)
                visited.remove(v)
                path.pop()
                if time.time() - start_time > time_limit:
                    return

    # Start DFS from each node
    for s in start_order:
        if time.time() - start_time > time_limit:
            break
        path = [s]
        visited = {s}
        dfs(path, visited)
        
        # If we found a Hamiltonian path, we are done
        if best_len == n:
            break

    return best_path
