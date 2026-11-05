import time
from collections import deque

def construct():
    n = 100
    edges = [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    visited_comp = [False] * n
    components = []
    for i in range(n):
        if not visited_comp[i]:
            comp = []
            stack = [i]
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited_comp[neighbor]:
                        visited_comp[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)
            
    best_path = []
    deadline = time.time() + 80.0
    
    for comp in components:
        if time.time() > deadline:
            break
            
        comp_set = set(comp)
        m_comp = sum(len(adj[u] & comp_set) for u in comp) // 2
        
        if m_comp == len(comp) - 1:
            # Tree
            def bfs_farthest(start):
                dist = {start: 0}
                parent = {start: None}
                queue = deque([start])
                farthest = start
                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        if v in comp_set and v not in dist:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                            if dist[v] > dist[farthest]:
                                farthest = v
                path = []
                curr = farthest
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                return path[::-1]
            
            p1 = bfs_farthest(comp[0])
            p2 = bfs_farthest(p1[-1])
            if len(p2) > len(best_path):
                best_path = p2
                
        elif m_comp == len(comp):
            # Unicyclic
            parent = {v: -1 for v in comp}
            in_cycle = set()
            
            def find_cycle(start):
                color = {v: 0 for v in comp}
                cycle = []
                def dfs(u):
                    color[u] = 1
                    for v in sorted(adj[u]):
                        if v not in comp_set: continue
                        if color[v] == 0:
                            parent[v] = u
                            if dfs(v): return True
                        elif color[v] == 1:
                            cycle.append(v)
                            curr = u
                            while curr != v:
                                cycle.append(curr)
                                curr = parent[curr]
                            cycle.append(v)
                            return True
                    color[u] = 2
                    return False
                dfs(start)
                return cycle
            
            cycle = find_cycle(comp[0])
            if not cycle:
                pass
            else:
                in_cycle = set(cycle)
                heights = {}
                for c in in_cycle:
                    depth = 0
                    stack = [(c, 0)]
                    visited_h = {c}
                    while stack:
                        u, d = stack.pop()
                        depth = max(depth, d)
                        for v in adj[u]:
                            if v in comp_set and v not in in_cycle and v not in visited_h:
                                visited_h.add(v)
                                stack.append((v, d+1))
                    heights[c] = depth
                
                cycle_ordered = []
                if cycle:
                    start_c = cycle[0]
                    cycle_ordered = [start_c]
                    curr = start_c
                    for _ in range(len(cycle)-1):
                        found = False
                        for v in sorted(adj[curr]):
                            if v in in_cycle and v not in cycle_ordered:
                                cycle_ordered.append(v)
                                curr = v
                                found = True
                                break
                        if not found: break
                
                k = len(cycle_ordered)
                best_len = -1
                best_i, best_j, best_dir = -1, -1, 1
                
                for i in range(k):
                    for j in range(k):
                        if i == j: continue
                        dist_f = (j - i) % k
                        dist_b = k - dist_f
                        
                        len_f = heights[cycle_ordered[i]] + dist_f + heights[cycle_ordered[j]] + 1
                        if len_f > best_len:
                            best_len = len_f
                            best_i, best_j, best_dir = i, j, 1
                            
                        len_b = heights[cycle_ordered[i]] + dist_b + heights[cycle_ordered[j]] + 1
                        if len_b > best_len:
                            best_len = len_b
                            best_i, best_j, best_dir = i, j, -1
                            
                if best_i >= 0:
                    ci = cycle_ordered[best_i]
                    cj = cycle_ordered[best_j]
                    
                    def get_tree_path(c):
                        best_leaf = c
                        best_depth = 0
                        parent3 = {c: -1}
                        queue = deque([c])
                        visited3 = {c}
                        while queue:
                            u = queue.popleft()
                            for v in adj[u]:
                                if v in comp_set and v not in in_cycle and v not in visited3:
                                    visited3.add(v)
                                    parent3[v] = u
                                    queue.append(v)
                        for v in visited3:
                            if v == c: continue
                            depth = 0
                            curr = v
                            while curr != c:
                                depth += 1
                                curr = parent3[curr]
                            if depth > best_depth:
                                best_depth = depth
                                best_leaf = v
                                
                        path = []
                        curr = best_leaf
                        while curr != -1:
                            path.append(curr)
                            if curr == c: break
                            curr = parent3[curr]
                        return path[::-1]
                    
                    tree_i = get_tree_path(ci)
                    tree_j = get_tree_path(cj)
                    
                    if best_dir == 1:
                        seg = []
                        idx = best_i
                        while True:
                            seg.append(cycle_ordered[idx])
                            if idx == best_j: break
                            idx = (idx + 1) % k
                    else:
                        seg = []
                        idx = best_i
                        while True:
                            seg.append(cycle_ordered[idx])
                            if idx == best_j: break
                            idx = (idx - 1) % k
                            
                    full_path = tree_i[:-1] + seg + tree_j[1:]
                    if len(full_path) > len(best_path):
                        best_path = full_path

        else:
            # Multi-cyclic B&B
            comp_list = list(comp)
            comp_index = {v: i for i, v in enumerate(comp_list)}
            m = len(comp_list)
            
            adj_comp = [set() for _ in range(m)]
            for u in comp:
                for v in adj[u]:
                    if v in comp_set:
                        adj_comp[comp_index[u]].add(comp_index[v])
            
            # Greedy LB
            def greedy_path(start_idx):
                path = [start_idx]
                visited = {start_idx}
                curr = start_idx
                while True:
                    neighbors = [v for v in adj_comp[curr] if v not in visited]
                    if not neighbors: break
                    neighbors.sort(key=lambda v: len(adj_comp[v]))
                    next_node = neighbors[0]
                    path.append(next_node)
                    visited.add(next_node)
                    curr = next_node
                return path
            
            init_lb = 0
            init_path = []
            for i in range(m):
                p = greedy_path(i)
                if len(p) > init_lb:
                    init_lb = len(p)
                    init_path = p
            if init_lb > len(best_path):
                best_path = [comp_list[i] for i in init_path]
                
            best_len_bb = init_lb
            best_path_bb_indices = init_path[:]
            
            def dfs(u, visited_mask, path_indices, length):
                nonlocal best_len_bb, best_path_bb_indices, deadline
                if time.time() > deadline:
                    return
                
                if length > best_len_bb:
                    best_len_bb = length
                    best_path_bb_indices = path_indices[:]
                    
                # Prune
                if length + (m - length) <= best_len_bb:
                    return
                    
                neighbors = sorted([v for v in adj_comp[u] if not (visited_mask & (1 << v))], 
                                   key=lambda v: len(adj_comp[v]))
                
                for v in neighbors:
                    visited_mask |= (1 << v)
                    path_indices.append(v)
                    dfs(v, visited_mask, path_indices, length + 1)
                    path_indices.pop()
                    visited_mask ^= (1 << v)
                    
                    if time.time() > deadline:
                        return
            
            for i in range(m):
                if time.time() > deadline:
                    break
                if best_len_bb == m:
                    break
                dfs(i, 1 << i, [i], 1)
                
            if best_len_bb > len(best_path):
                best_path = [comp_list[i] for i in best_path_bb_indices]

    if not best_path:
        return [0]
        
    return best_path
