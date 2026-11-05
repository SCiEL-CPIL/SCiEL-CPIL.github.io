import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)
            
    # Sort components by size, largest first
    components.sort(key=len, reverse=True)
    
    best_path = []
    deadline = time.time() + 85.0
    
    for comp in components:
        if time.time() > deadline:
            break
            
        comp_set = set(comp)
        m_comp = 0
        for u in comp:
            m_comp += len(adj[u] & comp_set)
        m_comp //= 2
        
        # Exact solvers for small/simple structures
        if m_comp == len(comp) - 1:
            # Tree case
            def bfs_farthest(start, comp_set, adj):
                dist = {start: 0}
                parent = {start: None}
                queue = deque([start])
                farthest = start
                max_dist = 0
                while queue:
                    u = queue.popleft()
                    if dist[u] > max_dist:
                        max_dist = dist[u]
                        farthest = u
                    for v in adj[u]:
                        if v in comp_set and v not in dist:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                path = []
                curr = farthest
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                return path[::-1]
            
            start_node = comp[0]
            path1 = bfs_farthest(start_node, comp_set, adj)
            end_node = path1[-1]
            path2 = bfs_farthest(end_node, comp_set, adj)
            
            if len(path2) > len(best_path):
                best_path = path2
                
        elif m_comp == len(comp):
            # Unicyclic graph case
            # Find cycle
            in_cycle = set()
            parent = {v: -1 for v in comp}
            
            def find_cycle(start, adj, comp_set):
                color = {v: 0 for v in comp_set}
                cycle = []
                
                def dfs(u):
                    color[u] = 1
                    for v in sorted(adj[u]):
                        if v not in comp_set:
                            continue
                        if color[v] == 0:
                            parent[v] = u
                            if dfs(v):
                                return True
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
            
            cycle = find_cycle(comp[0], adj, comp_set)
            
            if cycle:
                in_cycle = set(cycle)
                
                # Compute heights
                heights = {}
                for c in in_cycle:
                    h = 0
                    parent2 = {c: -1}
                    queue = deque([c])
                    while queue:
                        u = queue.popleft()
                        for v in adj[u]:
                            if v in comp_set and v not in in_cycle and v not in parent2:
                                parent2[v] = u
                                queue.append(v)
                    for v in parent2:
                        if v == c: continue
                        depth = 0
                        curr = v
                        while curr != c:
                            depth += 1
                            curr = parent2[curr]
                        h = max(h, depth)
                    heights[c] = h
                
                # Order cycle
                cycle_ordered = []
                if len(cycle) > 0:
                    start_c = cycle[0]
                    cycle_ordered = [start_c]
                    curr = start_c
                    for _ in range(len(cycle) - 1):
                        found = False
                        for v in sorted(adj[curr]):
                            if v in in_cycle and v not in cycle_ordered:
                                cycle_ordered.append(v)
                                curr = v
                                found = True
                                break
                        if not found:
                            break
                
                if len(cycle_ordered) < 2:
                    cycle_ordered = list(in_cycle)
                
                k = len(cycle_ordered)
                best_len = -1
                best_i, best_j = -1, -1
                best_dir = 1
                
                for i in range(k):
                    for j in range(k):
                        if i == j: continue
                        dist_f = (j - i) % k
                        dist_b = k - dist_f
                        
                        len_f = heights[cycle_ordered[i]] + dist_f + heights[cycle_ordered[j]] + 1
                        if len_f > best_len:
                            best_len = len_f
                            best_i, best_j = i, j
                            best_dir = 1
                        
                        len_b = heights[cycle_ordered[i]] + dist_b + heights[cycle_ordered[j]] + 1
                        if len_b > best_len:
                            best_len = len_b
                            best_i, best_j = i, j
                            best_dir = -1
                
                if best_i >= 0:
                    ci = cycle_ordered[best_i]
                    cj = cycle_ordered[best_j]
                    
                    def get_tree_path(c, adj, comp_set, in_cycle):
                        best_leaf = c
                        best_depth = 0
                        parent3 = {c: -1}
                        queue = deque([c])
                        while queue:
                            u = queue.popleft()
                            for v in adj[u]:
                                if v in comp_set and v not in in_cycle and v not in parent3:
                                    parent3[v] = u
                                    queue.append(v)
                        for v in parent3:
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
                        path.reverse()
                        return path
                    
                    tree_i = get_tree_path(ci, adj, comp_set, in_cycle)
                    tree_j = get_tree_path(cj, adj, comp_set, in_cycle)
                    
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
            # Giant Component: Use Simulated Annealing
            # Initial path: Greedy Warnsdorff
            def greedy_start(start_node):
                path = [start_node]
                visited = {start_node}
                current = start_node
                while True:
                    neighbors = [v for v in adj[current] if v in comp_set and v not in visited]
                    if not neighbors:
                        break
                    # Warnsdorff: pick neighbor with fewest unvisited neighbors
                    neighbors.sort(key=lambda v: len([n for n in adj[v] if n in comp_set and n not in visited]))
                    next_node = neighbors[0]
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                return path

            # Get an initial good path
            # Try a few starts
            init_paths = []
            for s in comp[:5]: # Try first 5 nodes
                init_paths.append(greedy_start(s))
            current_path = max(init_paths, key=len)
            
            # Simulated Annealing
            T = 10.0
            T_min = 0.01
            cooling = 0.99
            current_energy = -len(current_path)
            best_path_sa = current_path[:]
            best_energy = current_energy
            
            path_set = set(current_path)
            
            while time.time() < deadline and T > T_min:
                # Generate neighbor
                new_path = current_path[:]
                move_type = random.choice(['add', 'remove', 'reverse'])
                
                if move_type == 'add':
                    # Try to add to start or end
                    if random.random() < 0.5:
                        end_node = new_path[0]
                        candidates = [v for v in adj[end_node] if v in comp_set and v not in path_set]
                        if candidates:
                            new_node = random.choice(candidates)
                            new_path.insert(0, new_node)
                            path_set.add(new_node)
                        else:
                            continue
                    else:
                        end_node = new_path[-1]
                        candidates = [v for v in adj[end_node] if v in comp_set and v not in path_set]
                        if candidates:
                            new_node = random.choice(candidates)
                            new_path.append(new_node)
                            path_set.add(new_node)
                        else:
                            continue
                            
                elif move_type == 'remove':
                    if random.random() < 0.5:
                        if len(new_path) > 1:
                            removed = new_path.pop(0)
                            path_set.discard(removed)
                        else:
                            continue
                    else:
                        if len(new_path) > 1:
                            removed = new_path.pop(-1)
                            path_set.discard(removed)
                        else:
                            continue
                            
                else: # reverse
                    if len(new_path) > 2:
                        i = random.randint(0, len(new_path) - 2)
                        j = random.randint(i + 1, len(new_path) - 1)
                        new_path[i:j+1] = new_path[i:j+1][::-1]
                        # path_set remains same
                    else:
                        continue

                new_energy = -len(new_path)
                delta = new_energy - current_energy
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_path = new_path
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_path_sa = new_path[:]
                
                T *= cooling
            
            if len(best_path_sa) > len(best_path):
                best_path = best_path_sa

    if not best_path:
        return [0]
    
    return best_path
# EVOLVE-BLOCK-END
