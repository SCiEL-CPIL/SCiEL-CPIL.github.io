import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s margin before 90s kill
    
    n = P['n']
    # Build adjacency
    adj_list = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj_list[u].append(v)
        adj_list[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)
    
    def time_check():
        return time.time() > deadline
    
    def bfs_shortest(src, dst, blocked):
        """Find shortest simple path from src to dst avoiding blocked vertices (except src and dst)."""
        if src == dst:
            return [src]
        parent = {src: -1}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if node == dst:
                path = []
                cur = dst
                while cur != -1:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for neighbor in adj_list[node]:
                if neighbor not in parent and neighbor not in blocked:
                    parent[neighbor] = node
                    queue.append(neighbor)
        return None
    
    def max_flow_disjoint_paths(branches):
        """Use max-flow to find vertex-disjoint paths between all pairs of branches."""
        k = len(branches)
        if k < 2:
            return None
        
        branch_set = set(branches)
        num_pairs = k * (k - 1) // 2
        
        # Node indexing: for vertex v, v_in = 2*v, v_out = 2*v+1
        # S = 2*n, T = 2*n+1
        S = 2 * n
        T = 2 * n + 1
        total_nodes = 2 * n + 2
        
        # Build flow network
        cap = [dict() for _ in range(total_nodes)]
        flow = [dict() for _ in range(total_nodes)]
        
        def add_edge(u, v, c):
            if v not in cap[u]:
                cap[u][v] = 0
                cap[v][u] = 0
                flow[u][v] = 0
                flow[v][u] = 0
            cap[u][v] += c
        
        # Add vertex split edges
        for v in range(n):
            v_in = 2 * v
            v_out = 2 * v + 1
            if v in branch_set:
                add_edge(v_in, v_out, k - 1)
            else:
                add_edge(v_in, v_out, 1)
        
        # Add original graph edges
        INF = num_pairs + 1
        for u, v in P['edges']:
            u_out = 2 * u + 1
            v_in = 2 * v
            v_out = 2 * v + 1
            u_in = 2 * u
            add_edge(u_out, v_in, INF)
            add_edge(v_out, u_in, INF)
        
        # Add source and sink edges
        for b in branches:
            b_in = 2 * b
            b_out = 2 * b + 1
            add_edge(S, b_in, k - 1)
            add_edge(b_out, T, k - 1)
        
        # Edmonds-Karp (BFS-based Ford-Fulkerson)
        def bfs_find_path():
            parent = {S: -1}
            queue = deque([S])
            while queue:
                node = queue.popleft()
                if node == T:
                    path = []
                    cur = T
                    while cur != -1:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    return path
                for neighbor, c in cap[node].items():
                    if neighbor not in parent and c - flow[node][neighbor] > 0:
                        parent[neighbor] = node
                        queue.append(neighbor)
            return None
        
        max_flow_val = 0
        while True:
            if time_check():
                return None
            path = bfs_find_path()
            if path is None:
                break
            bottleneck = float('inf')
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                residual = cap[u][v] - flow[u][v]
                if residual < bottleneck:
                    bottleneck = residual
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                flow[u][v] += bottleneck
                flow[v][u] -= bottleneck
            max_flow_val += bottleneck
        
        if max_flow_val != num_pairs:
            return None
        
        # Decompose flow into individual paths
        # Copy flow for decomposition
        decomp = [dict() for _ in range(total_nodes)]
        for u in range(total_nodes):
            for v, f in flow[u].items():
                if f > 0:
                    decomp[u][v] = f
        
        paths_found = []
        
        for _ in range(num_pairs):
            if time_check():
                return None
            # Find path from S to T following positive flow using DFS
            visited = set()
            
            def dfs_decompose(node):
                if node == T:
                    return [node]
                visited.add(node)
                for neighbor, f in decomp[node].items():
                    if f > 0 and neighbor not in visited:
                        result = dfs_decompose(neighbor)
                        if result is not None:
                            return [node] + result
                return None
            
            path_nodes = dfs_decompose(S)
            if path_nodes is None:
                return None
            
            # Reduce flow along this path
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i+1]
                decomp[u][v] -= 1
            
            # Convert to vertex path
            vertex_path = []
            for node in path_nodes:
                if node == S or node == T:
                    continue
                v = node // 2
                if not vertex_path or vertex_path[-1] != v:
                    vertex_path.append(v)
            
            if len(vertex_path) >= 2:
                paths_found.append(vertex_path)
        
        if len(paths_found) != num_pairs:
            return None
        
        # Verify and build result
        result_paths = {}
        used_internal = set()
        
        for path in paths_found:
            u, v = path[0], path[-1]
            if u not in branch_set or v not in branch_set:
                return None
            if u == v:
                return None
            key = (min(u, v), max(u, v))
            if key in result_paths:
                return None
            result_paths[key] = path
            
            internal = set(path[1:-1])
            for w in internal:
                if w in branch_set:
                    return None
                if w in used_internal:
                    return None
            used_internal.update(internal)
            
            for i in range(len(path) - 1):
                if path[i+1] not in adj_set[path[i]]:
                    return None
        
        for u, v in combinations(branches, 2):
            key = (min(u, v), max(u, v))
            if key not in result_paths:
                return None
        
        return result_paths
    
    def find_paths_for_branches(branches):
        """Try to find disjoint internal paths for all pairs of branches."""
        # Try max-flow first
        result = max_flow_disjoint_paths(branches)
        if result is not None:
            return result
        
        # Fallback: greedy matching for length-2 paths, then BFS for longer
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        pair_info = {}
        non_direct_pairs = []
        for u, v in pairs:
            if u in adj_set[v]:
                pair_info[(u, v)] = ('direct', None)
            else:
                common = set()
                for w in adj_list[u]:
                    if w not in branch_set and w in adj_set[v]:
                        common.add(w)
                pair_info[(u, v)] = ('need', common)
                non_direct_pairs.append((u, v))
        
        for pair in non_direct_pairs:
            if len(pair_info[pair][1]) == 0:
                return None
        
        if non_direct_pairs:
            assignment = {}
            used = set()
            sorted_pairs = sorted(non_direct_pairs, key=lambda p: len(pair_info[p][1]))
            
            success = True
            for pair in sorted_pairs:
                assigned = False
                for w in pair_info[pair][1]:
                    if w not in used:
                        used.add(w)
                        assignment[pair] = [w]
                        assigned = True
                        break
                if not assigned:
                    success = False
                    break
            
            if not success:
                # Backtracking matching
                assignment = {}
                used = set()
                
                def backtrack_match(idx):
                    if time_check():
                        return False
                    if idx == len(sorted_pairs):
                        return True
                    pair = sorted_pairs[idx]
                    for w in pair_info[pair][1]:
                        if w not in used:
                            used.add(w)
                            assignment[pair] = [w]
                            if backtrack_match(idx + 1):
                                return True
                            used.remove(w)
                            del assignment[pair]
                    return False
                
                if not backtrack_match(0):
                    # Try with longer paths via BFS
                    assignment = {}
                    used_internals = set()
                    sorted_non_direct = sorted(non_direct_pairs, key=lambda p: len(pair_info[p][1]))
                    
                    def backtrack_paths(idx):
                        if time_check():
                            return False
                        if idx == len(sorted_non_direct):
                            return True
                        u, v = sorted_non_direct[idx]
                        
                        blocked = branch_set | used_internals
                        blocked.discard(u)
                        blocked.discard(v)
                        
                        path = bfs_shortest(u, v, blocked)
                        if path is None:
                            return False
                        
                        internal = path[1:-1]
                        
                        if len(internal) != len(set(internal)):
                            return False
                        for w in internal:
                            if w in branch_set:
                                return False
                        
                        for w in internal:
                            used_internals.add(w)
                        assignment[(u, v)] = internal
                        
                        if backtrack_paths(idx + 1):
                            return True
                        
                        for w in internal:
                            used_internals.discard(w)
                        del assignment[(u, v)]
                        
                        return False
                    
                    if not backtrack_paths(0):
                        return None
        else:
            assignment = {}
        
        result_paths = {}
        for pair in pairs:
            u, v = pair
            if pair_info[pair][0] == 'direct':
                result_paths[pair] = [u, v]
            else:
                internal = assignment[pair]
                result_paths[pair] = [u] + internal + [v]
        
        return result_paths
    
    def solve():
        """Find maximum number of branches using systematic backtracking."""
        best_result = None
        best_k = 0
        
        degree = [len(adj_list[i]) for i in range(n)]
        sorted_vertices = sorted(range(n), key=lambda x: -degree[x])
        
        current_branches = []
        
        def backtrack_clique(candidate_start_idx):
            nonlocal best_result, best_k
            
            if time_check():
                return
            
            if len(current_branches) > best_k:
                paths = find_paths_for_branches(current_branches)
                if paths is not None:
                    best_result = {
                        'branches': [int(b) for b in current_branches],
                        'paths': [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
                    }
                    best_k = len(current_branches)
            
            for i in range(candidate_start_idx, n):
                if time_check():
                    return
                
                candidate = sorted_vertices[i]
                
                remaining = n - i
                if len(current_branches) + remaining <= best_k:
                    break
                
                if candidate in current_branches:
                    continue
                
                current_branches.append(candidate)
                
                paths = find_paths_for_branches(current_branches)
                if paths is not None:
                    backtrack_clique(i + 1)
                
                current_branches.pop()
        
        backtrack_clique(0)
        
        if best_result is not None:
            return best_result
        
        for u, v in P['edges']:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        
        return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    return solve()
