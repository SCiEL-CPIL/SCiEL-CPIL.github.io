import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80

    n = P['n']
    adj_list = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj_list[u].append(v)
        adj_list[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)

    def time_check():
        return time.time() > deadline

    # Dinic max-flow
    def dinic_max_flow(n_nodes, edges, cap, source, sink):
        """edges: list of (u, v); cap: dict or list. Returns (maxflow, parent dict for residual graph BFS)."""
        # Build adjacency for residual graph
        # We'll use a simple Dinic with adjacency lists
        adj = [[] for _ in range(n_nodes)]
        # edge list: [u, v, cap, rev_index]
        edge_list = []
        def add_edge(u, v, c):
            edge_list.append([u, v, c, len(adj[v])])
            edge_list.append([v, u, 0, len(adj[u])])
            adj[u].append(len(edge_list) - 2)
            adj[v].append(len(edge_list) - 1)
        
        for u, v in edges:
            c = cap if isinstance(cap, (int, float)) else cap.get((u, v), 0)
            if c > 0:
                add_edge(u, v, c)
        
        flow = 0
        INF = float('inf')
        
        while True:
            level = [-1] * n_nodes
            level[source] = 0
            q = deque([source])
            while q:
                u = q.popleft()
                for ei in adj[u]:
                    e = edge_list[ei]
                    v = e[1]
                    if e[2] > 0 and level[v] < 0:
                        level[v] = level[u] + 1
                        q.append(v)
            if level[sink] < 0:
                break
            it = [0] * n_nodes
            def dfs(u, f):
                if u == sink:
                    return f
                while it[u] < len(adj[u]):
                    ei = adj[u][it[u]]
                    e = edge_list[ei]
                    v = e[1]
                    if e[2] > 0 and level[v] == level[u] + 1:
                        ret = dfs(v, min(f, e[2]))
                        if ret > 0:
                            e[2] -= ret
                            edge_list[e[3]][2] += ret
                            return ret
                    it[u] += 1
                return 0
            while True:
                pushed = dfs(source, INF)
                if pushed == 0:
                    break
                flow += pushed
        
        return flow

    def find_paths_for_branches(branches):
        """Try to find disjoint internal paths for all pairs of branches.
        Returns dict of (u,v) -> path, or None if infeasible."""
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Identify direct edges and compute options for non-direct pairs
        pair_info = {}
        non_direct_pairs = []
        for u, v in pairs:
            if u in adj_set[v]:
                pair_info[(u, v)] = ('direct', None)
            else:
                # Find internal vertices adjacent to both (for length-2 paths)
                common = set()
                for w in adj_list[u]:
                    if w not in branch_set and w in adj_set[v]:
                        common.add(w)
                pair_info[(u, v)] = ('need', common)
                non_direct_pairs.append((u, v))
        
        # Feasibility: every non-direct pair must have at least one common neighbor
        for pair in non_direct_pairs:
            if len(pair_info[pair][1]) == 0:
                return None
        
        # Try to assign distinct internal vertices via matching
        if non_direct_pairs:
            assignment = {}
            used = set()
            # Sort by fewest options first (MRV heuristic)
            sorted_pairs = sorted(non_direct_pairs, key=lambda p: len(pair_info[p][1]))
            
            # Greedy first
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
                    # Try with longer paths via BFS as fallback
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
                        
                        # BFS shortest path
                        parent = {u: -1}
                        queue = deque([u])
                        found = False
                        while queue and not found:
                            node = queue.popleft()
                            for neighbor in adj_list[node]:
                                if neighbor not in parent and neighbor not in blocked:
                                    parent[neighbor] = node
                                    if neighbor == v:
                                        found = True
                                        break
                                    queue.append(neighbor)
                        
                        if not found:
                            return False
                        
                        path = []
                        cur = v
                        while cur != -1:
                            path.append(cur)
                            cur = parent[cur]
                        path.reverse()
                        
                        internal = path[1:-1]
                        
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
        
        # Sort vertices by degree descending for better pruning
        degree = [len(adj_list[i]) for i in range(n)]
        sorted_vertices = sorted(range(n), key=lambda x: -degree[x])
        
        current_branches = []
        
        def backtrack_clique(candidate_start_idx):
            nonlocal best_result, best_k
            
            if time_check():
                return
            
            # Update best if current set is valid and larger
            if len(current_branches) > best_k:
                paths = find_paths_for_branches(current_branches)
                if paths is not None:
                    best_result = {
                        'branches': [int(b) for b in current_branches],
                        'paths': [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
                    }
                    best_k = len(current_branches)
            
            # Try adding more vertices
            for i in range(candidate_start_idx, n):
                if time_check():
                    return
                
                candidate = sorted_vertices[i]
                
                # Pruning: if we can't beat best_k even with all remaining, skip
                remaining = n - i
                if len(current_branches) + remaining <= best_k:
                    break
                
                if candidate in current_branches:
                    continue
                
                # Try adding candidate
                current_branches.append(candidate)
                
                # Verify feasibility of the new set
                paths = find_paths_for_branches(current_branches)
                if paths is not None:
                    # Feasible! Recurse to try adding more
                    backtrack_clique(i + 1)
                
                current_branches.pop()
        
        # Start backtracking from empty set
        backtrack_clique(0)
        
        if best_result is not None:
            return best_result
        
        # Fallback: find at least 2 branches
        for u, v in P['edges']:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        
        return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    return solve()
