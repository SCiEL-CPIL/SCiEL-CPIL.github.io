"""Rewritten program for large clique subdivisions in a pinned graph."""
# EVOLVE-BLOCK-START
import itertools
import sys
from collections import defaultdict, deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Max Flow Implementation (Dinic's Algorithm)
    class Dinic:
        def __init__(self, n):
            self.n = n
            self.graph = [[] for _ in range(n)]
            
        def add_edge(self, u, v, cap):
            self.graph[u].append([v, cap, len(self.graph[v])])
            self.graph[v].append([u, 0, len(self.graph[u]) - 1])
            
        def bfs(self, s, t):
            self.level = [-1] * self.n
            self.level[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v, cap, rev in self.graph[u]:
                    if cap > 0 and self.level[v] < 0:
                        self.level[v] = self.level[u] + 1
                        q.append(v)
            return self.level[t] >= 0

        def dfs(self, u, t, f):
            if u == t:
                return f
            for i in range(self.it[u], len(self.graph[u])):
                self.it[u] = i
                v, cap, rev = self.graph[u][i]
                if cap > 0 and self.level[u] < self.level[v]:
                    d = self.dfs(v, t, min(f, cap))
                    if d > 0:
                        self.graph[u][i][1] -= d
                        self.graph[v][rev][1] += d
                        return d
            return 0

        def max_flow(self, s, t):
            flow = 0
            INF = 10**9
            while self.bfs(s, t):
                self.it = [0] * self.n
                f = self.dfs(s, t, INF)
                while f > 0:
                    flow += f
                    f = self.dfs(s, t, INF)
            return flow

    # Helper to check if two pairs have vertex-disjoint paths
    # Uses vertex splitting
    def check_disjoint_pairs(pair1, pair2, forbidden):
        """
        pair1: (u1, v1)
        pair2: (u2, v2)
        forbidden: set of vertices that cannot be used as interiors
        Returns True if there exist vertex-disjoint paths for both pairs.
        """
        # Nodes:
        # 0..n-1: v_in
        # n..2n-1: v_out
        # 2n: Source
        # 2n+1: Sink
        N = 2 * n + 2
        S = 2 * n
        T = 2 * n + 1
        
        dinic = Dinic(N)
        
        # Add edges for vertex splitting
        for v in range(n):
            if v in forbidden:
                # Capacity 0, effectively removed
                dinic.add_edge(v, n + v, 0)
            else:
                # Capacity 1 for internal vertices
                # Branch vertices are not in forbidden, but they are endpoints.
                # We handle endpoints separately.
                dinic.add_edge(v, n + v, 1)
                
        # Add graph edges
        for u in range(n):
            for v in adj[u]:
                if u < v: # Avoid duplicates
                    dinic.add_edge(n + u, v, 1)
                    dinic.add_edge(n + v, u, 1)
                    
        # Connect sources and sinks
        # Path 1: u1 -> v1
        # Path 2: u2 -> v2
        # We need to ensure flow from S goes to u1/u2 and from v1/v2 goes to T.
        # But we need to distinguish which flow goes where?
        # Standard max flow doesn't distinguish commodities.
        # This check is only valid if we just want *some* 2 disjoint paths from {u1,u2} to {v1,v2}.
        # This is NOT the same as disjoint paths for specific pairs.
        # So this function is NOT suitable for specific pair constraints.
        return False

    # Since the flow check for specific pairs is complex, we rely on backtracking
    # but use the flow structure to *find* paths? No.
    
    # We will use the backtracking approach with optimized heuristics.
    
    def find_max_clique():
        best = []
        def dfs(candidates, current):
            nonlocal best
            if len(current) > len(best):
                best = current[:]
            if not candidates:
                return
            if len(current) + len(candidates) <= len(best):
                return
            for i in range(len(candidates)):
                v = candidates[i]
                new_candidates = [c for c in candidates[i+1:] if c in adj[v]]
                dfs(new_candidates, current + [v])
        dfs(list(range(n)), [])
        return best

    max_clique = find_max_clique()
    k_clique = len(max_clique)
    
    if k_clique >= 6:
        branches = list(max_clique)
        paths = []
        for u, v in itertools.combinations(branches, 2):
            paths.append([u, v])
        return {'branches': branches, 'paths': paths}

    def find_subdivision(k, max_path_len=6):
        degrees = [len(adj[i]) for i in range(n)]
        sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        candidates = sorted_verts[:12]
        
        if k > len(candidates):
            return None
            
        for combo in itertools.combinations(candidates, k):
            branches = list(combo)
            pairs = list(itertools.combinations(branches, 2))
            
            candidate_paths = {}
            valid = True
            for u, v in pairs:
                paths_uv = []
                def dfs_paths(current, visited, path):
                    if len(path) - 1 > max_path_len:
                        return
                    if current == v and len(path) > 1:
                        paths_uv.append(path[:])
                        return
                    for neighbor in adj[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            path.append(neighbor)
                            dfs_paths(neighbor, visited, path)
                            path.pop()
                            visited.remove(neighbor)
                
                dfs_paths(u, {u}, [u])
                
                if not paths_uv:
                    valid = False
                    break
                candidate_paths[(u, v)] = paths_uv
            
            if not valid:
                continue
                
            pairs_sorted = sorted(pairs, key=lambda p: len(candidate_paths[p]))
            
            used_interiors = set()
            assigned_paths = {}
            
            def backtrack(idx):
                if idx == len(pairs_sorted):
                    return True
                u, v = pairs_sorted[idx]
                for path in candidate_paths[(u, v)]:
                    interior = set(path[1:-1])
                    if interior.isdisjoint(used_interiors) and interior.isdisjoint(branches):
                        used_interiors.update(interior)
                        assigned_paths[(u, v)] = path
                        if backtrack(idx + 1):
                            return True
                        used_interiors.difference_update(interior)
                        del assigned_paths[(u, v)]
                return False
            
            if backtrack(0):
                paths_list = []
                for u, v in pairs:
                    paths_list.append(assigned_paths[(u, v)])
                return branches, paths_list
                
        return None

    res = find_subdivision(6, max_path_len=6)
    if res:
        branches, paths = res
        return {'branches': branches, 'paths': paths}
        
    res = find_subdivision(5, max_path_len=6)
    if res:
        branches, paths = res
        return {'branches': branches, 'paths': paths}
        
    if k_clique >= 2:
        branches = list(max_clique)
        paths = []
        for u, v in itertools.combinations(branches, 2):
            paths.append([u, v])
        return {'branches': branches, 'paths': paths}
        
    u, v = edges[0]
    return {'branches': [u, v], 'paths': [[u, v]]}

# EVOLVE-BLOCK-END
