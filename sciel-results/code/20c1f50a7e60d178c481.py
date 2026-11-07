"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and set for fast lookup
    adj = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    
    # Global deadline
    global_deadline = time.time() + 85.0
    
    # Dinic's algorithm for Max Flow
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
            queue = [s]
            while queue:
                u = queue.pop(0)
                for v, cap, rev in self.graph[u]:
                    if cap > 0 and self.level[v] < 0:
                        self.level[v] = self.level[u] + 1
                        if v == t:
                            return True
                        queue.append(v)
            return False
        
        def dfs(self, u, t, f):
            if u == t:
                return f
            for i in range(len(self.graph[u])):
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
            while self.bfs(s, t):
                while True:
                    f = self.dfs(s, t, float('inf'))
                    if f == 0:
                        break
                    flow += f
            return flow

    def check_feasibility(branches, used_internal, remaining_pairs, time_budget):
        """
        Check if remaining_pairs can be routed with vertex-disjoint paths
        avoiding used_internal and other branches.
        Returns True if feasible, False otherwise.
        """
        if not remaining_pairs:
            return True
        
        # Build flow network
        # Vertices: 0 to n-1 are original.
        # We split each vertex v into v_in (2*v) and v_out (2*v+1).
        # Total nodes in flow network: 2*n + 2 (source, sink)
        # Source: 2*n, Sink: 2*n+1
        
        N_flow = 2 * n + 2
        S = 2 * n
        T = 2 * n + 1
        
        dinic = Dinic(N_flow)
        
        # Add internal edges for all non-branch, non-used vertices
        branch_set = set(branches)
        forbidden_internal = used_internal | (branch_set - set()) # Branches are endpoints, not internal
        
        for v in range(n):
            if v in branch_set or v in used_internal:
                continue
            # v_in -> v_out with capacity 1
            dinic.add_edge(2*v, 2*v+1, 1)
        
        # Add graph edges
        for u, v in edges:
            # u_out -> v_in
            dinic.add_edge(2*u+1, 2*v, 1)
            # v_out -> u_in
            dinic.add_edge(2*v+1, 2*u, 1)
            
        # Connect sources and sinks for remaining pairs
        # For each branch u in remaining pairs, it needs to send flow to its partners.
        # We can just connect S to u_in with capacity = number of partners in remaining pairs
        # and v_out to T with capacity = number of partners in remaining pairs.
        
        # Count degrees in remaining pairs
        pair_counts = defaultdict(int)
        for u, v in remaining_pairs:
            pair_counts[u] += 1
            pair_counts[v] += 1
            
        for u, count in pair_counts.items():
            # u is a branch. Flow starts at u.
            # So S -> u_in? No, flow should start at u.
            # In our split graph, flow enters a vertex at v_in and leaves at v_out.
            # So for a source branch u, we connect S to u_in? Or u_out?
            # If we connect S to u_in, flow goes S->u_in->u_out->...
            # This works.
            dinic.add_edge(S, 2*u, count)
            # For a sink branch v, flow should end at v.
            # So ...->v_in->v_out->T.
            dinic.add_edge(2*v+1, T, count)
            
        required_flow = sum(counts // 2 for counts in pair_counts.values()) # Wait, sum of counts is 2 * num_pairs
        required_flow = len(remaining_pairs)
        
        max_flow_val = dinic.max_flow(S, T)
        
        return max_flow_val >= required_flow

    def find_paths_for_subset(branches, time_budget):
        """
        Find paths for a specific set of branches.
        Uses backtracking with flow-based pruning.
        """
        k = len(branches)
        if k < 2:
            return None
            
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Sort pairs by heuristic: fewer available internal neighbors first
        def pair_constraint(p):
            u, v = p
            # Count neighbors of u and v that are not branches and not used (initially none)
            # This is a static heuristic, good enough for ordering
            return len(adj_set[u] - branch_set) + len(adj_set[v] - branch_set)
            
        pairs_sorted = sorted(pairs, key=pair_constraint)
        
        best_result = None
        deadline = time.time() + time_budget
        
        # Precompute candidate paths for each pair, sorted by length
        # We only need paths that don't use other branches
        candidate_paths = {}
        for u, v in pairs:
            paths = []
            # DFS to find simple paths
            def dfs_path(current, target, path, visited):
                if len(paths) > 20: # Limit paths to keep memory/time low
                    return
                if current == target:
                    if len(path) > 1:
                        paths.append(list(path))
                    return
                for neighbor in adj[current]:
                    if neighbor == target:
                        dfs_path(neighbor, target, path + [neighbor], visited | {neighbor})
                    elif neighbor not in visited and neighbor not in branch_set:
                        dfs_path(neighbor, target, path + [neighbor], visited | {neighbor})
                        
            dfs_path(u, v, [u], {u})
            paths.sort(key=len)
            candidate_paths[(u, v)] = paths
            
        # If any pair has no path, return None
        for p in pairs:
            if not candidate_paths[p]:
                return None

        def solve(idx, used_internal, current_paths):
            nonlocal best_result
            if time.time() > deadline:
                return False
            
            if idx == len(pairs_sorted):
                best_result = current_paths
                return True
            
            u, v = pairs_sorted[idx]
            remaining_pairs = pairs_sorted[idx+1:]
            
            # Pruning: Check if remaining pairs are feasible with current used_internal
            # We pass used_internal to the feasibility checker
            if not check_feasibility(branches, used_internal, remaining_pairs, 0.1):
                return False
                
            candidates = candidate_paths[(u, v)]
            
            for path in candidates:
                internal_nodes = set(path[1:-1])
                # Check if internal nodes are free
                if internal_nodes & used_internal:
                    continue
                if internal_nodes & (branch_set - {u, v}):
                    continue
                
                new_used_internal = used_internal | internal_nodes
                new_paths = current_paths + [(u, v, path)]
                
                if solve(idx + 1, new_used_internal, new_paths):
                    return True
            
            return False
        
        if solve(0, set(), []):
            return [path for u, v, path in best_result]
        
        return None

    # Search strategy
    # Start with a pool of top-degree vertices
    candidate_indices = sorted(range(n), key=lambda i: degrees[i], reverse=True)
    
    best_solution = None
    best_k = 0
    
    # Try to find the maximum k
    # We know 8 works. Let's try 9, 10, etc.
    # If 9 fails, we return the best found (which might be 8 from a previous run or found here).
    # But we need to find a valid one for the returned k.
    
    # Let's iterate k from high to low, but we need to be careful with time.
    # Since 8 is known to work, we can try 10, 9. If they fail, we retry 8 with more time.
    
    for k in [10, 9, 8, 7, 6]:
        if time.time() > global_deadline - 5.0:
            break
            
        # Pool size: larger for smaller k
        pool_size = {10: 16, 9: 18, 8: 20, 7: 22, 6: 24}[k]
        candidates = candidate_indices[:pool_size]
        
        # Time budget per subset
        # Number of subsets C(pool, k)
        from math import comb
        num_subsets = comb(pool_size, k)
        remaining_time = global_deadline - time.time() - 2.0
        time_per_subset = min(remaining_time / max(num_subsets, 1), 1.5)
        time_per_subset = max(time_per_subset, 0.05)
        
        found = False
        # Shuffle candidates to randomize search order? Or use combinations directly.
        # Using combinations directly is deterministic.
        for subset in combinations(candidates, k):
            if time.time() > global_deadline - 2.0:
                break
                
            branches = list(subset)
            paths = find_paths_for_subset(branches, time_per_subset)
            
            if paths is not None:
                best_solution = {
                    'branches': branches,
                    'paths': paths
                }
                best_k = k
                found = True
                break
        
        if found:
            break
            
    if best_solution is None:
        # Fallback: Find a valid small clique subdivision
        # Try k=3, 4 with more time
        for k in [4, 3]:
            candidates = candidate_indices[:20]
            for subset in combinations(candidates, k):
                branches = list(subset)
                paths = find_paths_for_subset(branches, 1.0)
                if paths is not None:
                    best_solution = {
                        'branches': branches,
                        'paths': paths
                    }
                    break
            if best_solution:
                break
                
    if best_solution is None:
        # Absolute fallback
        u, v = edges[0]
        best_solution = {
            'branches': [u, v],
            'paths': [[u, v]]
        }
        
    # Ensure types are correct
    return {
        'branches': [int(b) for b in best_solution['branches']],
        'paths': [[int(x) for x in p] for p in best_solution['paths']]
    }

# EVOLVE-BLOCK-END
