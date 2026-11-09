import time
from collections import deque
from itertools import combinations
import random

# P is defined in the environment, but we assume it's available or we need to handle it.
# Based on the prompt, P is available.
try:
    P
except NameError:
    # Fallback if P is not defined in this specific test context, though it should be.
    P = {'n': 24, 'edges': []}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    n = P['n']
    edges = P['edges']

    # Build adjacency lists
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    adj_list = [sorted(adj[i]) for i in range(n)]
    degrees = [len(adj[i]) for i in range(n)]

    def time_ok():
        return time.time() < deadline

    # ---------- Max-Flow for Vertex Disjoint Paths ----------
    # We implement a simple Edmonds-Karp or Dinic's algorithm.
    # Given sets U and V of vertices, find max number of vertex-disjoint paths from U to V.
    # Terminals in U and V have infinite capacity. Others have capacity 1.
    
    def max_vertex_disjoint_paths(U, V, forbidden):
        """
        U, V: sets of vertex indices.
        forbidden: set of vertices that cannot be used as interiors (can be endpoints if in U/V? No, U/V are the endpoints).
        Actually, the path must start in U, end in V, and interiors must not be in forbidden.
        Also interiors cannot be in U or V (except the endpoints themselves).
        """
        if not U or not V:
            return 0
        
        # Node mapping for flow network
        # Original node v -> v_in (2*v) and v_out (2*v + 1)
        # Total nodes: 2*n + 2 (source, sink)
        num_nodes = 2 * n + 2
        source = 2 * n
        sink = 2 * n + 1
        
        # Adjacency list for flow graph: (neighbor, capacity)
        # We'll use a simple BFS-based Edmonds-Karp for small graphs (n=24 is small)
        # But we need residual graph.
        
        # Initialize capacities
        # cap[u][v]
        # Using a dict of dicts for sparse graph
        cap = {}
        for i in range(num_nodes):
            cap[i] = {}
        
        def add_edge(u, v, c):
            if v not in cap[u]:
                cap[u][v] = 0
            cap[u][v] += c
            if u not in cap[v]:
                cap[v][u] = 0
            cap[v][u] += c # Reverse edge with 0 capacity initially, but we need to handle residual properly
            # Standard Edmonds-Karp uses residual. Let's just store forward and back.
            # Actually, simpler: store edges in a list and use matrix if n is small?
            # n=24, nodes=50. 50x50 matrix is fine.
            
        # Let's use a matrix for simplicity and speed for n=24
        # 50x50 matrix
        flow_cap = [[0] * num_nodes for _ in range(num_nodes)]
        
        # 1. Split vertices
        for v in range(n):
            if v in U or v in V:
                # Infinite capacity for terminals
                flow_cap[2*v][2*v+1] = 1000 
            else:
                if v in forbidden:
                    # Cannot be used at all
                    flow_cap[2*v][2*v+1] = 0
                else:
                    flow_cap[2*v][2*v+1] = 1
        
        # 2. Original edges
        for u, v in edges:
            # u_out -> v_in
            flow_cap[2*u+1][2*v] = 1
            flow_cap[2*v+1][2*u] = 1
            
        # 3. Connect Source to U
        for u in U:
            flow_cap[source][2*u] = 1000
            
        # 4. Connect V to Sink
        for v in V:
            flow_cap[2*v+1][sink] = 1000

        # Edmonds-Karp
        total_flow = 0
        while True:
            # BFS to find augmenting path
            parent = [-1] * num_nodes
            parent[source] = source
            queue = deque([source])
            found = False
            
            while queue and not found:
                u = queue.popleft()
                for v in range(num_nodes):
                    if parent[v] == -1 and flow_cap[u][v] > 0:
                        parent[v] = u
                        if v == sink:
                            found = True
                            break
                        queue.append(v)
            
            if not found:
                break
            
            # Find bottleneck
            path_flow = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, flow_cap[u][v])
                v = u
            
            # Update capacities
            v = sink
            while v != source:
                u = parent[v]
                flow_cap[u][v] -= path_flow
                flow_cap[v][u] += path_flow
                v = u
                
            total_flow += path_flow
            
        return total_flow

    # ---------- Check Cut Condition for a Branch Set ----------
    # A set B of size k can support a K_k subdivision only if for every partition B = A U C (A, C non-empty, disjoint),
    # the max number of vertex-disjoint paths between A and C is at least |A| * |C|.
    # This is because each vertex in A must connect to each vertex in C via a distinct path.
    
    def check_cut_condition(B):
        k = len(B)
        if k < 2:
            return True
        
        B_set = set(B)
        # Check all non-trivial subsets A of B
        # To optimize, we only need to check subsets of size 1 to k//2
        for i in range(1, k // 2 + 1):
            if not time_ok():
                return False
            for A in combinations(B, i):
                A_set = set(A)
                C_set = B_set - A_set
                # Max disjoint paths between A_set and C_set
                # Forbidden: B_set (since paths must be internally disjoint from all branches)
                # Wait, the paths connecting A to C must not use any vertex in B as an interior.
                # So forbidden = B_set
                flow = max_vertex_disjoint_paths(A_set, C_set, B_set)
                if flow < i * (k - i):
                    return False
        return True

    # ---------- Build Subdivision (Greedy BFS with Backtracking) ----------
    # If cut condition passes, we try to actually build the paths.
    # We use a backtracking search for path assignment.
    
    def build_subdivision(branches):
        k = len(branches)
        if k < 2:
            return None
        
        branches_set = set(branches)
        pairs = [(branches[i], branches[j]) for i in range(k) for j in range(i+1, k)]
        
        # Order pairs: direct edges first, then by degree sum
        def pair_key(p):
            u, v = p
            if v in adj[u]:
                return (0, 0)
            return (1, -(degrees[u] + degrees[v]))
            
        pairs.sort(key=pair_key)
        
        # Backtracking
        used_interior = set()
        path_map = {}
        
        def backtrack(idx):
            if idx == len(pairs):
                return True
            if not time_ok():
                return False
                
            u, v = pairs[idx]
            a, b = min(u, v), max(u, v)
            key = (a, b)
            
            # Forbidden for this path: all branches except u, v, plus used interiors
            forbidden = branches_set - {u, v}
            forbidden_with_used = forbidden | used_interior
            
            # Find a path
            # We can try multiple paths if we want to be robust, but BFS shortest is a good start
            # Let's try BFS
            p = bfs_path(u, v, forbidden_with_used)
            
            if p is None:
                return False
            
            interior = set(p[1:-1])
            # Check for conflicts (should be handled by forbidden, but double check)
            if interior & used_interior:
                return False
            if interior & (branches_set - {u, v}):
                return False
                
            # Commit
            path_map[key] = p
            used_interior.update(interior)
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            used_interior.difference_update(interior)
            del path_map[key]
            
            return False

        # Try a few random orderings if the first one fails
        for restart in range(20):
            if not time_ok():
                return None
                
            if restart == 0:
                ordered_pairs = list(pairs)
            else:
                ordered_pairs = list(pairs)
                random.shuffle(ordered_pairs)
                
            used_interior.clear()
            path_map.clear()
            
            # We need a local backtracking that uses ordered_pairs
            # Let's just do a simple greedy with restarts for speed, as full backtracking is expensive
            success = True
            for u, v in ordered_pairs:
                a, b = min(u, v), max(u, v)
                key = (a, b)
                forbidden = branches_set - {u, v}
                forbidden_with_used = forbidden | used_interior
                p = bfs_path(u, v, forbidden_with_used)
                if p is None:
                    success = False
                    break
                interior = set(p[1:-1])
                if interior & used_interior or interior & (branches_set - {u, v}):
                    success = False
                    break
                path_map[key] = p
                used_interior.update(interior)
                
            if success:
                output_paths = []
                for i in range(k):
                    for j in range(i+1, k):
                        a, b = branches[i], branches[j]
                        key = (min(a, b), max(a, b))
                        p = path_map[key]
                        if p[0] == branches[i]:
                            output_paths.append(p)
                        else:
                            output_paths.append(p[::-1])
                return output_paths
                
        return None

    # ---------- BFS ----------
    def bfs_path(s, t, forbidden):
        if s == t:
            return [s]
        if t in adj[s]:
            return [s, t]
        
        visited = {s}
        parent = {s: None}
        queue = deque([s])
        while queue:
            cur = queue.popleft()
            for nb in adj_list[cur]:
                if nb == t:
                    path = [t]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                if nb in forbidden:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = cur
                    queue.append(nb)
        return None

    # ---------- Find Max Clique ----------
    def find_max_clique():
        best = []
        
        def bk(R, Pset, X):
            nonlocal best
            if not time_ok():
                return
            if not Pset and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            
            if Pset:
                pivot_candidates = list(Pset | X)
            else:
                pivot_candidates = list(X)
            
            if not pivot_candidates:
                return
            
            u = max(pivot_candidates, key=lambda v: len(adj[v] & Pset))
            candidates = [v for v in Pset if v not in adj[u]]
            candidates.sort(key=lambda x: -degrees[x])
            
            for v in candidates:
                if not time_ok():
                    return
                bk(R | {v}, Pset & adj[v], X & adj[v])
                Pset = Pset - {v}
                X = X | {v}
        
        bk(frozenset(), frozenset(range(n)), frozenset())
        return best

    # ---------- Main Search ----------
    max_clique = find_max_clique()
    clique_list = sorted(max_clique)
    
    best_result = None
    best_k = 0
    
    # Try subsets of the max clique, starting from largest
    for size in range(len(clique_list), 1, -1):
        if not time_ok():
            break
            
        found_valid = False
        # If size is small, try all subsets. If large, try random subsets.
        if size <= 10:
            subsets = list(combinations(clique_list, size))
        else:
            # Sample random subsets
            subsets = []
            for _ in range(50):
                if not time_ok():
                    break
                sub = random.sample(clique_list, size)
                subsets.append(tuple(sorted(sub)))
            
        for sub in subsets:
            if not time_ok():
                break
            sub_list = list(sub)
            
            # 1. Check Cut Condition (Structural Filter)
            if check_cut_condition(sub_list):
                # 2. Try to Build Paths
                paths = build_subdivision(sub_list)
                if paths is not None:
                    best_result = (sub_list, paths)
                    best_k = size
                    found_valid = True
                    break
                    
        if found_valid:
            break

    # If we didn't find a valid set from the clique, try to grow from a small valid set
    # Start with a pair
    if best_k < 2:
        # Find any edge
        if edges:
            u, v = edges[0]
            best_result = ([int(u), int(v)], [[int(u), int(v)]])
            best_k = 2
        else:
            # Should not happen for connected graph
            best_result = ([0, 1], [[0, 1]])
            best_k = 2

    # Try to improve by adding vertices
    if best_result is not None and time_ok():
        current_branches = list(best_result[0])
        current_paths = best_result[1]
        improved = True
        while improved and time_ok():
            improved = False
            # Try adding vertices from the max clique first, then others
            candidates = [v for v in range(n) if v not in set(current_branches)]
            # Sort by degree
            candidates.sort(key=lambda x: -degrees[x])
            
            for v in candidates:
                if not time_ok():
                    break
                new_branches = sorted(current_branches + [v])
                
                # Check cut condition
                if check_cut_condition(new_branches):
                    new_paths = build_subdivision(new_branches)
                    if new_paths is not None:
                        current_branches = new_branches
                        current_paths = new_paths
                        best_result = (current_branches, current_paths)
                        best_k = len(current_branches)
                        improved = True
                        break

    # ---------- Build Output ----------
    branches = [int(b) for b in best_result[0]]
    paths_out = [[int(v) for v in p] for p in best_result[1]]

    return {
        'branches': branches,
        'paths': paths_out
    }

if __name__ == '__main__':
    # For local testing
    import json
    # P = json.load(open('input.json'))
    res = construct()
    print(json.dumps(res))
