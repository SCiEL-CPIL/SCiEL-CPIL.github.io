"""Solver for large clique subdivisions in a pinned graph."""

import time
import random
from itertools import combinations
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    DEADLINE = 50.0
    random.seed(42)

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def time_up():
        return time.time() - t_start > DEADLINE

    # Precompute edge indices for ILP
    edge_list = []
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_list.append((u, v))
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    def solve_ilp(branches):
        """
        Solves for vertex-disjoint paths between all pairs in branches.
        Returns a dict of paths if feasible, else None.
        """
        k = len(branches)
        if k < 2:
            return None
        
        pairs = list(combinations(branches, 2))
        m = len(pairs)
        
        # Variables: y[e, p] for each edge e and pair p
        # Total variables: len(edges) * m
        num_vars = len(edges) * m
        
        # Objective: Minimize sum of all y (to find shortest paths, though any path works)
        c = np.ones(num_vars)
        
        # Constraints:
        # 1. Flow conservation for each pair p and each vertex v
        #    sum(y_in) - sum(y_out) = b_v
        #    b_v = 1 if v is source, -1 if v is sink, 0 otherwise.
        #    This ensures a path exists.
        # 2. Vertex capacity for non-branch vertices
        #    sum(p of y incident to v) <= 1 for v not in branches
        #    Note: This is tricky with edge variables. 
        #    A simpler constraint for vertex disjointness is to use vertex variables x[v, p].
        #    Let's switch to vertex variables x[v, p] and edge variables y[e, p].
        #    x[v, p] = 1 if v is in path p.
        #    y[e, p] = 1 if edge e is in path p.
        #    
        #    Constraints:
        #    a) x[source, p] = 1, x[sink, p] = 1
        #    b) For v not in {source, sink}, x[v, p] >= 0.5 * (sum of incident y) ? No.
        #       Standard formulation:
        #       x[v, p] >= y[u, v, p] for all u
        #       x[v, p] <= sum(y[u, v, p] for all u)
        #       This is complex.
        #       
        #    Alternative: Just use edge variables and enforce disjointness via vertex capacity on edges?
        #    No, vertex capacity requires vertex variables.
        
        # Let's use a simplified approach:
        # Since N is small, we can use a backtracking search for paths if ILP is too slow to set up.
        # But ILP is more robust.
        
        # Let's try to construct the ILP matrix A.
        # This is computationally expensive to build dynamically for every trial.
        # Given the time limit, a heuristic search with verification might be faster to implement correctly.
        
        # Let's revert to a robust heuristic: 
        # 1. Use BFS to find shortest paths for each pair.
        # 2. Check for conflicts.
        # 3. If conflict, try next shortest path? (K-shortest paths).
        # This is complex.
        
        # Let's try a different structural change: 
        # Use the previous greedy code but add a VERIFICATION step at the end.
        # If the result is invalid, discard it and try a smaller subset or restart.
        # The previous code likely failed because it returned an invalid result.
        # If we verify, we can ensure validity.
        
        pass

    # Given the complexity of ILP setup in Python within 50s, and the risk of bugs,
    # I will implement a "Greedy with Verification and Backtracking" approach.
    # The previous code's logic was mostly correct, but perhaps it got stuck or returned invalid data.
    # I will add a strict verification function.
    
    def verify_solution(branches, paths_dict):
        bset = set(branches)
        if len(branches) < 2:
            return False
        
        used_interiors = set()
        
        # Check all pairs
        for u, v in combinations(branches, 2):
            if (u, v) not in paths_dict:
                return False
            path = paths_dict[(u, v)]
            
            # Check endpoints
            if path[0] != u or path[-1] != v:
                # Paths are undirected in storage? 
                # My previous code stored (u,v) and (v,u) with same path.
                # If path is [v, ..., u], then path[0] is v.
                # We should normalize or check both.
                if path[0] != v or path[-1] != u:
                    return False
                # Reverse for interior check consistency? No, interior is same.
            
            # Check interior
            interior = path[1:-1]
            for w in interior:
                if w in bset:
                    return False
                if w in used_interiors:
                    return False
            
            used_interiors.update(interior)
            
            # Check adjacency
            for i in range(len(path) - 1):
                if path[i+1] not in adj[path[i]]:
                    return False
            
        return True

    def find_path_bfs(src, dst, forbidden_interior):
        """BFS from src to dst, interior vertices avoid forbidden_interior."""
        if src == dst:
            return [src]
        parent = {src: None}
        queue = [src]
        while queue:
            cur = queue.pop(0)
            if cur == dst:
                path = []
                node = dst
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in sorted(adj[cur]):
                if nb not in parent:
                    if nb != dst and nb in forbidden_interior:
                        continue
                    parent[nb] = cur
                    queue.append(nb)
        return None

    def try_add_vertex(branches, paths_dict, new_v):
        if time_up():
            return None
            
        bset = set(branches)
        forbidden = set()
        for p in paths_dict.values():
            forbidden.update(p[1:-1])
        forbidden.update(bset)
        
        targets = list(branches)
        # Sort targets by distance to new_v to prioritize easy connections?
        # Or just random order.
        random.shuffle(targets)
        
        new_paths = dict(paths_dict)
        used_interiors_new = set(forbidden)
        success = True
        temp_paths = {}
        
        for target in targets:
            path = find_path_bfs(new_v, target, used_interiors_new)
            if path is None:
                success = False
                break
            
            interior = set(path[1:-1])
            if interior & used_interiors_new:
                success = False
                break
            
            used_interiors_new |= interior
            temp_paths[(new_v, target)] = path
            temp_paths[(target, new_v)] = path

        if success:
            new_branches = branches + [new_v]
            for target in targets:
                new_paths[(new_v, target)] = temp_paths[(new_v, target)]
                new_paths[(target, new_v)] = temp_paths[(new_v, target)]
            
            # VERIFY the new solution
            if verify_solution(new_branches, new_paths):
                return new_branches, new_paths
            else:
                return None # Should not happen, but safe

        return None

    def greedy_grow(initial_branches, initial_paths):
        branches = list(initial_branches)
        paths = dict(initial_paths)
        
        candidates = [v for v in range(n) if v not in branches]
        candidates.sort(key=lambda v: -len(adj[v]))
        
        improved = True
        while improved:
            improved = False
            random.shuffle(candidates) # Randomize order to escape local minima
            for v in candidates:
                if time_up():
                    break
                if v in branches:
                    continue
                res = try_add_vertex(branches, paths, v)
                if res is not None:
                    branches, paths = res
                    improved = True
                    candidates = [x for x in range(n) if x not in branches]
                    candidates.sort(key=lambda x: -len(adj[x]))
                    break
            
        return branches, paths

    # Find initial triangles
    triangles = []
    for a, b, c in combinations(range(n), 3):
        if b in adj[a] and c in adj[a] and c in adj[b]:
            triangles.append([a, b, c])
            if len(triangles) >= 20:
                break

    best_size = 0
    best_result = None

    for tb in triangles:
        if time_up():
            break
        init_paths = {}
        for u, v in combinations(tb, 2):
            init_paths[(u, v)] = [u, v]
            init_paths[(v, u)] = [u, v]
        
        b, p = greedy_grow(tb, init_paths)
        if verify_solution(b, p):
            if len(b) > best_size:
                best_size = len(b)
                best_result = {'branches': b, 'paths': p}
        else:
            # Fallback to initial triangle if greedy failed verification (should not happen)
            if verify_solution(tb, init_paths) and len(tb) > best_size:
                best_size = len(tb)
                best_result = {'branches': tb, 'paths': init_paths}

    # If no triangle found, try edges
    if best_result is None:
        for u, v in edges:
            init_paths = {(u, v): [u, v], (v, u): [u, v]}
            b, p = greedy_grow([u, v], init_paths)
            if verify_solution(b, p):
                if len(b) > best_size:
                    best_size = len(b)
                    best_result = {'branches': b, 'paths': p}
            else:
                if verify_solution([u, v], init_paths):
                    best_size = 2
                    best_result = {'branches': [u, v], 'paths': init_paths}
            if best_size > 2:
                break

    if best_result is None:
        # Absolute fallback
        u, v = edges[0]
        best_result = {'branches': [u, v], 'paths': {(u, v): [u, v], (v, u): [u, v]}}

    # Convert to required format
    branches = [int(x) for x in best_result['branches']]
    paths_list = []
    for u, v in combinations(branches, 2):
        path = best_result['paths'][(u, v)]
        paths_list.append([int(x) for x in path])
        
    return {'branches': branches, 'paths': paths_list}
