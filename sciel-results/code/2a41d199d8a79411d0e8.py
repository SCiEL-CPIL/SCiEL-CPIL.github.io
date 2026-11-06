"""Path decomposition via ILP (minimum path cover of edges) with greedy fallback."""
import time
import sys
from collections import defaultdict
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def verify_decomposition(paths, edges, n):
        """Verify that paths are valid: simple, use edges, edge-disjoint, cover all edges."""
        all_edges = set(tuple(sorted(e)) for e in edges)
        used_edges = set()
        
        for path in paths:
            if len(path) < 2:
                return False
            if len(set(path)) != len(path):
                return False
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in all_edges:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        if used_edges != all_edges:
            return False
        
        return True

    def greedy_decomposition(edges, adj, n, time_budget=10):
        """Fast greedy fallback: repeatedly take longest available simple path."""
        remaining = set(tuple(sorted(e)) for e in edges)
        paths = []
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        end_time = time.time() + time_budget
        
        while remaining and time.time() < end_time:
            # Find longest simple path via greedy DFS with backtracking limit
            best_path = None
            best_len = 0
            
            # Try from high-degree vertices first
            degs = {v: len(neigh) for v, neigh in rem_adj.items() if len(neigh) > 0}
            if not degs:
                break
            sorted_verts = sorted(degs.keys(), key=lambda v: -degs[v])
            
            for start in sorted_verts[:min(8, len(sorted_verts))]:
                # Simple greedy path extension
                path = [start]
                visited = {start}
                used_e = set()
                current = start
                
                while True:
                    # Find neighbor with fewest remaining options (Warnsdorff)
                    best_nb = None
                    best_opts = float('inf')
                    for nb in sorted(rem_adj[current]):
                        e = tuple(sorted((current, nb)))
                        if e in remaining and e not in used_e and nb not in visited:
                            opts = sum(1 for nn in rem_adj[nb] 
                                      if nn != current and nn not in visited
                                      and tuple(sorted((nb, nn))) in remaining
                                      and tuple(sorted((nb, nn))) not in used_e)
                            if opts < best_opts:
                                best_opts = opts
                                best_nb = nb
                    if best_nb is None:
                        break
                    path.append(best_nb)
                    visited.add(best_nb)
                    e = tuple(sorted((current, best_nb)))
                    used_e.add(e)
                    current = best_nb
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path
            
            if best_path is None or len(best_path) < 2:
                e = next(iter(remaining))
                paths.append([e[0], e[1]])
                remaining.remove(e)
                rem_adj[e[0]].discard(e[1])
                rem_adj[e[1]].discard(e[0])
            else:
                paths.append(best_path)
                for i in range(len(best_path) - 1):
                    e = tuple(sorted((best_path[i], best_path[i+1])))
                    if e in remaining:
                        remaining.remove(e)
                        rem_adj[e[0]].discard(e[1])
                        rem_adj[e[1]].discard(e[0])
        
        # Fallback for any remaining edges
        while remaining:
            e = next(iter(remaining))
            paths.append([e[0], e[1]])
            remaining.remove(e)
            rem_adj[e[0]].discard(e[1])
            rem_adj[e[1]].discard(e[0])
        
        return paths

    def merge_paths(paths, edges, n):
        """Try to merge pairs of paths that share an edge between their endpoints."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        
        improved = True
        while improved:
            improved = False
            random.shuffle(current)
            for i in range(len(current)):
                if improved:
                    break
                for j in range(i + 1, len(current)):
                    if improved:
                        break
                    p1, p2 = current[i], current[j]
                    if set(p1) & set(p2):
                        continue
                    
                    # Try 4 merge orientations
                    for orient in range(4):
                        if orient == 0:  # p1 + p2
                            a, b = p1[-1], p2[0]
                            merged = p1 + p2
                        elif orient == 1:  # p2 + p1
                            a, b = p2[-1], p1[0]
                            merged = p2 + p1
                        elif orient == 2:  # p1 + reversed(p2)
                            a, b = p1[-1], p2[-1]
                            merged = p1 + p2[::-1]
                        else:  # p1[::-1] + p2
                            a, b = p1[0], p2[0]
                            merged = p1[::-1] + p2
                        
                        e = tuple(sorted((a, b)))
                        if e not in edge_set:
                            continue
                        if len(set(merged)) != len(merged):
                            continue
                        
                        # Verify all edges in merged path exist
                        valid = True
                        for k in range(len(merged) - 1):
                            e2 = tuple(sorted((merged[k], merged[k+1])))
                            if e2 not in edge_set:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        new_current = [current[k] for k in range(len(current)) if k != i and k != j]
                        new_current.append(merged)
                        if verify_decomposition(new_current, edges, n):
                            current = new_current
                            improved = True
                            break
        
        return current

    def ilp_decompose(edges, adj, n, time_budget=45):
        """
        Use scipy.optimize.milp to find minimum number of edge-disjoint simple paths.
        Formulation:
        - For each directed edge (u,v) where {u,v} in E: binary variable x_uv
        - For each vertex v and path index k: binary variable y_vk (vertex v is endpoint of path k)
        - Minimize sum of path indicators
        - Constraints:
          * Each undirected edge is covered exactly once: x_uv + x_vu = 1 for each {u,v}
          * Flow conservation: for each vertex v, sum_in - sum_out = 0 or -1 (endpoint)
          * Vertex capacity: each vertex appears in at most one path (ensures edge-disjoint paths)
        
        Simplified approach: Use a path-based formulation where we enumerate candidate paths
        and select a minimum set that covers all edges.
        """
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
        except ImportError:
            return None
        
        # For small graphs, enumerate all simple paths up to a reasonable length
        # and solve a set cover / minimum path cover ILP
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        m = len(edge_set)
        
        if m == 0:
            return []
        
        # Enumerate candidate simple paths (up to length n-1, but practically limited)
        # Use DFS with a limit on path count for tractability
        candidates = []
        max_paths_to_enumerate = 5000  # limit for tractability
        
        def dfs_paths(start, current_path, used_edges, visited):
            if len(candidates) >= max_paths_to_enumerate:
                return
            if len(current_path) >= 2:
                candidates.append(list(current_path))
            for nb in adj.get(start, []):
                e = tuple(sorted((start, nb)))
                if e not in used_edges and nb not in visited:
                    dfs_paths(nb, current_path + [nb], used_edges | {e}, visited | {nb})
        
        # Enumerate from each vertex, but limit total
        for v in range(n):
            if len(candidates) >= max_paths_to_enumerate:
                break
            dfs_paths(v, [v], set(), {v})
        
        if not candidates:
            return None
        
        # Deduplicate candidates (same set of edges)
        seen_edge_sets = set()
        unique_candidates = []
        for path in candidates:
            edge_tuple = frozenset(tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1))
            if edge_tuple not in seen_edge_sets:
                seen_edge_sets.add(edge_tuple)
                unique_candidates.append(path)
        
        if not unique_candidates:
            return None
        
        num_candidates = len(unique_candidates)
        
        # Build ILP: minimize sum of path selection variables
        # s.t. each edge is covered exactly once
        # x_j = 1 if candidate path j is selected
        
        # Constraint matrix: rows = edges, cols = candidates
        # A[i,j] = 1 if edge i is in candidate path j
        edge_indices = {e: i for i, e in enumerate(edge_set)}
        
        # Build constraint matrix sparsely
        rows = []
        cols = []
        vals = []
        
        for j, path in enumerate(unique_candidates):
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e in edge_indices:
                    rows.append(edge_indices[e])
                    cols.append(j)
                    vals.append(1)
        
        from scipy.sparse import csr_matrix
        A = csr_matrix((vals, (rows, cols)), shape=(m, num_candidates))
        
        # Objective: minimize sum of x_j
        c = np.ones(num_candidates)
        
        # Constraints: A @ x = 1 (each edge covered exactly once)
        constraints = LinearConstraint(A, lb=np.ones(m), ub=np.ones(m))
        
        # Bounds: 0 <= x_j <= 1
        bounds = Bounds(lb=np.zeros(num_candidates), ub=np.ones(num_candidates))
        
        # Solve
        try:
            result = milp(c, constraints=constraints, bounds=bounds,
                          integrality=np.ones(num_candidates))
        except Exception:
            return None
        
        if result.status != 0 or result.x is None:
            return None
        
        # Extract selected paths
        selected = [unique_candidates[j] for j in range(num_candidates) if result.x[j] > 0.5]
        
        if not selected:
            return None
        
        # Verify
        if not verify_decomposition(selected, edges, n):
            return None
        
        return selected

    def solve_graph(edges, adj, n, time_budget):
        """Solve one graph instance with time budget."""
        # First try ILP if time allows
        if time_budget > 20:
            ilp_time = min(time_budget * 0.6, 45)
            ilp_result = ilp_decompose(edges, adj, n, ilp_time)
            if ilp_result is not None:
                # Try to improve with merging
                merged = merge_paths(ilp_result, edges, n)
                if verify_decomposition(merged, edges, n):
                    return merged
                return ilp_result
        
        # Fallback to greedy
        greedy_result = greedy_decomposition(edges, adj, n, min(time_budget, 15))
        if not verify_decomposition(greedy_result, edges, n):
            return [list(e) for e in edges]
        
        # Try merging
        merged = merge_paths(greedy_result, edges, n)
        if verify_decomposition(merged, edges, n):
            return merged
        
        return greedy_result

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for idx, spec in enumerate(specs):
        n, p, seed = spec
        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        
        remaining_time = deadline - time.time() - 15
        if remaining_time < 5:
            paths = [list(e) for e in edges]
        else:
            # Allocate time: more for later graphs if time permits
            time_budget = max(5, min(remaining_time / (3 - idx), 50))
            paths = solve_graph(edges, adj, n, time_budget)
        
        results.append(paths)
        
        if time.time() - start_time > deadline - 10:
            for idx2 in range(len(results), 3):
                n2, p2, seed2 = specs[idx2]
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
