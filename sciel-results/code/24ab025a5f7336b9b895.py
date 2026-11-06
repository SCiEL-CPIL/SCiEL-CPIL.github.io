"""Path decomposition: randomized greedy longest-path + aggressive merging + ILP refinement."""
import time
import sys
from collections import defaultdict
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

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

    def greedy_decomposition(edges, adj, n, rng, time_budget=15):
        """
        Randomized greedy: repeatedly find a long simple path using Warnsdorff heuristic.
        Try multiple starting vertices and random restarts for diversity.
        """
        remaining = set(tuple(sorted(e)) for e in edges)
        paths = []
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        end_time = start_time + time_budget
        
        while remaining and time.time() < end_time:
            degs = {v: len(neigh) for v, neigh in rem_adj.items() if len(neigh) > 0}
            if not degs:
                break
            
            # Try multiple starting points
            num_starts = min(15, len(degs))
            candidates = sorted(degs.keys(), key=lambda v: (-degs[v], rng.random()))[:num_starts]
            
            best_path = None
            best_len = 0
            
            for start in candidates:
                if not remaining:
                    break
                
                # Greedy walk with Warnsdorff
                path = [start]
                visited = {start}
                current = start
                
                while True:
                    best_nb = None
                    best_opts = float('inf')
                    for nb in rem_adj[current]:
                        e = tuple(sorted((current, nb)))
                        if e in remaining and nb not in visited:
                            # Count unvisited neighbors of nb
                            opts = sum(1 for nn in rem_adj[nb] 
                                      if nn != current and nn not in visited
                                      and tuple(sorted((nb, nn))) in remaining)
                            if opts < best_opts:
                                best_opts = opts
                                best_nb = nb
                    if best_nb is None:
                        break
                    path.append(best_nb)
                    visited.add(best_nb)
                    e = tuple(sorted((current, best_nb)))
                    current = best_nb
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
            
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

    def merge_paths_aggressive(paths, edges, n, max_rounds=300):
        """
        Aggressive merging: try to merge any pair of paths if the result is a valid simple path.
        Check all 4 orientations. Repeat until no more merges possible.
        """
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        
        improved = True
        round_num = 0
        while improved and round_num < max_rounds:
            improved = False
            round_num += 1
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

    def ilp_refine(paths, edges, n, time_budget=30):
        """
        ILP-based refinement: enumerate candidate paths and solve set cover.
        Only used if greedy+merge hasn't achieved a good solution.
        """
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            from scipy.sparse import csr_matrix
        except ImportError:
            return None
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        m = len(edge_set)
        if m == 0:
            return []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Enumerate candidate simple paths (limited)
        candidates = []
        max_paths_to_enumerate = 3000
        
        def dfs_paths(start, current_path, used_edges, visited, depth=0):
            if len(candidates) >= max_paths_to_enumerate or depth > n - 1:
                return
            if len(current_path) >= 2:
                candidates.append(list(current_path))
            for nb in adj.get(start, []):
                e = tuple(sorted((start, nb)))
                if e not in used_edges and nb not in visited:
                    dfs_paths(nb, current_path + [nb], used_edges | {e}, visited | {nb}, depth + 1)
        
        for v in range(n):
            if len(candidates) >= max_paths_to_enumerate:
                break
            dfs_paths(v, [v], set(), {v})
        
        if not candidates:
            return None
        
        # Deduplicate by edge set
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
        edge_indices = {e: i for i, e in enumerate(edge_set)}
        
        rows, cols, vals = [], [], []
        for j, path in enumerate(unique_candidates):
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e in edge_indices:
                    rows.append(edge_indices[e])
                    cols.append(j)
                    vals.append(1)
        
        A = csr_matrix((vals, (rows, cols)), shape=(m, num_candidates))
        c = np.ones(num_candidates)
        constraints = LinearConstraint(A, lb=np.ones(m), ub=np.ones(m))
        bounds = Bounds(lb=np.zeros(num_candidates), ub=np.ones(num_candidates))
        
        try:
            result = milp(c, constraints=constraints, bounds=bounds,
                          integrality=np.ones(num_candidates))
        except Exception:
            return None
        
        if result.status != 0 or result.x is None:
            return None
        
        selected = [unique_candidates[j] for j in range(num_candidates) if result.x[j] > 0.5]
        
        if not selected:
            return None
        
        if not verify_decomposition(selected, edges, n):
            return None
        
        return selected

    def solve_graph(edges, adj, n, time_budget, rng):
        """Solve one graph instance with time budget."""
        # Phase 1: Greedy decomposition
        greedy_time = min(time_budget * 0.4, 12)
        greedy_result = greedy_decomposition(edges, adj, n, rng, greedy_time)
        
        if not verify_decomposition(greedy_result, edges, n):
            return [list(e) for e in edges]
        
        # Phase 2: Aggressive merging
        merged = merge_paths_aggressive(greedy_result, edges, n)
        if not verify_decomposition(merged, edges, n):
            merged = greedy_result
        
        # Phase 3: Try ILP refinement if time permits
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 15 and len(merged) > 10:
            ilp_time = min(remaining_time * 0.5, 30)
            ilp_result = ilp_refine(merged, edges, n, ilp_time)
            if ilp_result is not None and len(ilp_result) < len(merged):
                ilp_merged = merge_paths_aggressive(ilp_result, edges, n)
                if verify_decomposition(ilp_merged, edges, n):
                    merged = ilp_merged
                elif verify_decomposition(ilp_result, edges, n):
                    merged = ilp_result
        
        return merged

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
            # Allocate time proportional to graph size
            time_budget = max(5, min(remaining_time / (3 - idx), 45))
            rng = random.Random(seed * 1000 + idx)
            paths = solve_graph(edges, adj, n, time_budget, rng)
        
        results.append(paths)
        
        if time.time() - start_time > deadline - 10:
            for idx2 in range(len(results), 3):
                n2, p2, seed2 = specs[idx2]
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
