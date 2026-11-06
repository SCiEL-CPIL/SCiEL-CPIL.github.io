"""Path decomposition: ILP set-cover with smart candidate generation + local refinement."""
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

    def generate_candidate_paths(adj, n, max_paths=5000, max_depth=15, rng=None):
        """
        Generate a diverse set of candidate simple paths using DFS with heuristics.
        Mix of Warnsdorff (prefer low-degree neighbors) and random restarts.
        """
        candidates = []
        seen_edge_sets = set()
        
        def add_candidate(path):
            if len(path) < 2:
                return
            edge_tuple = frozenset(tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1))
            if edge_tuple not in seen_edge_sets:
                seen_edge_sets.add(edge_tuple)
                candidates.append(list(path))
        
        # Strategy 1: Warnsdorff-like from each vertex
        for start in range(n):
            if len(candidates) >= max_paths:
                break
            if not adj[start]:
                continue
            
            for attempt in range(3):  # multiple attempts per start
                if len(candidates) >= max_paths:
                    break
                
                path = [start]
                visited = {start}
                current = start
                
                for _ in range(max_depth):
                    # Get valid neighbors
                    valid_nbs = []
                    for nb in adj[current]:
                        if nb not in visited:
                            # Check edge exists
                            e = tuple(sorted((current, nb)))
                            valid_nbs.append(nb)
                    
                    if not valid_nbs:
                        break
                    
                    if attempt == 0:
                        # Warnsdorff: prefer neighbor with fewest unvisited neighbors
                        def warnsdorff(nb):
                            return sum(1 for x in adj[nb] if x not in visited)
                        valid_nbs.sort(key=warnsdorff)
                        next_nb = valid_nbs[0]
                    elif attempt == 1:
                        # Prefer high degree
                        valid_nbs.sort(key=lambda nb: -len(adj[nb]))
                        next_nb = valid_nbs[0]
                    else:
                        # Random
                        next_nb = valid_nbs[rng.randint(0, len(valid_nbs)-1)] if rng else valid_nbs[0]
                    
                    path.append(next_nb)
                    visited.add(next_nb)
                    current = next_nb
                
                add_candidate(path)
        
        # Strategy 2: Random walks
        for _ in range(max(50, n * 5)):
            if len(candidates) >= max_paths:
                break
            start = rng.randint(0, n-1) if rng else random.randint(0, n-1)
            if not adj[start]:
                continue
            
            path = [start]
            visited = {start}
            current = start
            
            for _ in range(max_depth):
                valid_nbs = [nb for nb in adj[current] if nb not in visited]
                if not valid_nbs:
                    break
                next_nb = valid_nbs[rng.randint(0, len(valid_nbs)-1)] if rng else valid_nbs[0]
                path.append(next_nb)
                visited.add(next_nb)
                current = next_nb
            
            add_candidate(path)
        
        return candidates

    def ilp_path_decomposition(edges, adj, n, max_candidates=5000, time_budget=30, rng=None):
        """
        ILP-based path decomposition: generate candidate paths, solve set cover.
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
        
        # Generate candidate paths
        candidates = generate_candidate_paths(adj, n, max_paths=max_candidates, max_depth=min(n-1, 20), rng=rng)
        
        if not candidates:
            return None
        
        # Build constraint matrix
        edge_indices = {e: i for i, e in enumerate(sorted(edge_set))}
        
        rows, cols, vals = [], [], []
        for j, path in enumerate(candidates):
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e in edge_indices:
                    rows.append(edge_indices[e])
                    cols.append(j)
                    vals.append(1)
        
        if not rows:
            return None
        
        num_candidates = len(candidates)
        A = csr_matrix((vals, (rows, cols)), shape=(m, num_candidates))
        c = np.ones(num_candidates)
        constraints = LinearConstraint(A, lb=np.ones(m), ub=np.ones(m))
        bounds = Bounds(lb=np.zeros(num_candidates), ub=np.ones(num_candidates))
        
        end_time = start_time + time_budget
        if time.time() > end_time:
            return None
        
        try:
            result = milp(c, constraints=constraints, bounds=bounds,
                          integrality=np.ones(num_candidates),
                          options={"time_limit": max(1, time_budget - (time.time() - start_time))})
        except Exception:
            return None
        
        if result.status not in (0, 1) or result.x is None:
            return None
        
        selected = [candidates[j] for j in range(num_candidates) if result.x[j] > 0.5]
        
        if not selected:
            return None
        
        if not verify_decomposition(selected, edges, n):
            return None
        
        return selected

    def merge_paths(paths, edges, n, max_rounds=500):
        """
        Merge pairs of paths if possible. Two paths can be merged if:
        1. They share an endpoint and the merge is simple, OR
        2. An unused edge connects an endpoint of one to an endpoint of the other.
        """
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        
        improved = True
        round_num = 0
        while improved and round_num < max_rounds:
            improved = False
            round_num += 1
            
            # Try merging via shared endpoints
            for i in range(len(current)):
                if improved:
                    break
                for j in range(i + 1, len(current)):
                    if improved:
                        break
                    
                    p1, p2 = current[i], current[j]
                    if set(p1) & set(p2):
                        continue
                    
                    # Try 4 orientations
                    for orient in range(4):
                        if orient == 0:  # p1 + p2 (p1 tail to p2 head)
                            a, b = p1[-1], p2[0]
                            merged = p1 + p2
                        elif orient == 1:  # p2 + p1
                            a, b = p2[-1], p1[0]
                            merged = p2 + p1
                        elif orient == 2:  # p1 + reversed(p2)
                            a, b = p1[-1], p2[-1]
                            merged = p1 + p2[::-1]
                        else:  # reversed(p1) + p2
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

    def greedy_decomposition(edges, adj, n, rng, time_budget=10):
        """
        Randomized greedy: repeatedly find a long simple path using Warnsdorff heuristic.
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
            
            num_starts = min(10, len(degs))
            candidates = sorted(degs.keys(), key=lambda v: (-degs[v], rng.random()))[:num_starts]
            
            best_path = None
            best_len = 0
            
            for start in candidates:
                if not remaining:
                    break
                
                path = [start]
                visited = {start}
                current = start
                
                while True:
                    best_nb = None
                    best_opts = float('inf')
                    for nb in rem_adj[current]:
                        e = tuple(sorted((current, nb)))
                        if e in remaining and nb not in visited:
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
        
        while remaining:
            e = next(iter(remaining))
            paths.append([e[0], e[1]])
            remaining.remove(e)
            rem_adj[e[0]].discard(e[1])
            rem_adj[e[1]].discard(e[0])
        
        return paths

    def solve_graph(edges, adj, n, time_budget, rng):
        """Solve one graph instance with time budget."""
        # Phase 1: Greedy decomposition for initial solution
        greedy_time = min(time_budget * 0.2, 8)
        greedy_result = greedy_decomposition(edges, adj, n, rng, greedy_time)
        
        if not verify_decomposition(greedy_result, edges, n):
            return [list(e) for e in edges]
        
        # Phase 2: ILP refinement (main approach)
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 10:
            ilp_time = min(remaining_time * 0.6, 40)
            ilp_result = ilp_path_decomposition(edges, adj, n, max_candidates=4000, time_budget=ilp_time, rng=rng)
            if ilp_result is not None:
                # Try to merge ILP result
                merged_ilp = merge_paths(ilp_result, edges, n, max_rounds=200)
                if verify_decomposition(merged_ilp, edges, n):
                    if len(merged_ilp) < len(greedy_result):
                        greedy_result = merged_ilp
                elif verify_decomposition(ilp_result, edges, n):
                    if len(ilp_result) < len(greedy_result):
                        greedy_result = ilp_result
        
        # Phase 3: Merge greedy result
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 5:
            merged = merge_paths(greedy_result, edges, n, max_rounds=min(300, int(remaining_time * 20)))
            if verify_decomposition(merged, edges, n):
                greedy_result = merged
        
        return greedy_result

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for idx, spec in enumerate(specs):
        n, p, seed = spec
        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        
        remaining_time = deadline - time.time() - 10
        if remaining_time < 5:
            paths = [list(e) for e in edges]
        else:
            # Allocate time proportional to graph size, give more to larger graphs
            weights = [1, 1.2, 1.5]
            total_weight = sum(weights)
            time_budget = max(8, remaining_time * weights[idx] / total_weight)
            rng = random.Random(seed * 1000 + idx)
            paths = solve_graph(edges, adj, n, time_budget, rng)
        
        results.append(paths)
        
        if time.time() - start_time > deadline - 8:
            for idx2 in range(len(results), 3):
                n2, p2, seed2 = specs[idx2]
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
