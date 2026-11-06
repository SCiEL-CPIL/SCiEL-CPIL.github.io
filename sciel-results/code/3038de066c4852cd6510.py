import time
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin

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

    def find_candidate_paths(adj, odd_vertices, max_len=8, max_paths_per_pair=20):
        """
        Generate candidate simple paths.
        Focus on paths connecting odd vertices, as they are necessary for the parity bound.
        Also include some paths between even vertices to help cover cycles?
        Actually, if we cover all edges, the ILP will handle it.
        But we need a rich enough set of candidates.
        """
        candidates = []
        seen = set()
        
        nodes = sorted(adj.keys())
        
        # We need to cover all edges.
        # If we only use paths between odd vertices, we might not be able to cover
        # the "cycle" parts if they are disconnected from the odd backbone in terms of path options.
        # However, any edge is part of some path.
        
        # Strategy: Generate all simple paths up to max_len.
        # Filter to those that are "useful".
        
        def dfs(start, visited, path):
            if time.time() - start_time > deadline - 5:
                return
            if len(path) >= 2:
                # Check if this path is a valid candidate
                # We might want to limit candidates to avoid huge ILP
                key = frozenset(tuple(sorted(path))) # Path identity is order, but for dedup
                # Actually, path (1,2,3) is different from (1,3,2) if edges are different?
                # No, simple path is a sequence.
                # Dedup by vertex set? No, (1,2,3) and (1,3,2) use different edges.
                # Dedup by the set of edges used?
                edge_set = frozenset(tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1))
                if edge_set not in seen:
                    seen.add(edge_set)
                    candidates.append(list(path))
            
            if len(path) >= max_len:
                return
            
            last = path[-1]
            for nb in sorted(adj.get(last, [])):
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, visited, path)
                    path.pop()
                    visited.remove(nb)
        
        for start in nodes:
            dfs(start, {start}, [start])
            
        return candidates

    def solve_ilp(edges, candidates, time_limit=20.0):
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not candidates:
            return None
            
        num_edges = len(edge_set)
        num_candidates = len(candidates)
        
        # Incidence matrix: A[i, j] = 1 if candidate i uses edge j
        A = np.zeros((num_candidates, num_edges), dtype=np.int8)
        edge_index = {e: i for i, e in enumerate(edge_set)}
        
        for i, path in enumerate(candidates):
            for k in range(len(path) - 1):
                u, v = path[k], path[k+1]
                e = tuple(sorted((u, v)))
                if e in edge_index:
                    A[i, edge_index[e]] = 1
        
        c = np.ones(num_candidates)
        lower = np.ones(num_edges)
        upper = np.ones(num_edges)
        
        constraint = LinearConstraint(A.T, lb=lower, ub=upper)
        bounds = Bounds(lb=0, ub=1)
        
        try:
            result = milp(
                c=c,
                constraints=constraint,
                bounds=bounds,
                integrality=np.ones(num_candidates)
            )
            
            if result.status == 0 and result.x is not None:
                selected = [i for i in range(num_candidates) if result.x[i] > 0.5]
                if len(selected) > 0:
                    # Verify
                    covered = set()
                    for i in selected:
                        path = candidates[i]
                        for k in range(len(path)-1):
                            u, v = path[k], path[k+1]
                            covered.add(tuple(sorted((u, v))))
                    if covered == edge_set:
                        return [candidates[i] for i in selected]
        except Exception:
            pass
            
        return None

    def greedy_fallback(edges, adj):
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        while remaining_edges:
            # Find longest path in remaining graph
            # Simple DFS
            best_path = None
            best_len = 0
            
            # Try starting from each vertex
            nodes = list(adj.keys())
            for start in nodes:
                # DFS
                visited = set()
                path = []
                
                def dfs(v):
                    path.append(v)
                    visited.add(v)
                    for nb in adj[v]:
                        if nb not in visited and tuple(sorted((v, nb))) in remaining_edges:
                            dfs(nb)
                    if len(path) > best_len:
                        nonlocal best_path, best_len
                        best_path = list(path)
                        best_len = len(path)
                    path.pop()
                    visited.remove(v)
                
                dfs(start)
            
            if best_path is None:
                # Single edge
                e = next(iter(remaining_edges))
                best_path = list(e)
            
            paths.append(best_path)
            for k in range(len(best_path)-1):
                u, v = best_path[k], best_path[k+1]
                remaining_edges.discard(tuple(sorted((u, v))))
                
        return paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for n, p, seed in specs:
        if time.time() - start_time > deadline - 10:
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        
        # Calculate odd vertices
        degrees = defaultdict(int)
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
        odd_vertices = [v for v, d in degrees.items() if d % 2 != 0]
        
        # Generate candidates
        # Limit max_len to keep ILP small
        max_len = 10 if n <= 25 else 8
        candidates = find_candidate_paths(adj, odd_vertices, max_len=max_len)
        
        # If too many candidates, reduce
        if len(candidates) > 2000:
            # Prioritize longer paths? Or just take a random subset?
            # Let's just take the first 2000 or filter by length
            candidates = candidates[:2000]
            
        # Try ILP
        ilp_result = solve_ilp(edges, candidates, time_limit=deadline - time.time() - 5)
        
        if ilp_result:
            results.append(ilp_result)
        else:
            # Fallback to greedy
            greedy_paths = greedy_fallback(edges, adj)
            results.append(greedy_paths)
            
    return results
