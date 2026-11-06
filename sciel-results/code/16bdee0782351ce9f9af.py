import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        # Path edges
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        # Random edges
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return list(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    all_results = []

    # Allocate time budget proportionally to expected difficulty (n^2)
    weights = [s[0]**2 for s in specs]
    total_weight = sum(weights)
    time_budget = deadline - start_time - 5.0  # 5s safety margin
    budgets = [time_budget * w / total_weight for w in weights]

    for spec_idx, spec in enumerate(specs):
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Calculate lower bound
        deg = {i: 0 for i in range(n)}
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        odd_count = sum(1 for d in deg.values() if d % 2 != 0)
        lower_bound = max(1, odd_count // 2)
        
        best_paths = None
        best_count = float('inf')
        
        # Heuristic: Greedy Longest Path with Randomized Restart
        max_restarts = 200  # Reduced to allow more time per restart
        if n <= 20:
            max_restarts = 400
        elif n <= 25:
            max_restarts = 300
        else:
            max_restarts = 200

        spec_deadline = start_time + budgets[spec_idx]
        if spec_deadline > deadline - 2.0:
            spec_deadline = deadline - 2.0

        for r in range(max_restarts):
            if time.time() > spec_deadline:
                break
            if best_count == lower_bound:
                break
                
            curr_edges = set(edges)
            curr_paths = []
            
            # Randomize seed for tie-breaking in this restart
            rng_restart = random.Random(r + seed)
            
            while curr_edges:
                if time.time() > spec_deadline - 1.0:
                    curr_paths = None
                    break
                
                # Find the longest simple path in curr_edges
                # We try starting from every vertex with degree > 0
                active_vertices = [v for v in range(n) if adj[v] and any((min(v, nb), max(v, nb)) in curr_edges for nb in adj[v])]
                
                if not active_vertices:
                    break
                
                best_path_in_iter = []
                
                # To save time, we can skip vertices that are likely not to yield long paths?
                # But N is small. Let's try all.
                for s in active_vertices:
                    # Greedy Warnsdorff from s
                    path = [s]
                    visited = {s}
                    curr_node = s
                    
                    while True:
                        candidates = []
                        for nb in adj[curr_node]:
                            if nb in visited:
                                continue
                            e = (min(curr_node, nb), max(curr_node, nb))
                            if e in curr_edges:
                                # Calculate unvisited degree of nb in curr_edges
                                unvis_deg = 0
                                for m in adj[nb]:
                                    if m in visited:
                                        continue
                                    e2 = (min(nb, m), max(nb, m))
                                    if e2 in curr_edges:
                                        unvis_deg += 1
                                candidates.append((unvis_deg, nb))
                        
                        if not candidates:
                            break
                        
                        # Sort by unvisited degree ascending (Warnsdorff)
                        candidates.sort(key=lambda x: x[0])
                        
                        # Randomize tie-breaking among those with minimum degree
                        min_deg = candidates[0][0]
                        min_candidates = [c for c in candidates if c[0] == min_deg]
                        next_node = rng_restart.choice(min_candidates)[1]
                        
                        path.append(next_node)
                        visited.add(next_node)
                        curr_node = next_node
                    
                    if len(path) > len(best_path_in_iter):
                        best_path_in_iter = path[:]

                if best_path_in_iter and len(best_path_in_iter) > 1:
                    curr_paths.append(best_path_in_iter)
                    # Remove edges from curr_edges
                    for i in range(len(best_path_in_iter) - 1):
                        u, v = best_path_in_iter[i], best_path_in_iter[i + 1]
                        e = (min(u, v), max(u, v))
                        curr_edges.discard(e)
                else:
                    # Should not happen if curr_edges is not empty, but safe fallback
                    if curr_edges:
                        e = next(iter(curr_edges))
                        curr_paths.append(list(e))
                        curr_edges.discard(e)
            
            if curr_paths and len(curr_paths) < best_count:
                best_count = len(curr_paths)
                best_paths = curr_paths
                
        # If no best_paths found due to timeout, fallback to edge list
        if best_paths is None:
            best_paths = [list(e) for e in edges]
            
        all_results.append(best_paths)

    return all_results
