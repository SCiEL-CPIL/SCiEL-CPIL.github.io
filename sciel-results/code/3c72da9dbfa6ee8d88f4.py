import time
import random
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

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
    # n=20, 25, 30 -> weights 400, 625, 900
    weights = [s[0]**2 for s in specs]
    total_weight = sum(weights)
    time_budget = deadline - start_time - 5.0 # 5s safety margin
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
        # We use a fixed number of restarts per graph, but break early if we hit the lower bound or time out.
        # Since we have a time budget, we just run until deadline or success.
        
        # The previous program used max_restarts. We will keep a similar loop structure but rely on time.
        # However, to ensure we don't get stuck in an infinite loop if a restart is very slow, we check time inside.
        
        # We can try to be smarter about the greedy path.
        # The current greedy finds the longest path by trying all start vertices and using Warnsdorff.
        # This is O(V * (V+E)) per path extraction.
        # Number of paths is ~E/2. Total ~ V * E * (V+E) which is roughly N^4.
        # For N=30, 30^4 = 810,000. With constants, this is fast.
        # The bottleneck is the number of restarts.
        
        # Let's run restarts until time runs out or we find the optimal solution.
        # We'll use a counter for restarts to ensure progress.
        
        restart_count = 0
        # We'll define a "max" restarts not as a hard limit but as a soft one if we are very fast?
        # No, just loop while time remains.
        
        while time.time() < deadline - 5.0:
            if best_count == lower_bound:
                break
            
            curr_edges = set(edges)
            curr_paths = []
            
            # Randomize seed for tie-breaking in this restart
            # Use a global random state or a local one?
            # Using local one based on restart count ensures reproducibility if needed, but random is fine.
            rng_restart = random.Random(restart_count + seed)
            
            success = True
            while curr_edges:
                if time.time() > deadline - 5.0:
                    success = False
                    break
                
                # Find the longest simple path in curr_edges
                # We try starting from every vertex with degree > 0
                active_vertices = [v for v in range(n) if adj[v] and any((min(v, nb), max(v, nb)) in curr_edges for nb in adj[v])]
                
                if not active_vertices:
                    break
                
                best_path_in_iter = []
                
                # Optimization: Instead of trying ALL active vertices, we can try a subset?
                # But for N=30, trying all is fast enough.
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
                    # Fallback for safety
                    if curr_edges:
                        e = next(iter(curr_edges))
                        curr_paths.append(list(e))
                        curr_edges.discard(e)
            
            if success and curr_paths and len(curr_paths) < best_count:
                best_count = len(curr_paths)
                best_paths = curr_paths
                
            restart_count += 1
            
            # If we've done a lot of restarts and haven't improved, maybe we should break?
            # But the time check is the primary guard.
            
        # If no best_paths found due to timeout or other issues, fallback to edge list
        if best_paths is None:
            best_paths = [list(e) for e in edges]
            
        all_results.append(best_paths)

    return all_results
