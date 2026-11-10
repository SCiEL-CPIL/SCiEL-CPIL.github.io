import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s buffer

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_graph(edges):
        if not edges:
            return []
        
        # Build adjacency list to find max degree
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        max_deg = max(len(neigh) for neigh in adj.values()) if adj else 0
        target_paths = (max_deg + 1) // 2
        
        # Path representation: (vertex_list, vertex_set)
        # Initial paths
        init_paths = [([u, v], {u, v}) for u, v in edges]
        
        def try_merge(p1, p2):
            v1, s1 = p1
            v2, s2 = p2
            
            # Check intersection
            common = s1 & s2
            if len(common) != 1:
                return None
            
            z = next(iter(common))
            
            # z must be endpoint of both
            if z not in (v1[0], v1[-1]) or z not in (v2[0], v2[-1]):
                return None
            
            # Construct new path
            if v1[0] == z and v2[0] == z:
                new_v = v1[1:] + v2[1:]
            elif v1[0] == z and v2[-1] == z:
                new_v = v1[1:] + v2[:-1]
            elif v1[-1] == z and v2[0] == z:
                new_v = v1[:-1] + v2[1:]
            else:
                new_v = v1[:-1] + v2[:-1][::-1]
                
            new_s = s1 | s2
            return (new_v, new_s)

        def greedy_solve(init_paths):
            paths = list(init_paths)
            
            # To speed up, we can track which vertices are in which paths
            # But for small N, O(N^2) is fine.
            
            while True:
                found_merge = False
                indices = list(range(len(paths)))
                random.shuffle(indices)
                
                for i in indices:
                    if found_merge:
                        break
                    for j in range(i + 1, len(paths)):
                        new_path = try_merge(paths[i], paths[j])
                        if new_path:
                            paths[i] = new_path
                            paths.pop(j)
                            found_merge = True
                            break
                            
                if not found_merge:
                    break
                    
            return [p[0] for p in paths]

        best_paths = None
        best_count = len(edges)
        
        # Run restarts
        # Estimate time per restart:
        # ~100 merges * 5000 checks * 30 ops = 15M ops.
        # ~0.1s per restart.
        # We can run ~1000 restarts in 100s.
        
        while time.time() < deadline - 10:
            res = greedy_solve(init_paths)
            if len(res) < best_count:
                best_count = len(res)
                best_paths = res
                if best_count == target_paths:
                    break
            # If we have achieved target, break
            if best_count == target_paths:
                break

        return best_paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    # Allocate time roughly equally
    time_per_graph = (deadline - time.time()) / 3
    
    for spec in specs:
        edges = instance(*spec)
        # Pass a local deadline for this graph
        local_deadline = time.time() + time_per_graph
        # Modify solve_graph to accept deadline? 
        # Or just rely on the global check inside the loop.
        # The loop in solve_graph checks `deadline` which is global.
        # We should update `deadline` or pass it.
        
        # Let's refactor solve_graph to take a deadline
        paths = solve_graph_with_deadline(edges, local_deadline)
        results.append(paths)
        
    return results

# Need to define solve_graph_with_deadline or modify the above.
# I will wrap it properly in the final code.
