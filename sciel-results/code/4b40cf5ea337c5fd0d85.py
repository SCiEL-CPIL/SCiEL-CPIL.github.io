import random
import time

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_path = []
    
    # Heuristic: Greedy with Warnsdorff's rule (choose neighbor with fewest unvisited neighbors)
    # We run multiple instances with different random tie-breakers to escape local optima.
    
    def greedy_path(start, rng):
        path = [start]
        seen = {start}
        while True:
            current = path[-1]
            candidates = [v for v in adj[current] if v not in seen]
            if not candidates:
                break
            # Sort by number of unvisited neighbors (Warnsdorff)
            # Tie-break randomly
            def score(v):
                return (len(adj[v] - seen), rng.random())
            candidates.sort(key=score)
            next_v = candidates[0]
            path.append(next_v)
            seen.add(next_v)
        return path

    def try_extend(path):
        """Try to extend the path at either end."""
        changed = True
        while changed:
            changed = False
            # Try start
            if path:
                start_v = path[0]
                for u in adj[start_v]:
                    if u not in path:
                        path.insert(0, u)
                        changed = True
                        break
            # Try end
            if path:
                end_v = path[-1]
                for u in adj[end_v]:
                    if u not in path:
                        path.append(u)
                        changed = True
                        break
        return path

    def try_insert(path):
        """Try to insert any unvisited vertex into the path."""
        path_set = set(path)
        unvisited = [i for i in range(n) if i not in path_set]
        improved = True
        while improved:
            improved = False
            for u in unvisited:
                # Check if u can be inserted between path[i] and path[i+1]
                for i in range(len(path) - 1):
                    if path[i] in adj[u] and path[i+1] in adj[u]:
                        path.insert(i+1, u)
                        improved = True
                        unvisited.remove(u)
                        break
                if improved:
                    break
        return path

    # Multi-start greedy
    rng = random.Random(42)
    for i in range(500):
        if time.time() > deadline:
            break
        start = rng.randint(0, n-1)
        path = greedy_path(start, rng)
        path = try_extend(path)
        path = try_insert(path)
        if len(path) > len(best_path):
            best_path = path
    
    # If we have time, try to improve the best path found using local search / simulated annealing ideas
    # But the above multi-start is often sufficient for sparse graphs.
    # Let's add a simple local search on the best path found.
    
    if best_path and time.time() < deadline - 5.0:
        # Try to find better paths by restarting from vertices in the best path
        for v in best_path:
            if time.time() > deadline:
                break
            path = greedy_path(v, rng)
            path = try_extend(path)
            path = try_insert(path)
            if len(path) > len(best_path):
                best_path = path
                
    return best_path
# EVOLVE-BLOCK-END
