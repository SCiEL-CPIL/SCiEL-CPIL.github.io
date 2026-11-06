"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute bitmask of neighbors for each vertex
    adj_mask = [0] * n
    for u in range(n):
        for v in adj[u]:
            adj_mask[u] |= (1 << v)
    
    best = []
    best_mask = 0
    
    def update_best(path):
        nonlocal best, best_mask
        if len(path) > len(best):
            best = path[:]
            best_mask = 0
            for v in path:
                best_mask |= (1 << v)
    
    # Strategy 1: Greedy min-remaining (degree-based)
    for start in range(n):
        if time.time() > deadline:
            break
        path = [start]
        seen = 1 << start
        while True:
            avail = adj_mask[path[-1]] & ~seen
            if avail == 0:
                break
            # Pick neighbor with fewest remaining unvisited neighbors
            best_v = -1
            best_score = (999, 999)
            for v in range(n):
                if (avail >> v) & 1:
                    rem = bin(adj_mask[v] & ~seen).count('1')
                    score = (rem, v)
                    if score < best_score:
                        best_score = score
                        best_v = v
            path.append(best_v)
            seen |= (1 << best_v)
        update_best(path)
    
    # Strategy 2: Greedy max-remaining
    for start in range(n):
        if time.time() > deadline:
            break
        path = [start]
        seen = 1 << start
        while True:
            avail = adj_mask[path[-1]] & ~seen
            if avail == 0:
                break
            best_v = -1
            best_score = (-1, 999)
            for v in range(n):
                if (avail >> v) & 1:
                    rem = bin(adj_mask[v] & ~seen).count('1')
                    score = (rem, -v)
                    if score > best_score:
                        best_score = score
                        best_v = v
            path.append(best_v)
            seen |= (1 << best_v)
        update_best(path)
    
    # Strategy 3: Randomized greedy with multiple trials
    rng = random.Random(42)
    for trial in range(2000):
        if time.time() > deadline:
            break
        start = rng.randint(0, n - 1)
        path = [start]
        seen = 1 << start
        while True:
            avail = adj_mask[path[-1]] & ~seen
            if avail == 0:
                break
            candidates = [v for v in range(n) if (avail >> v) & 1]
            # Sort by remaining degree with random tiebreak
            candidates.sort(key=lambda w: (bin(adj_mask[w] & ~seen).count('1'), rng.random()))
            v = candidates[0]
            path.append(v)
            seen |= (1 << v)
        update_best(path)
    
    # Strategy 4: Deeper backtracking with bitmask
    for start in range(n):
        if time.time() > deadline:
            break
        path = [start]
        seen = 1 << start
        
        def backtrack(path, seen):
            nonlocal best, best_mask
            if time.time() > deadline:
                return
            if len(path) > len(best):
                best = path[:]
                best_mask = seen
            cur = path[-1]
            avail = adj_mask[cur] & ~seen
            if avail == 0:
                return
            candidates = [v for v in range(n) if (avail >> v) & 1]
            candidates.sort(key=lambda w: (bin(adj_mask[w] & ~seen).count('1'), w))
            for v in candidates[:6]:
                path.append(v)
                seen |= (1 << v)
                backtrack(path, seen)
                seen &= ~(1 << v)
                path.pop()
        
        backtrack(path, seen)
    
    # Strategy 5: Path extension - try to extend the best path from both ends
    if best:
        if time.time() < deadline:
            # Try extending from the end
            cur = best[-1]
            seen = best_mask
            extended = False
            while True:
                avail = adj_mask[cur] & ~seen
                if avail == 0:
                    break
                candidates = [v for v in range(n) if (avail >> v) & 1]
                candidates.sort(key=lambda w: (bin(adj_mask[w] & ~seen).count('1'), w))
                v = candidates[0]
                best.append(v)
                seen |= (1 << v)
                cur = v
                extended = True
            if extended:
                best_mask = seen
            
            # Try extending from the beginning
            if time.time() < deadline:
                cur = best[0]
                seen = best_mask
                extended = False
                while True:
                    avail = adj_mask[cur] & ~seen
                    if avail == 0:
                        break
                    candidates = [v for v in range(n) if (avail >> v) & 1]
                    candidates.sort(key=lambda w: (bin(adj_mask[w] & ~seen).count('1'), w))
                    v = candidates[0]
                    best.insert(0, v)
                    seen |= (1 << v)
                    cur = v
                    extended = True
                if extended:
                    best_mask = seen
        
        # Strategy 6: Try to insert unvisited vertices into gaps in the best path
        if time.time() < deadline and len(best) < n:
            visited = set(best)
            unvisited = [v for v in range(n) if v not in visited]
            
            # For each pair of consecutive vertices in best, check if an unvisited
            # vertex can be inserted between them
            improved = True
            while improved and time.time() < deadline:
                improved = False
                new_best = best[:]
                for i in range(len(new_best) - 1):
                    u = new_best[i]
                    v = new_best[i + 1]
                    # Check if there's an unvisited vertex w such that u-w and w-v are edges
                    common = adj_mask[u] & adj_mask[v] & ~best_mask
                    if common:
                        # Pick the one with highest degree to common neighbors
                        candidates = [w for w in range(n) if (common >> w) & 1]
                        candidates.sort(key=lambda w: -bin(adj_mask[w] & ~best_mask).count('1'))
                        w = candidates[0]
                        new_best.insert(i + 1, w)
                        best_mask |= (1 << w)
                        improved = True
                        break
                if improved:
                    best = new_best
    
    return best if best else [0]
# EVOLVE-BLOCK-END
