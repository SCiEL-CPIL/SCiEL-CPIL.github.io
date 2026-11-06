"""Long path in a fixed G(100, 0.018) graph via randomized greedy + local search."""
import time
import random
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

# Build adjacency
N = P['n']
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)

def greedy_path(start, adj, rng, heuristic='min_degree'):
    """Greedy path extension from start."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = adj[cur] - seen
        if not candidates:
            break
        if heuristic == 'min_degree':
            # Prefer neighbors with fewer unvisited neighbors (to avoid dead ends)
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        elif heuristic == 'max_degree':
            v = max(candidates, key=lambda w: (len(adj[w] - seen), w))
        elif heuristic == 'random':
            v = rng.choice(list(candidates))
        else:
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        path.append(v)
        seen.add(v)
    return path

def try_extend(path, adj, rng):
    """Try to extend the path at either end."""
    improved = True
    while improved:
        improved = False
        # Try extending at the start
        cur = path[0]
        candidates = adj[cur] - set(path)
        if candidates:
            v = min(candidates, key=lambda w: (len(adj[w] - set(path)), w))
            path.insert(0, v)
            improved = True
        # Try extending at the end
        cur = path[-1]
        candidates = adj[cur] - set(path)
        if candidates:
            v = min(candidates, key=lambda w: (len(adj[w] - set(path)), w))
            path.append(v)
            improved = True
    return path

def local_search(path, adj, rng, max_iter=2000):
    """Local search: try to swap endpoints or do 2-opt style moves to extend."""
    path = list(path)
    path_set = set(path)
    best_len = len(path)
    
    for _ in range(max_iter):
        # Try to extend at either end
        extended = False
        # Extend at start
        cur = path[0]
        cands = adj[cur] - path_set
        if cands:
            v = min(cands, key=lambda w: (len(adj[w] - path_set), w))
            path.insert(0, v)
            path_set.add(v)
            best_len = len(path)
            extended = True
        # Extend at end
        cur = path[-1]
        cands = adj[cur] - path_set
        if cands:
            v = min(cands, key=lambda w: (len(adj[w] - path_set), w))
            path.append(v)
            path_set.add(v)
            best_len = len(path)
            extended = True
        
        if not extended:
            # Try a 2-opt reversal to create new endpoints
            # Pick a random internal edge and reverse
            if len(path) >= 4:
                i = rng.randint(1, len(path) - 2)
                # Check if reversing segment [0..i] and [i+1..end] creates valid path
                # Actually, let's try: reverse the entire path
                path_rev = path[::-1]
                # This doesn't help directly. Instead, try swapping two internal vertices
                # Simpler: try to find a vertex not in path that is adjacent to both ends
                # or do a "slide" operation
                
                # Try: for each position i, check if we can insert a new vertex
                # between path[i] and path[i+1]
                for i in range(len(path) - 1):
                    a, b = path[i], path[i+1]
                    common = (adj[a] & adj[b]) - path_set
                    if common:
                        v = min(common, key=lambda w: (len(adj[w] - path_set), w))
                        path.insert(i + 1, v)
                        path_set.add(v)
                        best_len = len(path)
                        break
    
    return path

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    rng = random.Random(42)
    
    best_path = []
    best_len = 0
    
    # Phase 1: Greedy from every vertex with different heuristics
    for heuristic in ['min_degree', 'max_degree', 'random']:
        for s in range(N):
            if time.time() > deadline:
                break
            path = greedy_path(s, adj, rng, heuristic)
            if len(path) > best_len:
                best_len = len(path)
                best_path = path
        if time.time() > deadline:
            break
    
    # Phase 2: Try to extend the best path
    if best_path:
        best_path = try_extend(best_path, adj, rng)
        if len(best_path) > best_len:
            best_len = len(best_path)
    
    # Phase 3: Local search on the best path
    if best_path:
        best_path = local_search(best_path, adj, rng, max_iter=5000)
        if len(best_path) > best_len:
            best_len = len(best_path)
    
    # Phase 4: Multiple random restarts with greedy + extend + local search
    for trial in range(5000):
        if time.time() > deadline:
            break
        s = rng.randint(0, N - 1)
        heuristic = rng.choice(['min_degree', 'max_degree', 'random'])
        path = greedy_path(s, adj, rng, heuristic)
        path = try_extend(path, adj, rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            # Do a quick local search
            path = local_search(path, adj, rng, max_iter=1000)
            if len(path) > best_len:
                best_len = len(path)
                best_path = path
    
    # Phase 5: Final extension attempt
    if best_path:
        best_path = try_extend(best_path, adj, rng)
        if len(best_path) > best_len:
            best_len = len(best_path)
            best_path = local_search(best_path, adj, rng, max_iter=10000)
            if len(best_path) > best_len:
                best_len = len(best_path)
    
    # Validate
    if not best_path:
        # Fallback: return a single vertex
        return [0]
    
    # Verify path validity
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in adj[best_path[i]]:
            # Invalid, return a safe single vertex
            return [0]
    
    return best_path
