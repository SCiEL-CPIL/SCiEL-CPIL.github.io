import time
import random

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Convert to lists for faster iteration
    adj_list = [list(s) for s in adj]
    
    best_path = []
    deadline = time.time() + 85  # 85s to leave margin for 90s limit
    rng = random.Random(12345)
    
    def try_extend(path, seen):
        """Try to extend path using randomized DFS with backtracking."""
        nonlocal best_path, deadline
        if time.time() > deadline:
            return path
        
        if len(path) > len(best_path):
            best_path = path[:]
        
        current = path[-1]
        # Get unvisited neighbors
        unvisited = [v for v in adj_list[current] if v not in seen]
        
        if not unvisited:
            return path
        
        # Shuffle for randomization
        rng.shuffle(unvisited)
        
        # Try each neighbor
        for v in unvisited:
            if time.time() > deadline:
                return path
            path.append(v)
            seen.add(v)
            try_extend(path, seen)
            path.pop()
            seen.discard(v)
        
        return path
    
    # Try from each starting vertex with multiple random restarts
    for start in range(n):
        if time.time() > deadline:
            break
        for attempt in range(20):
            if time.time() > deadline:
                break
            path = [start]
            seen = {start}
            try_extend(path, seen)
    
    # Also try a greedy heuristic as a fallback/improver
    for start in range(n):
        if time.time() > deadline:
            break
        path = [start]
        seen = {start}
        while True:
            current = path[-1]
            unvisited = [v for v in adj_list[current] if v not in seen]
            if not unvisited:
                break
            # Choose neighbor with fewest unvisited neighbors (to avoid dead ends)
            v = min(unvisited, key=lambda w: (len([x for x in adj_list[w] if x not in seen]), w))
            path.append(v)
            seen.add(v)
        if len(path) > len(best_path):
            best_path = path[:]
    
    if not best_path:
        best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
