"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    best_path = []
    best_len = 0
    
    # Try all starting vertices with randomized DFS
    rng = random.Random(748)
    
    # Time limit
    start_time = time.time()
    max_time = 55  # seconds, leave buffer
    
    # First, do a greedy pass to get a baseline
    for start in range(n):
        path = [start]
        seen = {start}
        while True:
            candidates = [v for v in adj[path[-1]] if v not in seen]
            if not candidates:
                break
            # Pick the candidate with fewest unvisited neighbors (to avoid dead ends)
            candidates.sort(key=lambda v: (len(adj[v] - seen), v))
            path.append(candidates[0])
            seen.add(path[-1])
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Now do randomized DFS with backtracking from each start
    for start in range(n):
        if time.time() - start_time > max_time:
            break
        
        # Try multiple random orderings
        for trial in range(20):
            if time.time() - start_time > max_time:
                break
            
            # Shuffle adjacency lists for this trial
            adj_shuffled = [list(adj[i]) for i in range(n)]
            for i in range(n):
                rng.shuffle(adj_shuffled[i])
            
            # DFS with backtracking
            path = [start]
            seen = {start}
            # Use a stack-based approach with backtracking
            # We'll do a recursive-like DFS using a stack of (node, neighbors_iterator)
            
            def dfs_extend(path, seen):
                nonlocal best_path, best_len
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
                
                if time.time() - start_time > max_time:
                    return False
                
                current = path[-1]
                for neighbor in adj_shuffled[current]:
                    if neighbor not in seen:
                        path.append(neighbor)
                        seen.add(neighbor)
                        dfs_extend(path, seen)
                        path.pop()
                        seen.remove(neighbor)
                        if time.time() - start_time > max_time:
                            return False
                return True
            
            dfs_extend(path, seen)
    
    # Final greedy extension on the best path found
    if best_path:
        seen = set(best_path)
        # Try to extend from both ends
        for _ in range(100):
            extended = False
            # Extend from end
            for v in adj[best_path[-1]]:
                if v not in seen:
                    best_path.append(v)
                    seen.add(v)
                    extended = True
                    break
            # Extend from start
            for v in adj[best_path[0]]:
                if v not in seen:
                    best_path.insert(0, v)
                    seen.add(v)
                    extended = True
                    break
            if not extended:
                break
    
    return best_path if best_path else [0]
# EVOLVE-BLOCK-END
