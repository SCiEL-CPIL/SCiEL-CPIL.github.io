import time
import sys

# Increase recursion limit for DFS
sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    best_path = []
    
    # Helper to check time
    def time_check():
        return time.time() > deadline

    # Strategy 1: Greedy Warnsdorff to get a baseline
    for start in range(n):
        if time_check():
            break
        path = [start]
        visited = {start}
        while True:
            curr = path[-1]
            candidates = [v for v in adj[curr] if v not in visited]
            if not candidates:
                break
            # Sort by number of unvisited neighbors (Warnsdorff)
            candidates.sort(key=lambda v: len(adj[v] - visited))
            next_v = candidates[0]
            path.append(next_v)
            visited.add(next_v)
        
        if len(path) > len(best_path):
            best_path = path[:]
    
    if len(best_path) == n:
        return best_path

    # Strategy 2: DFS with Warnsdorff heuristic using bitset for speed
    # We try to find a path longer than best_path
    # We use recursion with backtracking
    
    visited = [False] * n
    path = []
    best_len = len(best_path)
    
    def dfs(u):
        nonlocal best_path, best_len
        if time.time() > deadline:
            return False
        
        path.append(u)
        visited[u] = True
        
        cur_len = len(path)
        if cur_len > best_len:
            best_len = cur_len
            best_path = path[:]
            
        if cur_len == n:
            # Success! Keep the path as-is (don't pop)
            return True
        
        curr = u
        candidates = [v for v in adj[curr] if not visited[v]]
        
        if not candidates:
            path.pop()
            visited[u] = False
            return False
        
        # Sort candidates: prefer those with fewer unvisited neighbors (Warnsdorff)
        # This helps avoid creating isolated vertices
        unvisited_set = set()
        for x in range(n):
            if not visited[x]:
                unvisited_set.add(x)
        candidates.sort(key=lambda v: len(adj[v] & unvisited_set))
        
        for v in candidates:
            if time.time() > deadline:
                break
            if dfs(v):
                # Found a full path - unwind without popping (path is already saved)
                return True
        
        # If we didn't return True from the loop, we failed to extend
        path.pop()
        visited[u] = False
        return False

    # Run DFS from vertex 0 first
    dfs(0)
    
    if len(best_path) == n:
        return best_path

    # If DFS from 0 didn't find a full path, try a few other starts
    for start in range(1, 20):
        if time_check():
            break
        if len(best_path) == n:
            break
        dfs(start)
        
    if len(best_path) == n:
        return best_path

    # Strategy 3: Local search - try to extend the best path by reordering
    # Try to find a Hamiltonian path using the greedy path as a seed
    # Use iterative improvement: try swapping vertices to create longer paths
    
    # Try a more targeted approach: for each vertex not in best_path,
    # try to insert it into the path
    if best_path:
        in_path = set(best_path)
        not_in_path = [v for v in range(n) if v not in in_path]
        
        # Try inserting each missing vertex into the path
        for missing in not_in_path:
            if time_check():
                break
            if len(best_path) == n:
                break
            
            # Try inserting missing at each position
            for i in range(len(best_path) + 1):
                if time_check():
                    break
                # Check if insertion is valid
                if i == 0:
                    # Insert at beginning: missing must connect to best_path[0]
                    if best_path[0] in adj[missing]:
                        new_path = [missing] + best_path
                        if len(new_path) > len(best_path):
                            best_path = new_path
                elif i == len(best_path):
                    # Insert at end: missing must connect to best_path[-1]
                    if best_path[-1] in adj[missing]:
                        new_path = best_path + [missing]
                        if len(new_path) > len(best_path):
                            best_path = new_path
                else:
                    # Insert in middle: missing must connect to best_path[i-1] and best_path[i]
                    if best_path[i-1] in adj[missing] and best_path[i] in adj[missing]:
                        new_path = best_path[:i] + [missing] + best_path[i:]
                        if len(new_path) > len(best_path):
                            best_path = new_path
        
        # Try 2-opt style improvements
        if len(best_path) < n:
            in_path = set(best_path)
            not_in_path = [v for v in range(n) if v not in in_path]
            
            # Try replacing a segment to include a missing vertex
            for missing in not_in_path:
                if time_check():
                    break
                if len(best_path) == n:
                    break
                
                # Find where missing can be inserted
                # Try all positions again after previous insertions
                for i in range(len(best_path) + 1):
                    if time_check():
                        break
                    if i == 0:
                        if best_path[0] in adj[missing]:
                            new_path = [missing] + best_path
                            if len(new_path) > len(best_path):
                                best_path = new_path
                    elif i == len(best_path):
                        if best_path[-1] in adj[missing]:
                            new_path = best_path + [missing]
                            if len(new_path) > len(best_path):
                                best_path = new_path
                    else:
                        if best_path[i-1] in adj[missing] and best_path[i] in adj[missing]:
                            new_path = best_path[:i] + [missing] + best_path[i:]
                            if len(new_path) > len(best_path):
                                best_path = new_path
    
    return best_path if best_path else [0]
