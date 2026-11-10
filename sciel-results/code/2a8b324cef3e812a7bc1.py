import time
import random
import sys
from collections import defaultdict, deque

sys.setrecursionlimit(10000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check if remaining unvisited vertices form a connected subgraph
    def is_connected(visited, adj, n):
        unvisited = [v for v in range(n) if not visited[v]]
        if not unvisited:
            return True
        start = unvisited[0]
        seen = set()
        stack = [start]
        seen.add(start)
        while stack:
            node = stack.pop()
            for nb in adj[node]:
                if not visited[nb] and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return len(seen) == len(unvisited)

    # Find Hamiltonian cycle with pruning
    def find_hc(adj, seed=0, use_pruning=True):
        for v in range(n):
            if len(adj[v]) < 2:
                return None
        
        visited = [False] * n
        path = [0]
        visited[0] = True
        rng = random.Random(seed)
        
        def dfs(current, depth):
            if depth == n:
                if 0 in adj[current]:
                    return True
                return False
            
            unvisited = [v for v in adj[current] if not visited[v]]
            if not unvisited:
                return False
            
            # Warnsdorff's heuristic with randomization
            unvisited.sort(key=lambda v: len([x for x in adj[v] if not visited[x]]))
            for i in range(len(unvisited) - 1, 0, -1):
                rng.shuffle(unvisited[:i+1])
            
            for nxt in unvisited:
                visited[nxt] = True
                path.append(nxt)
                
                # Pruning: check connectivity of remaining unvisited vertices
                if use_pruning and depth + 1 < n:
                    if not is_connected(visited, adj, n):
                        path.pop()
                        visited[nxt] = False
                        continue
                
                # Pruning: check if last unvisited vertex can connect to start
                if depth + 1 == n - 1:
                    remaining = [v for v in range(n) if not visited[v]]
                    if remaining and 0 not in adj[remaining[0]]:
                        path.pop()
                        visited[nxt] = False
                        continue
                
                if dfs(nxt, depth + 1):
                    return True
                path.pop()
                visited[nxt] = False
            
            return False
        
        if dfs(0, 1):
            return list(path)
        return None

    deadline = time.time() + 85  # Safety margin for 90s limit
    best_cycles = []
    
    # Try multiple strategies with different seeds
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        current_adj = [set(s) for s in adj]
        cycles = []
        success = True
        
        for i in range(3):
            seed = attempt * 100 + i
            hc = find_hc(current_adj, seed=seed, use_pruning=True)
            if hc is None:
                hc = find_hc(current_adj, seed=seed + 1000, use_pruning=False)
            if hc is None:
                success = False
                break
            cycles.append(hc)
            for j in range(len(hc)):
                u = hc[j]
                v = hc[(j + 1) % len(hc)]
                current_adj[u].remove(v)
                current_adj[v].remove(u)
        
        if success and len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) >= 3:
                break
    
    # Ensure types are correct
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    
    return result

# Test the program
if __name__ == '__main__':
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Found {len(result)} cycles in {elapsed:.2f}s")
    for i, cycle in enumerate(result):
        print(f"Cycle {i+1}: {cycle}")
