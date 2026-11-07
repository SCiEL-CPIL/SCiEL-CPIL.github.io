import time
import random
from itertools import combinations

def count_distinct_cycle_lengths(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    lengths = set()
    
    # DFS to find cycles
    # We visit each node as a start point.
    # To avoid duplicates, only start from the smallest node in the cycle?
    # Actually, standard cycle enumeration:
    # For each start node s:
    #   DFS from s, only allowing nodes > s? No, that misses cycles not containing s as min?
    #   If we require s = min(cycle), then we must only visit nodes > s in the path?
    #   Yes.
    
    for start in range(n):
        # DFS stack: (node, path)
        # path includes start
        stack = [(start, [start])]
        visited = [False] * n
        visited[start] = True
        
        while stack:
            node, path = stack.pop()
            for neighbor in adj[node]:
                if neighbor < start:
                    continue
                if neighbor == start:
                    if len(path) >= 4:
                        lengths.add(len(path))
                elif not visited[neighbor]:
                    # Ensure we don't close a triangle
                    if len(path) == 2 and neighbor == path[0]: 
                         # This would be a 3-cycle (node - start). 
                         # path is [start, node]. len=2. Cycle len 3.
                         # We want >= 4.
                         pass
                    # We can prune if len(path) + 1 > 12? No, max simple cycle is 12.
                    if len(path) < 12:
                         visited[neighbor] = True
                         stack.append((neighbor, path + [neighbor]))
                         # Backtrack
                         visited[neighbor] = False # Wait, standard DFS with stack needs care with visited state
                         # Actually, simple recursion is easier for small n.
        # Recursion is cleaner for correct backtracking.
    
    # Rewrite with recursion for clarity and correctness
    lengths = set()
    for start in range(n):
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        def dfs(u):
            if len(lengths) == 9: # Max lengths 4..12 is 9 distinct values
                return
            for v in adj[u]:
                if v < start:
                    continue
                if v == start:
                    if len(path) >= 4:
                        lengths.add(len(path))
                elif not visited[v]:
                    if len(path) < 12:
                        visited[v] = True
                        path.append(v)
                        dfs(v)
                        path.pop()
                        visited[v] = False
        
        dfs(start)
        
    return len(lengths)

def construct():
    start_time = time.time()
    deadline = start_time + 85 # 5s margin
    
    n = 12
    
    # Candidate 1: Two K3,3
    edges_k33 = []
    # 0-2 vs 3-5
    for u in range(3):
        for v in range(3, 6):
            edges_k33.append([u, v])
    # 6-8 vs 9-11
    for u in range(6, 9):
        for v in range(9, 12):
            edges_k33.append([u, v])
    
    val1 = count_distinct_cycle_lengths(edges_k33, n)
    best_val = val1
    best_edges = edges_k33
    
    return [list(e) for e in sorted([tuple(sorted(e)) for e in best_edges])]
