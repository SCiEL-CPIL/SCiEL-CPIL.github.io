import time
import random
import sys
from collections import defaultdict, deque

sys.setrecursionlimit(10000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    # Fast deterministic HC finder using Warnsdorff's rule with limited backtracking
    def find_hc_fast(adj_remaining, start=0, max_nodes=50000):
        visited = [False] * n
        path = [start]
        visited[start] = True
        nodes_explored = [0]
        
        def dfs(current, depth):
            if nodes_explored[0] > max_nodes:
                return False
            if depth == n:
                if start in adj_remaining[current]:
                    return True
                return False
            
            # Warnsdorff: prefer neighbors with fewest unvisited neighbors
            unvisited = [v for v in adj_remaining[current] if not visited[v]]
            if not unvisited:
                return False
            
            unvisited.sort(key=lambda v: len([x for x in adj_remaining[v] if not visited[x]]))
            
            for nxt in unvisited:
                nodes_explored[0] += 1
                if nodes_explored[0] > max_nodes:
                    return False
                visited[nxt] = True
                path.append(nxt)
                
                if dfs(nxt, depth + 1):
                    return True
                
                path.pop()
                visited[nxt] = False
            
            return False
        
        if dfs(start, 1):
            return list(path)
        return None
    
    # Try to find 3 edge-disjoint Hamiltonian cycles
    deadline = time.time() + 80  # Safety margin for 90s limit
    best_cycles = []
    
    # Strategy 1: Greedy deterministic search
    current_adj = [set(s) for s in adj]
    cycles = []
    success = True
    
    for i in range(3):
        # Try different starting vertices
        found = False
        for start_v in range(n):
            if time.time() > deadline:
                break
            hc = find_hc_fast(current_adj, start=start_v, max_nodes=200000)
            if hc is not None:
                found = True
                break
        
        if not found:
            success = False
            break
        
        cycles.append(hc)
        for j in range(len(hc)):
            u = hc[j]
            v = hc[(j + 1) % len(hc)]
            current_adj[u].discard(v)
            current_adj[v].discard(u)
    
    if success and len(cycles) > len(best_cycles):
        best_cycles = cycles
    
    # If we found 3 cycles, validate and return
    if len(best_cycles) >= 3:
        result = []
        used_edges = set()
        valid = True
        for cycle in best_cycles[:3]:
            if len(cycle) != n or len(set(cycle)) != n:
                valid = False
                break
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                ek = (min(u, v), max(u, v))
                if ek not in edge_set:
                    valid = False
                    break
                if ek in used_edges:
                    valid = False
                    break
                used_edges.add(ek)
            if not valid:
                break
        
        if valid:
            return [[int(v) for v in cycle] for cycle in best_cycles[:3]]
    
    # Fallback: return best found
    if best_cycles:
        result = []
        used_edges = set()
        valid = True
        for cycle in best_cycles:
            if len(cycle) != n or len(set(cycle)) != n:
                valid = False
                break
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                ek = (min(u, v), max(u, v))
                if ek not in edge_set:
                    valid = False
                    break
                if ek in used_edges:
                    valid = False
                    break
                used_edges.add(ek)
            if not valid:
                break
        
        if valid:
            return [[int(v) for v in cycle] for cycle in best_cycles]
    
    # Last resort: return empty (should not happen)
    return []

# Test the program
if __name__ == '__main__':
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Found {len(result)} cycles in {elapsed:.2f}s")
    for i, cycle in enumerate(result):
        print(f"Cycle {i+1}: {cycle}")
