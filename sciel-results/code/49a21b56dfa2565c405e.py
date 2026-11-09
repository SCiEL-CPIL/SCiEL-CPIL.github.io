import time
import sys

N = 40
DEADLINE_SECONDS = 50

def construct():
    t0 = time.time()
    
    # Define the graph: Rim + 8 Chords
    # Chords selected to maximize coverage of cycle lengths
    chord_ends = [2, 4, 8, 16, 24, 32, 36, 38]
    
    edges = []
    # Rim
    for i in range(N - 1):
        edges.append((i, i + 1))
    edges.append((N - 1, 0))
    
    # Chords
    for c in chord_ends:
        edges.append((0, c))
        
    # Deduplicate
    seen = set()
    unique_edges = []
    for a, b in edges:
        if a == b: continue
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            unique_edges.append((a, b))
    edges = unique_edges
    edge_set = set((min(a, b), max(a, b)) for a, b in edges)
    
    # Build adjacency list for DFS
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Find cycle of length L using DFS
    def find_cycle(L):
        if L == N:
            return list(range(N)) + [0]
            
        def dfs(current, path, visited):
            if len(path) == L:
                if path[0] == 0 and (current, 0) in edge_set:
                    return path + [0]
                return None
            
            for neighbor in adj[current]:
                if neighbor == 0:
                    if len(path) == L - 1 and path[0] == 0:
                        return path + [0]
                    continue
                if not visited[neighbor]:
                    visited[neighbor] = True
                    result = dfs(neighbor, path + [neighbor], visited)
                    if result:
                        return result
                    visited[neighbor] = False
            return None
            
        visited = [False] * N
        visited[0] = True
        result = dfs(0, [0], visited)
        if result:
            return result
        return None

    cycles = []
    valid = True
    for L in range(3, N + 1):
        if time.time() - t0 > DEADLINE_SECONDS:
            valid = False
            break
            
        cyc = find_cycle(L)
        if not cyc:
            valid = False
            break
        cycles.append(cyc)
        
    if not valid:
        # Fallback to the 59-edge construction which is guaranteed valid
        rim_edges = [(i, i + 1) for i in range(N - 1)]
        rim_edges.append((N - 1, 0))
        chord_edges = [(0, 2 * k) for k in range(1, N // 2)]
        edges = rim_edges + chord_edges
        seen = set()
        unique_edges = []
        for a, b in edges:
            if a == b: continue
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                unique_edges.append((a, b))
        edges = unique_edges
        cycles = []
        for L in range(3, N + 1):
            if L == N:
                cyc = list(range(N)) + [0]
            elif L % 2 == 1:
                m = (L - 1) // 2
                cyc = list(range(0, 2 * m + 1)) + [0]
            else:
                m = L // 2
                cyc = [0] + list(range(2, 2 * m + 1)) + [0]
            cycles.append(cyc)
            
    return {'edges': edges, 'cycles': cycles}
