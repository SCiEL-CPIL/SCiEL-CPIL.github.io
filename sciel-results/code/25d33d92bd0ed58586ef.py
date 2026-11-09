import time
import sys

N = 40
DEADLINE_SECONDS = 50

def construct():
    t0 = time.time()
    
    # Define the graph: Rim + 8 Chords
    # Chords selected to maximize coverage of cycle lengths via single and pair combinations
    # Chords: (0, 2), (0, 4), (0, 8), (0, 16), (0, 24), (0, 32), (0, 36), (0, 38)
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
    # Returns list of vertices in cycle or None
    def find_cycle(L):
        # Simple DFS
        # Start from 0, try to find path of length L-1 back to 0
        # To avoid exponential blowup, prune if remaining steps > remaining nodes
        # Since N=40 and L up to 40, this is fast enough for 38 lengths
        
        def dfs(current, path, visited):
            if len(path) == L:
                if 0 in path and len(path) > 1:
                    # Check if 0 is the last one? No, path ends at current.
                    # We need to close the cycle to 0.
                    # So if current has edge to 0, and 0 is in path (start), we check if path[0]==0.
                    if path[0] == 0 and (current, 0) in edge_set:
                        return path + [0]
                    # Or if we just want the cycle vertices, path is enough if we verify closure
                    return None
                return None
            
            for neighbor in adj[current]:
                if neighbor == 0:
                    # Only close cycle if we have the full length
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
            
        # Optimization: If L is small, direct search. If L is large, search on complement?
        # For now, standard DFS.
        # Special case L=N is the rim.
        if L == N:
            return list(range(N)) + [0]
            
        visited = [False] * N
        visited[0] = True
        # Start DFS
        result = dfs(0, [0], visited)
        if result:
            return result
        return None

    cycles = []
    valid = True
    for L in range(3, N + 1):
        if time.time() - t0 > DEADLINE_SECONDS:
            # Fallback to current known valid (slower) construction if we time out?
            # But we need to return *something*.
            # The static construction below is safer.
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
                cyc = list(range(N))
            elif L % 2 == 1:
                m = (L - 1) // 2
                cyc = list(range(0, 2 * m + 1))
            else:
                m = L // 2
                cyc = [0] + list(range(2, 2 * m + 1))
            cycles.append(cyc)
            
    return {'edges': edges, 'cycles': cycles}
