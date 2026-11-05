import time
import random

N = 40

def generate_ladder_edges(n):
    """
    Generates edges for a 2 x (n/2) ladder graph.
    Vertices 0..n-1.
    Top: 0, 2, 4, ...
    Bottom: 1, 3, 5, ...
    """
    edges = []
    half = n // 2
    # Top path
    for i in range(half - 1):
        edges.append((2 * i, 2 * i + 2))
    # Bottom path
    for i in range(half - 1):
        edges.append((2 * i + 1, 2 * i + 3))
    # Rungs
    for i in range(half):
        edges.append((2 * i, 2 * i + 1))
    return edges

def find_cycle_of_length(edges, n, target_len):
    """
    Finds a simple cycle of exactly target_len in the graph.
    Returns list of vertices or None.
    Uses backtracking with pruning.
    """
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Sort neighbors for determinism and potential pruning
    for i in range(n):
        adj[i].sort()
        
    visited = [False] * n
    path = []
    
    def dfs(u, depth, target):
        if depth == target:
            # Check if u connects to start
            if path[0] in adj[u]:
                return path[:] + [path[0]]
            return None
        
        # Pruning: if remaining depth > n - depth, impossible?
        # Simple bound: max possible cycle length is n.
        
        for v in adj[u]:
            if not visited[v]:
                # If v is the start node, we can only take it if depth+1 == target
                if v == path[0]:
                    if depth + 1 == target:
                        return path[:] + [v]
                    else:
                        continue
                
                visited[v] = True
                path.append(v)
                
                res = dfs(v, depth + 1, target)
                if res:
                    return res
                
                path.pop()
                visited[v] = False
                
        return None

    # Try starting from each node?
    # To be fast, just try starting from 0, then 1, etc.
    # But cycles can be anywhere.
    # For a connected graph, if a cycle of length L exists, it's likely reachable.
    # We can restrict search to a subset if needed, but N=40 is small.
    
    # Optimization: Only start DFS from nodes with degree >= 2
    for start in range(n):
        if len(adj[start]) < 2:
            continue
        visited[start] = True
        path = [start]
        res = dfs(start, 1, target_len)
        if res:
            return res
        visited[start] = False
        path = []
        
    return None

def is_pancyclic(edges, n, deadline):
    """
    Checks if the graph is pancyclic.
    Returns list of cycles if so, else None.
    """
    cycles = []
    for L in range(3, n + 1):
        if time.time() > deadline:
            return None
        cycle = find_cycle_of_length(edges, n, L)
        if cycle is None:
            return None
        cycles.append(cycle)
    return cycles

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    # Strategy 1: Ladder + Chord (0,3)
    # This has 59 edges.
    edges = generate_ladder_edges(N)
    edges.append((0, 3))
    
    cycles = is_pancyclic(edges, N, deadline)
    if cycles:
        return {'edges': edges, 'cycles': cycles}
        
    # Strategy 2: Ladder + More Chords
    # If (0,3) wasn't enough, add more
    extra_chords = [(0, 5), (0, 7), (1, 4), (1, 6), (2, 5), (2, 7), (3, 6), (3, 8)]
    for chord in extra_chords:
        if time.time() > deadline:
            break
        edges.append(chord)
        cycles = is_pancyclic(edges, N, deadline)
        if cycles:
            return {'edges': edges, 'cycles': cycles}
            
    # Strategy 3: Fallback to Seed (Star + Path)
    # Edges: (0,i) for i=1..39, (i,i+1) for i=1..38
    seed_edges = [(0, i) for i in range(1, N)] + [(i, i+1) for i in range(1, N-1)]
    cycles = is_pancyclic(seed_edges, N, deadline)
    if cycles:
        return {'edges': seed_edges, 'cycles': cycles}
        
    # Emergency fallback: Return seed structure even if verification timed out?
    # No, must return valid.
    # Assume seed is valid.
    return {'edges': seed_edges, 'cycles': [list(range(k)) for k in range(3, N+1)]}

# EVOLVE-BLOCK-END
