import time
import random
from itertools import combinations

def _get_cycles(graph, n):
    """Get all simple cycles in the graph."""
    cycles = []
    adj = {i: set() for i in range(n)}
    for u, v in graph:
        adj[u].add(v)
        adj[v].add(u)
    
    def dfs(start, current, visited, path):
        if len(path) >= 3:
            # Check if we can close the cycle back to start
            if start in adj[current] and len(path) >= 4:
                cycle = tuple(sorted(path))
                if cycle not in cycles:
                    cycles.append(cycle)
        for neighbor in adj[current]:
            if neighbor not in visited and neighbor > start:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(start, neighbor, visited, path)
                path.pop()
                visited.remove(neighbor)
    
    for i in range(n):
        dfs(i, i, {i}, [i])
    
    return cycles

def _check_validity(edges, n=12):
    """Check if the graph satisfies all constraints."""
    # Check no loops or duplicates
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))
    
    # Check degree >= 3
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    for i in range(n):
        if degree[i] < 3:
            return False
    
    # Check girth >= 4 (no triangles)
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        for j in adj[i]:
            for k in adj[i]:
                if j != k and k in adj[j]:
                    return False
    
    # Check there is at least one cycle
    if len(edges) < n:
        return False
    
    return True

def _count_distinct_cycle_lengths(edges, n=12):
    """Count distinct cycle lengths in the graph."""
    cycles = _get_cycles(edges, n)
    if not cycles:
        return 0, set()
    lengths = set(len(c) for c in cycles)
    return len(lengths), lengths

def construct():
    # Try to find a graph with only 5-cycles (value=1)
    # Start with Petersen graph on vertices 0-9
    petersen_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],  # outer 5-cycle
        [5, 6], [6, 7], [7, 8], [8, 9], [9, 5],  # inner 5-cycle
        [0, 5], [1, 6], [2, 7], [3, 8], [4, 9]   # spokes
    ]
    
    # Verify Petersen graph has only 5-cycles
    val, lens = _count_distinct_cycle_lengths(petersen_edges, 10)
    # Petersen should have only 5-cycles
    
    # Now add vertices 10 and 11
    # We need to connect them to existing vertices while maintaining:
    # 1. All degrees >= 3
    # 2. No triangles or 4-cycles
    # 3. Only 5-cycles
    
    # Strategy: Connect 10 and 11 to each other and to 3 existing vertices each
    # But we need to be careful about creating new cycles
    
    # Let's try: 10-11, and connect 10 to {0,1,2}, 11 to {3,4,5}
    # This creates paths like 10-0-1-11-10 (length 4) - BAD!
    
    # Alternative: Don't connect 10 and 11 to each other
    # Connect 10 to {0, 2, 4} and 11 to {1, 3, 5}
    # Check for 4-cycles: 10-0-1-11-10? No, 10 not connected to 11
    # 10-0-4-10 is a 3-cycle - BAD!
    
    # Let me think differently. The Petersen graph is 3-regular.
    # If I add vertices 10 and 11, I need to give them degree >= 3.
    # Each connection to an existing vertex increases that vertex's degree.
    
    # Let's try a different approach: use backtracking search
    deadline = time.time() + 45
    
    # Start with Petersen on 0-9
    base_edges = set()
    for u, v in petersen_edges:
        if u > v:
            u, v = v, u
        base_edges.add((u, v))
    
    # We need to add edges involving 10 and 11
    # Each of 10, 11 needs degree >= 3
    # Possible neighbors for 10: 0-9
    # Possible neighbors for 11: 0-9
    
    best_edges = None
    best_val = float('inf')
    
    # Try all combinations of 3 neighbors for 10 and 3 neighbors for 11
    # from 0-9, plus optionally edge (10,11)
    for n10 in combinations(range(10), 3):
        if time.time() > deadline:
            break
        for n11 in combinations(range(10), 3):
            if time.time() > deadline:
                break
            
            # Try with and without edge (10,11)
            for has_10_11 in [False, True]:
                new_edges = set(base_edges)
                
                for v in n10:
                    new_edges.add((10, v) if 10 < v else (v, 10))
                for v in n11:
                    new_edges.add((11, v) if 11 < v else (v, 11))
                if has_10_11:
                    new_edges.add((10, 11))
                
                edge_list = [[u, v] for u, v in new_edges]
                
                if not _check_validity(edge_list, 12):
                    continue
                
                val, lens = _count_distinct_cycle_lengths(edge_list, 12)
                if val < best_val:
                    best_val = val
                    best_edges = edge_list
                    if val == 1:
                        # Can't do better than 1
                        return [[int(u), int(v)] for u, v in best_edges]
    
    if best_edges is not None:
        return [[int(u), int(v)] for u, v in best_edges]
    
    # Fallback: return the original K3,3 construction
    edges = []
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
    return edges
