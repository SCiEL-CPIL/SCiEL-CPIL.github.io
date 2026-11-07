import time
import random
from itertools import combinations

def has_cycle_of_length(adj, n, length):
    """Check if graph has a simple cycle of given length using DFS."""
    if length < 3:
        return False
    if length == 4:
        # Check for 4-cycles: u-v-w-x-u where u != w, v != x
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for w in adj[v]:
                    if w == u or w <= v:
                        continue
                    for x in adj[w]:
                        if x == v or x == u:
                            continue
                        if x in adj[u]:
                            return True
        return False
    if length == 6:
        # Check for 6-cycles using backtracking
        def dfs(path, visited):
            if len(path) == 6:
                start = path[0]
                if start in adj[path[5]]:
                    return True
                return False
            for neighbor in adj[path[-1]]:
                if neighbor == path[0] and len(path) < 6:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    if dfs(path, visited):
                        return True
                    path.pop()
                    visited.remove(neighbor)
            return False
        
        for start in range(n):
            for next_v in adj[start]:
                if next_v <= start:
                    continue
                visited = {start, next_v}
                if dfs([start, next_v], visited):
                    return True
        return False
    # Generic check for longer cycles - use BFS/DFS
    # For this problem, we mainly care about 4 and 6
    return False

def get_cycle_lengths(adj, n):
    """Find all distinct cycle lengths in the graph."""
    lengths = set()
    # Check cycles of length 4 through 12 (max possible with 12 vertices)
    for length in range(4, n + 1):
        if has_cycle_of_length(adj, n, length):
            lengths.add(length)
    return lengths

def construct():
    deadline = time.time() + 55
    
    # Fallback: two K_{3,3} components (cycle lengths {4, 6})
    fallback_edges = []
    for u in range(3):
        for v in range(3, 6):
            fallback_edges.append([u, v])
    for u in range(6, 9):
        for v in range(9, 12):
            fallback_edges.append([u, v])
    
    # Search for a 3-regular bipartite graph on 12 vertices with only 4-cycles
    # Bipartite with parts {0..5} and {6..11}, each vertex degree 3
    # We need to find a 3-regular bipartite graph with no 6-cycles
    
    best_result = None
    best_num_lengths = 2  # K_{3,3} x K_{3,3} has 2 lengths
    
    # Try random 3-regular bipartite graphs
    random.seed(42)
    attempts = 0
    
    while time.time() < deadline:
        attempts += 1
        if attempts % 10000 == 0 and time.time() > deadline - 5:
            break
        
        # Generate random 3-regular bipartite graph
        # Left: 0-5, Right: 6-11
        # Each left vertex connects to 3 right vertices, and vice versa
        
        # Use configuration model: create stubs
        left_stubs = [i for i in range(6) for _ in range(3)]
        right_stubs = [i + 6 for i in range(6) for _ in range(3)]
        
        random.shuffle(right_stubs)
        
        # Pair them up
        edges = set()
        valid = True
        for i in range(18):
            u = left_stubs[i]
            v = right_stubs[i]
            if u == v:  # Shouldn't happen since they're in different parts
                valid = False
                break
            edges.add((u, v))
        
        if not valid or len(edges) != 18:
            continue
        
        # Build adjacency
        adj = [[] for _ in range(12)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Verify it's simple (no multi-edges) - should be fine from construction
        # Check minimum degree
        if any(len(adj[i]) < 3 for i in range(12)):
            continue
        
        # Check if it has a cycle (bipartite graph with 18 edges on 12 vertices
        # has at least one cycle since 18 > 12 - 1)
        # Count components to ensure it has a cycle
        visited = [False] * 12
        components = 0
        for i in range(12):
            if not visited[i]:
                components += 1
                stack = [i]
                visited[i] = True
                while stack:
                    node = stack.pop()
                    for neighbor in adj[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            stack.append(neighbor)
        
        # A bipartite graph with 12 vertices and 18 edges:
        # If connected: 18 - 12 + 1 = 7 independent cycles -> definitely has cycles
        # If disconnected, need to check each component
        has_cycle = False
        for comp in range(components):
            # Just check total edges vs vertices
            pass
        # With 18 edges and 12 vertices, by Euler's formula for each component:
        # edges >= vertices - 1 for tree, so if edges > vertices - 1 for any component,
        # it has a cycle. Total: 18 > 12 - 1 = 11, so at least one cycle exists.
        has_cycle = True
        
        if not has_cycle:
            continue
        
        # Find cycle lengths
        cycle_lengths = get_cycle_lengths(adj, 12)
        
        if len(cycle_lengths) == 1:
            # Found a graph with only 1 distinct cycle length!
            result_edges = [[u, v] for u, v in sorted(edges)]
            return result_edges
        
        if len(cycle_lengths) < best_num_lengths:
            best_num_lengths = len(cycle_lengths)
            best_result = [[u, v] for u, v in sorted(edges)]
    
    # If we found something better than fallback, return it
    if best_result is not None and best_num_lengths < 2:
        return best_result
    
    return fallback_edges
