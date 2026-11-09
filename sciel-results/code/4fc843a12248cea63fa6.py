import time
import random

def _get_distinct_cycle_lengths(adj, n):
    """Get the set of distinct simple cycle lengths in the graph."""
    lengths = set()
    
    def dfs(node, start, visited, path_len):
        for neighbor in adj[node]:
            if neighbor == start and path_len >= 3:
                lengths.add(path_len + 1)
            elif neighbor > start and neighbor not in visited:
                visited.add(neighbor)
                dfs(neighbor, start, visited, path_len + 1)
                visited.remove(neighbor)
    
    for start in range(n):
        dfs(start, start, {start}, 1)
    
    return lengths

def construct():
    """
    Constructs a graph on 12 vertices (0..11) with minimum degree 3 and girth 4.
    Searches for a graph with minimum number of distinct cycle lengths.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # Default: two disjoint K_{3,3} components (cycle lengths {4, 6})
    default_edges = []
    A1 = [0, 1, 2]
    B1 = [3, 4, 5]
    for u in A1:
        for v in B1:
            default_edges.append([u, v])
    A2 = [6, 7, 8]
    B2 = [9, 10, 11]
    for u in A2:
        for v in B2:
            default_edges.append([u, v])
    
    n = 12
    best_edges = default_edges
    best_num_lengths = 2
    
    # Search over 3-regular bipartite graphs on 12 vertices (6+6)
    random.seed(42)
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        # Generate a random 3-regular bipartite graph using configuration model
        a_part = list(range(6))       # Vertices 0-5
        b_part = list(range(6, 12))   # Vertices 6-11
        
        a_stubs = [i for i in a_part for _ in range(3)]
        b_stubs = [i for i in b_part for _ in range(3)]
        
        random.shuffle(b_stubs)
        
        adj = {i: set() for i in range(n)}
        valid = True
        
        for a, b in zip(a_stubs, b_stubs):
            if b in adj[a]:  # Multi-edge
                valid = False
                break
            adj[a].add(b)
            adj[b].add(a)
        
        if not valid:
            continue
        
        # Check min degree (should be 3 by construction, but verify)
        if any(len(adj[i]) != 3 for i in range(n)):
            continue
        
        # Bipartite graph, so no odd cycles. Check for cycles.
        lengths = _get_distinct_cycle_lengths(adj, n)
        
        if len(lengths) < best_num_lengths:
            best_num_lengths = len(lengths)
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append([i, j])
            best_edges = edges
            if best_num_lengths == 1:
                break
    
    return best_edges
