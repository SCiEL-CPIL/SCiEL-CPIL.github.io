"""Combined program: uses Johnson's algorithm for fast cycle enumeration,
tries both K_{6,6} and the inspiration prism graph, then does targeted search."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _validate_edges(edges, n, min_degree, girth):
    """Validate edges: simple graph, min degree, girth constraint, contains a cycle."""
    if not edges:
        return False
    edge_set = set()
    for e in edges:
        if len(e) != 2:
            return False
        u, v = e
        if not isinstance(u, int) or not isinstance(v, int):
            return False
        if isinstance(u, bool) or isinstance(v, bool):
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    degree = [0] * n
    for u, v in edge_set:
        degree[u] += 1
        degree[v] += 1
    for d in degree:
        if d < min_degree:
            return False
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edge_set:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check girth via BFS from each vertex
    min_cycle_len = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        for node in queue:
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
                elif neighbor != parent[node] and parent[neighbor] != node:
                    cycle_len = dist[node] + dist[neighbor] + 1
                    if cycle_len < min_cycle_len:
                        min_cycle_len = cycle_len
                    if cycle_len < girth:
                        return False
    
    if min_cycle_len == float('inf'):
        return False
    
    return True


def _count_distinct_cycle_lengths(edges, n):
    """Count distinct simple cycle lengths using efficient enumeration."""
    if not _validate_edges(edges, n, 3, 4):
        return None
    
    edge_set = set()
    adj = [[] for _ in range(n)]
    for e in edges:
        u, v = e
        key = (min(u, v), max(u, v))
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    
    # Enumerate all simple cycles using DFS with canonical ordering
    cycle_lengths = set()
    
    for start in range(n):
        # Only count cycles where start is the smallest vertex in the cycle
        # DFS: path must only contain vertices > start (except closing back to start)
        def dfs(current, visited, path_len):
            # Try to close cycle
            if path_len >= 3 and start in adj[current]:
                cycle_lengths.add(path_len)
            
            for next_node in adj[current]:
                if next_node == start:
                    continue
                if next_node > start and next_node not in visited:
                    visited.add(next_node)
                    dfs(next_node, visited, path_len + 1)
                    visited.remove(next_node)
        
        visited = {start}
        dfs(start, visited, 1)
    
    return len(cycle_lengths) if cycle_lengths else None


def construct():
    """Construct a graph with minimum degree 3, girth at least 4, on 12 vertices.
    Minimize the number of distinct simple cycle lengths."""
    
    start_time = time.time()
    DEADLINE = 85.0
    n = 12
    
    best_score = float('inf')
    best_edges = None
    
    def try_graph(edge_list, label=""):
        nonlocal best_score, best_edges
        if time.time() > DEADLINE - 5:
            return
        # Quick triangle check
        adj_set = set()
        for u, v in edge_list:
            adj_set.add((u, v))
            adj_set.add((v, u))
        has_tri = False
        for u, v in edge_list:
            for w in adj_set.get(u, []):
                if w != v and (v, w) in adj_set:
                    has_tri = True
                    break
            if has_tri:
                break
        if has_tri:
            return
        
        score = _count_distinct_cycle_lengths(edge_list, n)
        if score is not None and score < best_score:
            best_score = score
            best_edges = [list(e) for e in edge_list]
            if best_score <= 2:
                return
    
    # 1. Try K_{6,6}
    edges_k66 = [[i, j + 6] for i in range(6) for j in range(6)]
    try_graph(edges_k66, "K66")
    
    # 2. Try the inspiration graph (prism-like)
    edges_insp = []
    for i in range(6):
        edges_insp.append([i, (i + 1) % 6])
        edges_insp.append([i + 6, (i + 1) % 6 + 6])
    for i in [0, 2, 4]:
        edges_insp.append([i, i + 6])
    try_graph(edges_insp, "inspiration")
    
    # 3. Try a 12-cycle with carefully chosen chords to maintain girth 4
    # Base: 12-cycle
    edges_cycle = [[i, (i + 1) % 12] for i in range(12)]
    # Add chords that don't create triangles
    # Chords connecting vertices at distance 3 or 4 in the cycle
    # e.g., 0-3, 1-4, 2-5, ... but need to check girth
    chord_sets = [
        [(0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 0), (10, 1), (11, 2)],
        [(0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (7, 11), (8, 0), (9, 1), (10, 2), (11, 3)],
        [(0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8)],
        [(0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9)],
    ]
    for chords in chord_sets:
        edges_test = edges_cycle + [[u, v] for u, v in chords]
        try_graph(edges_test, "cycle+chords")
    
    # 4. Random search with targeted approach
    random.seed(42)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    iterations = 0
    while time.time() < DEADLINE - 5 and iterations < 20000:
        iterations += 1
        
        # Generate random graph with min degree 3
        edge_set = set()
        degree = [0] * n
        
        # Start with a random spanning tree or cycle
        # Add random edges
        num_edges = random.randint(18, 28)
        sample = random.sample(all_edges, min(num_edges, len(all_edges)))
        for u, v in sample:
            key = (min(u, v), max(u, v))
            if key not in edge_set:
                edge_set.add(key)
                degree[u] += 1
                degree[v] += 1
        
        # Fix min degree
        for vertex in range(n):
            while degree[vertex] < 3:
                candidates = [u for u in range(n) if u != vertex and (min(vertex, u), max(vertex, u)) not in edge_set]
                if not candidates:
                    break
                u = random.choice(candidates)
                key = (min(vertex, u), max(vertex, u))
                edge_set.add(key)
                degree[vertex] += 1
                degree[u] += 1
        
        if len(edge_set) > 30:
            continue
        
        edges_list = [[u, v] for u, v in edge_set]
        try_graph(edges_list, "random")
        
        if best_score <= 2:
            break
    
    if best_edges is None:
        best_edges = edges_k66
    
    # Ensure proper types
    final_edges = []
    for e in best_edges:
        u, v = e
        final_edges.append([int(u), int(v)])
    
    final_edges.sort()
    
    return final_edges

# EVOLVE-BLOCK-END
