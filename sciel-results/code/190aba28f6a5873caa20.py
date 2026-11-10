"""Combined program: Uses constructive disjoint K_{3,3} components for score 2,
with fallback to random search and validation from the current program."""
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
    
    adj = [[] for _ in range(n)]
    for u, v in edge_set:
        adj[u].append(v)
        adj[v].append(u)
    
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
    """Count distinct simple cycle lengths. Returns None if invalid."""
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
    
    cycle_lengths = set()
    
    for start in range(n):
        def dfs(current, visited, path_len):
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
    Aim to minimize the number of distinct simple cycle lengths."""
    
    start_time = time.time()
    DEADLINE = 85.0
    n = 12
    
    best_score = float('inf')
    best_edges = None
    
    def try_graph(edge_list, label=""):
        nonlocal best_score, best_edges
        if time.time() > DEADLINE - 5:
            return
        score = _count_distinct_cycle_lengths(edge_list, n)
        if score is not None and score < best_score:
            best_score = score
            best_edges = [list(e) for e in edge_list]
            if best_score <= 1:
                return

    # 1. Constructive approach: Two disjoint K_{3,3} components
    # K_{3,3} has cycle lengths {4, 6}. Disjoint union maintains this set.
    # This should yield a score of 2.
    edges_k33 = []
    # First K_{3,3} on 0-5
    for u in range(3):
        for v in range(3, 6):
            edges_k33.append([u, v])
    # Second K_{3,3} on 6-11
    for u in range(6, 9):
        for v in range(9, 12):
            edges_k33.append([u, v])
    
    try_graph(edges_k33, "disjoint_k33")
    
    # If we already found score 1 (impossible) or 2 (likely), we can stop or continue to verify.
    # Since 2 is very good, we might just return if valid.
    if best_score <= 2:
        # Ensure types are correct
        final_edges = []
        for e in best_edges:
            u, v = e
            final_edges.append([int(u), int(v)])
        final_edges.sort()
        return final_edges

    # 2. Fallback: Random search if constructive fails (should not happen)
    random.seed(42)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    iterations = 0
    while time.time() < DEADLINE - 5 and iterations < 5000:
        iterations += 1
        num_edges = random.randint(18, 25)
        sample = random.sample(all_edges, min(num_edges, len(all_edges)))
        edge_set = set()
        degree = [0] * n
        for u, v in sample:
            key = (min(u, v), max(u, v))
            if key not in edge_set:
                edge_set.add(key)
                degree[u] += 1
                degree[v] += 1
        
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
        
        edges_list = [[u, v] for u, v in edge_set]
        try_graph(edges_list, "random")
        if best_score <= 2:
            break

    if best_edges is None:
        # Absolute fallback: K_{6,6} (score 3)
        best_edges = [[i, j + 6] for i in range(6) for j in range(6)]

    final_edges = []
    for e in best_edges:
        u, v = e
        final_edges.append([int(u), int(v)])
    final_edges.sort()
    return final_edges

# EVOLVE-BLOCK-END
