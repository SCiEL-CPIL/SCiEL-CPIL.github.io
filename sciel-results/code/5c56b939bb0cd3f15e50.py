"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_valid(edges, n=12):
    """Check if edges form a valid graph: no loops, no duplicates, min degree >= 3, triangle-free, has a cycle."""
    edge_set = set()
    adj = {i: set() for i in range(n)}
    for e in edges:
        u, v = e[0], e[1]
        if not isinstance(u, int) or not isinstance(v, int):
            return False
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
    
    # Check min degree
    for i in range(n):
        if len(adj[i]) < 3:
            return False
    
    # Check triangle-free
    for i in range(n):
        neighbors = list(adj[i])
        for a, b in combinations(neighbors, 2):
            if b in adj[a]:
                return False
    
    # Check has a cycle (for n=12, min degree 3, always has cycle)
    # A graph with min degree >= 2 on any component with >= 3 vertices has a cycle
    # With 12 vertices and min degree 3, definitely has cycles
    return True

def _get_cycle_lengths(edges, n=12, max_cycles=10000):
    """Find distinct cycle lengths in the graph."""
    adj = {i: set() for i in range(n)}
    for e in edges:
        u, v = e[0], e[1]
        adj[u].add(v)
        adj[v].add(u)
    
    cycle_lengths = set()
    
    # BFS-based cycle finding: for each edge, check if there's an alternative path
    # More efficient: find all simple cycles up to a reasonable length
    # For 12 vertices, cycles can be at most length 12
    
    # Use DFS to find all simple cycles
    def find_cycles(start, current, visited, path):
        if len(path) > 12:
            return
        for neighbor in sorted(adj[current]):
            if neighbor == start and len(path) >= 4:  # girth >= 4
                cycle_lengths.add(len(path))
            elif neighbor not in visited and neighbor > start:  # avoid duplicates
                visited.add(neighbor)
                path.append(neighbor)
                find_cycles(start, neighbor, visited, path)
                path.pop()
                visited.remove(neighbor)
    
    for start in range(n):
        visited = {start}
        find_cycles(start, start, visited, [start])
    
    return cycle_lengths

def _count_distinct_lengths(edges, n=12):
    """Count distinct cycle lengths."""
    lengths = _get_cycle_lengths(edges, n)
    return len(lengths)

def construct():
    """Construct graph on vertices 0..11 with min degree >= 3, girth >= 4, minimizing distinct cycle lengths."""
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    # Default: two K_{3,3} components (known to have 2 distinct cycle lengths: 4 and 6)
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
    
    best_edges = default_edges
    best_count = _count_distinct_lengths(default_edges)  # Should be 2
    
    # Try to find a graph with 1 distinct cycle length
    # A graph where all cycles have the same length
    # For min degree 3, the only possibility would be all cycles length 4 (cubic bipartite with girth 4)
    # or all cycles length 6, etc.
    
    # Random search for a graph with 1 distinct cycle length
    random.seed(42)
    
    all_vertices = list(range(12))
    
    attempts = 0
    while time.time() < deadline and attempts < 5000:
        attempts += 1
        
        # Generate a random bipartite graph with 6+6 partition
        # Ensure min degree >= 3
        left = [0, 1, 2, 3, 4, 5]
        right = [6, 7, 8, 9, 10, 11]
        
        edges = []
        edge_set = set()
        
        # For each left vertex, connect to at least 3 right vertices
        for u in left:
            candidates = right[:]
            random.shuffle(candidates)
            # Pick at least 3
            k = random.randint(3, 6)
            for v in candidates[:k]:
                key = (min(u, v), max(u, v))
                if key not in edge_set:
                    edge_set.add(key)
                    edges.append([u, v])
        
        # Ensure right vertices also have degree >= 3
        deg_right = {v: 0 for v in right}
        for e in edges:
            for x in e:
                if x in deg_right:
                    deg_right[x] += 1
        
        # Add edges to boost right vertex degrees
        for v in right:
            while deg_right[v] < 3:
                candidates = [u for u in left if (min(u, v), max(u, v)) not in edge_set]
                if not candidates:
                    break
                u = random.choice(candidates)
                key = (min(u, v), max(u, v))
                edge_set.add(key)
                edges.append([u, v])
                deg_right[v] += 1
                # Also update left degree tracking
                # (we'll check validity later)
        
        # Check validity
        if not _is_valid(edges):
            continue
        
        # Check triangle-free (bipartite graphs are automatically triangle-free)
        # Since we constructed a bipartite graph, it's triangle-free
        
        count = _count_distinct_lengths(edges)
        if count < best_count:
            best_count = count
            best_edges = edges[:]
            if best_count == 1:
                break
    
    # Verify final result
    if not _is_valid(best_edges):
        return default_edges
    
    # Ensure proper types
    result = []
    for e in best_edges:
        u, v = int(e[0]), int(e[1])
        result.append([u, v])
    
    return result

# EVOLVE-BLOCK-END
