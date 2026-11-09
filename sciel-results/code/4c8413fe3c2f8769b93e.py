"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _count_distinct_cycle_lengths(adj, n):
    """Find all distinct simple cycle lengths in the graph."""
    cycle_lengths = set()
    
    def dfs(start, current, visited, path):
        if len(path) >= 4:
            if current == start:
                cycle_lengths.add(len(path))
                return
        for neighbor in adj[current]:
            if neighbor == start and len(path) >= 4:
                cycle_lengths.add(len(path))
                return
            if neighbor not in visited and neighbor > start:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(start, neighbor, visited, path)
                path.pop()
                visited.remove(neighbor)
    
    for start in range(n):
        visited = {start}
        dfs(start, start, visited, [start])
    
    return len(cycle_lengths)

def _is_valid_graph(edges, n=12, min_degree=3):
    """Check if graph is valid: min degree, triangle-free, has a cycle."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check min degree
    for i in range(n):
        if len(adj[i]) < min_degree:
            return False
    
    # Check triangle-free
    for i in range(n):
        for j in adj[i]:
            if j > i:
                for k in adj[j]:
                    if k > j and k in adj[i]:
                        return False
    
    # Check has a cycle (for connected component with >=4 vertices, or just check)
    # Simple check: if any vertex has degree >= 2 and graph has enough edges
    total_edges = len(edges)
    if total_edges < n:
        return False
    
    return True

def _get_cycle_lengths_fast(adj, n):
    """Get distinct cycle lengths using BFS/DFS from each vertex."""
    cycle_lengths = set()
    
    for start in range(n):
        # DFS to find cycles back to start
        visited = {start}
        path = [start]
        
        def dfs(current):
            if len(path) >= 4:
                if start in adj[current]:
                    cycle_lengths.add(len(path))
            for neighbor in adj[current]:
                if neighbor == start:
                    continue
                if neighbor not in visited and neighbor > start:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor)
                    path.pop()
                    visited.remove(neighbor)
        
        dfs(start)
    
    return len(cycle_lengths)

def construct():
    """
    Tries to find a graph with minimum distinct cycle lengths.
    Falls back to two disjoint K_{3,3} components (2 distinct cycle lengths).
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline with 10s safety margin
    
    # Default: two K_{3,3} components
    default_edges = set()
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            default_edges.add(tuple(sorted((u, v))))
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            default_edges.add(tuple(sorted((u, v))))
    
    default_result = [list(e) for e in sorted(default_edges)]
    default_score = 2  # K_{3,3} has cycle lengths {4, 6}
    
    # Try to find a graph with 1 distinct cycle length
    # Strategy: search 3-regular bipartite graphs on 12 vertices (6+6)
    # where all cycles are the same length
    
    n = 12
    best_edges = default_edges
    best_score = default_score
    
    # Try specific known constructions first
    # Construction 1: 3-regular bipartite graph with specific structure
    # Let's try the "crown" or other structured graphs
    
    # Try: a 3-regular bipartite graph where each vertex in A connects to 3 in B
    # such that no 6-cycles exist
    
    # One approach: use a graph based on a 6-cycle with chords removed
    # Actually, let's try a systematic small search
    
    # Try all 3-regular bipartite graphs? Too many.
    # Instead, try specific parameterized families
    
    # Family: "shifted" bipartite graphs
    # A[i] connects to B[(i+k) mod 6] for k in some set of size 3
    
    for shifts in itertools.combinations(range(6), 3):
        if time.time() > deadline:
            break
        edges_set = set()
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(6):
            for k in shifts:
                u = i  # in A: 0-5
                v = 6 + ((i + k) % 6)  # in B: 6-11
                if u > v:
                    u, v = v, u
                edge = (u, v)
                edges_set.add(edge)
                adj[u].add(v)
                adj[v].add(u)
        
        # Check 3-regular
        if any(len(adj[i]) != 3 for i in range(n)):
            continue
        
        # Check triangle-free (bipartite, so automatically triangle-free)
        # Check distinct cycle lengths
        ncl = _get_cycle_lengths_fast(adj, n)
        
        if ncl < best_score:
            best_score = ncl
            best_edges = edges_set
            if best_score == 1:
                break
    
    if time.time() > deadline:
        return default_result
    
    if best_score < default_score:
        return [list(e) for e in sorted(best_edges)]
    else:
        return default_result

# EVOLVE-BLOCK-END
