"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def _count_distinct_cycle_lengths(adj, n):
    """Count distinct simple cycle lengths in a graph using DFS."""
    from collections import deque
    
    cycle_lengths = set()
    
    # Find all simple cycles using a modified DFS
    # For small graphs (12 vertices), this is feasible
    def find_cycles(start, current, visited, path):
        for nb in adj[current]:
            if nb == start and len(path) >= 3:
                cycle_lengths.add(len(path))
            elif nb not in visited and nb > start:  # nb > start to avoid duplicates
                visited.add(nb)
                path.append(nb)
                find_cycles(start, nb, visited, path)
                path.pop()
                visited.remove(nb)
    
    for v in range(n):
        visited = {v}
        find_cycles(v, v, visited, [v])
    
    return len(cycle_lengths)

def _check_validity(edges, n=12, min_deg=3):
    """Check if graph is valid: min degree, triangle-free, has a cycle."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check minimum degree
    for i in range(n):
        if len(adj[i]) < min_deg:
            return False
    
    # Check triangle-free
    for i in range(n):
        for j in adj[i]:
            for k in adj[j]:
                if k == i:
                    continue
                if k in adj[i]:
                    return False
    
    # Check has a cycle (with n=12, min degree 3, always has a cycle)
    return True

def _get_cycle_lengths_set(adj, n):
    """Get the set of distinct cycle lengths."""
    cycle_lengths = set()
    
    def find_cycles(start, current, visited, path):
        for nb in adj[current]:
            if nb == start and len(path) >= 3:
                cycle_lengths.add(len(path))
            elif nb not in visited and nb > start:
                visited.add(nb)
                path.append(nb)
                find_cycles(start, nb, visited, path)
                path.pop()
                visited.remove(nb)
    
    for v in range(n):
        visited = {v}
        find_cycles(v, v, visited, [v])
    
    return cycle_lengths

def construct():
    n = 12
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s
    
    # Strategy: Search for a bipartite graph with parts of 6+6, degree 3,
    # where all cycles are 4-cycles (raw value = 1)
    
    # First, try known constructions
    # A 4-cycle-only graph: each vertex in degree 3 means each vertex is in 3 "directions"
    # Structure: think of it as a 3-regular bipartite graph on 12 vertices
    
    # Try: circulant bipartite graph
    # A = 0..5, B = 6..11
    # Connect i to 6+((i+d) % 6) for d in some set S of size 3
    
    best_edges = None
    best_score = float('inf')
    
    # Try all subsets S of {0,1,2,3,4,5} of size 3
    for S in itertools.combinations(range(6), 3):
        edges = []
        adj = [set() for _ in range(n)]
        for i in range(6):
            for d in S:
                j = (i + d) % 6
                u, v = i, 6 + j
                edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
        
        # Check min degree
        valid = all(len(adj[i]) >= 3 for i in range(n))
        if not valid:
            continue
        
        # Check triangle-free (bipartite, so always triangle-free)
        # Get cycle lengths
        cl_set = _get_cycle_lengths_set(adj, n)
        
        if len(cl_set) < best_score:
            best_score = len(cl_set)
            best_edges = edges
        
        if best_score == 1:
            break
    
    if best_edges is not None and best_score == 1:
        return [list(e) for e in sorted(best_edges)]
    
    # If no 1-cycle-length found, try more general 3-regular bipartite graphs
    # Use a different approach: try all 3-regular bipartite graphs on 6+6
    
    # Actually, let's try a specific known good structure
    # The "Petersen-like" bipartite or a specific configuration
    
    # Try: two K_{3,3} but modify to eliminate 6-cycles
    # K_{3,3} has 4-cycles and 6-cycles. If we can break the 6-cycles...
    
    # Alternative: use a "ladder" of 4-cycles
    # 4-cycles: (0,6,1,7), (1,7,2,8), (2,8,3,9), (3,9,4,10), (4,10,5,11), (5,11,0,6)
    # This is a 6-cycle of 4-cycles, giving a 12-cycle and 6-cycles
    
    # Let me try a specific construction that might work:
    # A = {0,1,2,3,4,5}, B = {6,7,8,9,10,11}
    # Each A vertex connects to 3 B vertices
    # Design so that any two A vertices share at most 1 B neighbor (prevents 4-cycles from being too many)
    # AND no 6-cycles exist
    
    # A 6-cycle: a1-b1-a2-b2-a3-b3-a1 requires a1,a2,a3 in A and b1,b2,b3 in B
    # with a1~b1, b1~a2, a2~b2, b2~a3, a3~b3, b3~a1
    
    # Let's just do a broader search with time limit
    if time.time() < deadline - 5:
        # Try random 3-regular bipartite graphs
        import random
        random.seed(42)
        
        for trial in range(5000):
            if time.time() > deadline - 5:
                break
            
            # Generate random 3-regular bipartite graph
            # Use configuration model
            left = list(range(6)) * 3
            right = list(range(6, 12)) * 3
            random.shuffle(right)
            
            edges = set()
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(18):
                u = left[i]
                v = right[i]
                if u == v:  # shouldn't happen since different ranges
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
            
            if not valid:
                continue
            
            if not all(len(adj[i]) == 3 for i in range(n)):
                continue
            
            cl_set = _get_cycle_lengths_set(adj, n)
            if len(cl_set) < best_score:
                best_score = len(cl_set)
                best_edges = list(edges)
            
            if best_score == 1:
                break
    
    if best_edges is None:
        # Fallback: two K_{3,3}
        edges = set()
        for i in range(6):
            for u, v in [(i, (i+1) % 6), (i+6, (i+1) % 6 + 6), (i, i+6)]:
                edges.add(tuple(sorted((u, v))))
        best_edges = list(edges)
    
    return [list(e) for e in sorted(best_edges)]

# EVOLVE-BLOCK-END
