"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _get_cycles(edges, n=12):
    """Find all simple cycles in an undirected graph. Returns set of cycle lengths."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u,v), max(u,v)))
    
    cycle_lengths = set()
    
    def dfs(start, current, visited, path):
        for neighbor in adj[current]:
            if neighbor == start and len(path) >= 4:
                # Found a cycle back to start
                cycle_lengths.add(len(path))
            elif neighbor not in visited and neighbor > start:
                # Only visit nodes with higher index than start to avoid duplicates
                visited.add(neighbor)
                path.append(neighbor)
                dfs(start, neighbor, visited, path)
                path.pop()
                visited.remove(neighbor)
    
    for start in range(n):
        visited = {start}
        dfs(start, start, visited, [start])
    
    return cycle_lengths

def _validate(edges, n=12):
    """Check if edges form a valid graph: min degree >= 3, no loops, no duplicates, triangle-free, has a cycle."""
    if len(edges) == 0:
        return False
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    
    # Check min degree
    for i in range(n):
        if len(adj[i]) < 3:
            return False
    
    # Check triangle-free
    for i in range(n):
        for j in adj[i]:
            for k in adj[j]:
                if k == i:
                    continue
                if k in adj[i] and j < k:
                    return False
    
    # Check has a cycle
    cycle_lengths = _get_cycles(edges, n)
    if len(cycle_lengths) == 0:
        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Try the current best: two K_3,3 components
    def make_two_k33():
        edges = []
        for u in [0,1,2]:
            for v in [3,4,5]:
                edges.append([u,v])
        for u in [6,7,8]:
            for v in [9,10,11]:
                edges.append([u,v])
        return edges
    
    best_edges = make_two_k33()
    best_score = len(_get_cycles(best_edges))
    
    # Try various candidate graphs
    candidates = []
    
    # Candidate 1: Q3 (hypercube) on vertices 0-7, plus 4 isolated? No, need degree >= 3
    # Q3 has 8 vertices, degree 3. Add 4 more vertices connected to make degree >= 3.
    
    # Candidate 2: A cubic bipartite graph on 12 vertices with controlled cycle structure
    # Let's try the "3-prism" or other structures
    
    # Try: bipartite graph with parts {0,1,2,3,4,5} and {6,7,8,9,10,11}
    # where each vertex has degree 3, and try to minimize distinct cycle lengths
    
    # Generate random cubic bipartite graphs and check
    import random
    random.seed(42)
    
    def random_cubic_bipartite():
        # Part A: 0-5, Part B: 6-11
        # Each vertex in A connects to 3 in B, each in B connects to 3 in A
        # This is a 3-regular bipartite graph on 12 vertices
        # Use configuration model
        while time.time() < deadline:
            # Shuffle connections
            a_list = []
            for _ in range(3):
                a_list.extend(range(6))
            b_list = []
            for _ in range(3):
                b_list.extend(range(6, 12))
            
            random.shuffle(b_list)
            
            edges = []
            edge_set = set()
            valid = True
            for i, a in enumerate(a_list):
                b = b_list[i]
                key = (min(a,b), max(a,b))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                edges.append([a,b])
            
            if not valid or len(edge_set) != 18:
                continue
            
            if _validate(edges):
                candidates.append(edges)
            
            if len(candidates) > 5000:
                break
    
    # Evaluate all candidates
    for edges in candidates:
        if time.time() > deadline - 10:
            break
        cl = _get_cycles(edges)
        if len(cl) < best_score:
            best_score = len(cl)
            best_edges = edges
            if best_score == 1:
                break
    
    # Convert to proper types
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    
    return result

# EVOLVE-BLOCK-END
