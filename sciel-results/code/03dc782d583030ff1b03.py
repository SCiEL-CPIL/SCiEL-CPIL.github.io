"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _get_cycles(edges, n=12):
    """Find all simple cycles in an undirected graph. Returns set of cycle lengths."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    cycle_lengths = set()
    
    def dfs(start, current, visited, path):
        for neighbor in adj[current]:
            if neighbor == start and len(path) >= 4:
                cycle_lengths.add(len(path))
            elif neighbor not in visited and neighbor > start:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(start, neighbor, visited, path)
                path.pop()
                visited.remove(neighbor)
    
    for start in range(n):
        visited = {start}
        dfs(start, start, visited, [start])
    
    return cycle_lengths

def _validate_fast(edges, n=12):
    """Fast check: min degree >= 3, no loops/dupes, triangle-free."""
    if len(edges) == 0:
        return False
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v or u < 0 or u >= n or v < 0 or v >= n:
            return False
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) < 3:
            return False
    # Triangle-free check
    for i in range(n):
        adj_set_i = set(adj[i])
        for j in adj[i]:
            for k in adj[j]:
                if k != i and k in adj_set_i and j < k:
                    return False
    return True

def _has_cycle(edges, n=12):
    """Check if graph has at least one cycle using DFS."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited = [False] * n
    def dfs(u, parent):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                if dfs(v, u):
                    return True
            elif v != parent:
                return True
        return False
    for i in range(n):
        if not visited[i]:
            if dfs(i, -1):
                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s limit
    
    # Known best: two disjoint K_{3,3} -> cycle lengths {4, 6}, raw=2
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
    
    # Try to find a graph with only 1 distinct cycle length (raw=1)
    # This requires all simple cycles to have the same length.
    # Strategy: systematic search over cubic bipartite graphs on 12 vertices
    # with a stronger filter: check if all cycles have the same length early.
    
    random.seed(42)
    
    # Precompute: in a 3-regular bipartite graph on 12 vertices,
    # if we can avoid 6-cycles, we might get only 4-cycles (impossible for 3-reg)
    # or avoid 4-cycles to get only 6-cycles (also hard).
    # Let's do a targeted random search with early termination.
    
    def random_cubic_bipartite():
        """Generate a random 3-regular bipartite graph on vertices 0-11."""
        part_a = list(range(6))
        part_b = list(range(6, 12))
        
        # Configuration model: 18 stubs on each side
        a_stubs = []
        for a in part_a:
            a_stubs.extend([a, a, a])
        b_stubs = []
        for b in part_b:
            b_stubs.extend([b, b, b])
        
        random.shuffle(b_stubs)
        
        edges = []
        edge_set = set()
        for a, b in zip(a_stubs, b_stubs):
            key = (min(a,b), max(a,b))
            if key in edge_set:
                return None  # Multi-edge, reject
            edge_set.add(key)
            edges.append([a, b])
        
        return edges
    
    # Also try non-bipartite cubic triangle-free graphs
    # e.g., Petersen-like structures
    
    def try_candidate(edges):
        """Check if candidate improves best_score. Returns new best_score if improved."""
        nonlocal best_score, best_edges
        if not _validate_fast(edges):
            return False
        if not _has_cycle(edges):
            return False
        cl = _get_cycles(edges)
        if len(cl) < best_score:
            best_score = len(cl)
            best_edges = edges
            return True
        return False
    
    # Phase 1: Try many random cubic bipartite graphs
    attempts = 0
    max_attempts = 20000
    while time.time() < deadline - 5.0 and attempts < max_attempts:
        edges = random_cubic_bipartite()
        if edges is not None:
            try_candidate(edges)
            if best_score == 1:
                break
        attempts += 1
    
    # Phase 2: Try structured candidates
    # Candidate: C4-free cubic bipartite (would have only 6+ cycles)
    # or 6-cycle-free (only 4-cycles) - both unlikely to exist on 12 vertices
    
    # Try the "circular ladder" C6 x K2 (Prism graph P6 x K2)
    # This is 3-regular, 12 vertices, but has triangles? No, it's bipartite.
    # C6 x K2 = two C6s with rungs. Cycle lengths: 4, 6, 8, 10, 12?
    # Actually C6 x K2 has girth 4 (the rungs + edges form 4-cycles)
    # and also 6-cycles (the outer C6s). So {4, 6, ...}
    
    # Try a different approach: use scipy to enumerate or use more seeds
    for seed in range(1, 100):
        if time.time() > deadline - 5.0:
            break
        random.seed(seed)
        for _ in range(200):
            edges = random_cubic_bipartite()
            if edges is not None:
                try_candidate(edges)
                if best_score == 1:
                    break
        if best_score == 1:
            break
    
    # Convert to proper types
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    
    return result

# EVOLVE-BLOCK-END
