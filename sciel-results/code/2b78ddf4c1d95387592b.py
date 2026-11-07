"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u,v), max(u,v)))
    
    # Precompute path distances
    path_dist = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            path_dist[i][j] = abs(i-j)
    
    # Build adjacency for BFS (base graph + added edges)
    # We'll use a helper to check diameter <= 4 and triangle-free
    
    def is_valid(added_edges):
        # Check no duplicates with base or each other
        added_set = set()
        for u, v in added_edges:
            a, b = min(u,v), max(u,v)
            if (a,b) in base_edges:
                return False
            if (a,b) in added_set:
                return False
            added_set.add((a,b))
        
        # Check triangle-free
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        for i in range(n):
            neighbors = list(adj[i])
            for j in range(len(neighbors)):
                for k in range(j+1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[k]]:
                        return False
        
        # Check connected and diameter <= 4 via BFS
        # BFS from each vertex
        for start in range(n):
            dist = [-1]*n
            dist[start] = 0
            queue = [start]
            idx = 0
            while idx < len(queue):
                u = queue[idx]
                idx += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for d in dist:
                if d == -1 or d > 4:
                    return False
        return True
    
    # Generate all candidate cross-parity edges
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u % 2) != (v % 2):
                a, b = min(u,v), max(u,v)
                if (a,b) not in base_edges:
                    candidates.append((u,v))
    
    # Greedy + local search approach
    # Start with empty set, greedily add edges that reduce "diameter violation" most
    
    # First, let's try a constructive approach:
    # The path has diameter 23. We need to reduce it to 4.
    # With cross-parity edges only, we can create shortcuts.
    
    # Let's try a simple greedy: repeatedly add the edge that most reduces max distance
    # But that's expensive. Let's try a heuristic: add edges that create "jumps"
    
    # A known good structure: add edges that connect vertices ~6-8 apart
    # Let's try: for each even vertex, connect to an odd vertex about 6-8 steps away
    
    # Strategy: use a set of "jump" edges
    # Try adding edges (i, i+k) for various k, ensuring cross-parity
    
    # Let's do a simple local search:
    # Start with a candidate set, then try to remove edges while maintaining validity
    
    # Initial candidate: add edges that form a "grid-like" structure
    # For the path 0-1-2-...-23, add edges:
    # (0,6), (2,8), (4,10), (6,12), (8,14), (10,16), (12,18), (14,20), (16,22)
    # These are all cross-parity (even to even? No: 0 even, 6 even -> same parity!)
    # Need cross-parity: even to odd or odd to even
    
    # Let's reconsider. Cross-parity means one even, one odd.
    # Path edges connect consecutive (always cross-parity).
    # So added edges must also be cross-parity.
    
    # Let's try a different approach: use ILP-like reasoning
    # We need every pair within distance 4.
    # Key pairs that are far apart: (0,23) dist 23, (0,22) dist 22, etc.
    
    # Let's just do a randomized local search with a time budget
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    best = None
    best_count = float('inf')
    
    # Try a few heuristic starting points
    def try_solution(added):
        nonlocal best, best_count
        if is_valid(added):
            if len(added) < best_count:
                best_count = len(added)
                best = list(added)
    
    # Heuristic 1: jump edges of length 5 (cross-parity: 0->5, 1->6, etc.)
    jumps5 = [(i, i+5) for i in range(n-5) if (i%2) != ((i+5)%2)]
    try_solution(jumps5)
    
    # Heuristic 2: jump edges of length 6 (0->6 is same parity, skip; 1->7 cross)
    jumps6 = [(i, i+6) for i in range(n-6) if (i%2) != ((i+6)%2)]
    try_solution(jumps6)
    
    # Heuristic 3: mix of jumps
    jumps = []
    for i in range(n):
        for j in range(i+4, n):
            if (i%2) != (j%2) and (min(i,j), max(i,j)) not in base_edges:
                if j - i in (5, 7, 8):
                    jumps.append((i, j))
    try_solution(jumps[:20])
    
    # Local search: start from a random valid-ish set and try to remove edges
    import random
    random.seed(42)
    
    # Generate a random candidate set and try to minimize
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Random starting set
        random_added = random.sample(candidates, min(len(candidates), random.randint(10, 30)))
        
        # Try to remove edges one by one
        current = list(random_added)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                candidate_remove = current[:i] + current[i+1:]
                if is_valid(candidate_remove):
                    current = candidate_remove
                    improved = True
                    break
        
        try_solution(current)
    
    if best is not None:
        return [[u, v] for u, v in best]
    
    # Fallback: return empty (invalid, but should not happen)
    return []

# EVOLVE-BLOCK-END
