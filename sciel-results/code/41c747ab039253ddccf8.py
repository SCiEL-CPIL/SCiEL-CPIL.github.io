import time
import random
import itertools

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 45  # Safety margin
    
    best_edges = None
    best_score = float('inf')
    
    # Fallback: Two disjoint K_{3,3}
    # Vertices 0-5 and 6-11
    # K_{3,3} on 0-2 (A) and 3-5 (B)
    edges_c1 = []
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges_c1.append([u, v])
        
    # K_{3,3} on 6-8 (A) and 9-11 (B)
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges_c1.append([u, v])
            
    best_edges = edges_c1
    best_score = evaluate(edges_c1)
    
    # Generate random 3-regular bipartite graphs
    # Partition: 0-5 (Left), 6-11 (Right)
    # Each left vertex connects to 3 right vertices.
    # Each right vertex connects to 3 left vertices.
    
    def generate_random_3_reg_bipartite():
        # Represent edges as a set of (u, v) where u in 0..5, v in 6..11
        # We can use a backtracking approach or random permutation swaps.
        # Start with a valid configuration, e.g., two K_{3,3}s?
        # Or start with empty and fill?
        
        # Let's use a simple backtracking with randomization
        adj_u = [set() for _ in range(6)]
        adj_v = [set() for _ in range(6)]
        
        def backtrack(u_idx):
            if u_idx == 6:
                return all(len(adj_v[v]) == 3 for v in range(6))
            
            # Choose a random order of right vertices
            vs = list(range(6))
            random.shuffle(vs)
            
            for v in vs:
                if len(adj_v[v]) >= 3:
                    continue
                # Try adding edge (u_idx, v)
                adj_u[u_idx].add(v)
                adj_v[v].add(u_idx)
                
                if backtrack(u_idx + 1):
                    return True
                
                # Undo
                adj_u[u_idx].remove(v)
                adj_v[v].remove(u_idx)
            return False
        
        if backtrack(0):
            edges = []
            for u in range(6):
                for v in adj_u[u]:
                    edges.append([u, v + 6])
            return edges
        return None

    # Try to find a better graph
    # We want to minimize distinct cycle lengths.
    # Known best: 2 ({4, 6})
    # Can we get 1?
    # If we find a graph with only 4-cycles, score is 5.
    # If we find a graph with only 6-cycles (impossible for 12 vertices 3-reg bip? girth 6 requires 14 vertices min for 3-reg), 
    # then we must have 4-cycles.
    # So minimum distinct lengths is likely 2 or 3.
    # If we find a graph with only {4, 8}, that's 2.
    # If we find a graph with {4, 6, 8}, that's 3.
    
    # Let's run a search
    attempts = 0
    while time.time() < deadline:
        cand = generate_random_3_reg_bipartite()
        if cand:
            score = evaluate(cand)
            if score < best_score:
                best_score = score
                best_edges = cand
                # If we find score 1, we can stop early? 
                # But 1 is likely impossible for 12 vertices.
                # Let's just keep searching for a while.
        attempts += 1
        if attempts % 1000 == 0:
            # Optional: print debug
            pass
            
    return [list(e) for e in sorted(best_edges)]

def evaluate(edges):
    # Check constraints
    if not is_valid(edges):
        return float('inf')
    
    # Count distinct cycle lengths
    lengths = get_cycle_lengths(edges)
    return len(lengths)

def is_valid(edges):
    # Check simple, no loops
    for u, v in edges:
        if u == v:
            return False
        if not (0 <= u < 12 and 0 <= v < 12):
            return False
        
    # Check no duplicates
    es = set()
    for u, v in edges:
        e = tuple(sorted((u, v)))
        if e in es:
            return False
        es.add(e)
        
    # Check degree
    deg = [0] * 12
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d < 3 for d in deg):
        return False
        
    # Check girth >= 4 (no triangles)
    # Build adjacency map
    adj_map = {i: set() for i in range(12)}
    for u, v in edges:
        adj_map[u].add(v)
        adj_map[v].add(u)
        
    for u in range(12):
        for v in adj_map[u]:
            for w in adj_map[v]:
                if w == u:
                    continue
                if w in adj_map[u]:
                    return False
    return True

def get_cycle_lengths(edges):
    adj_map = {i: set() for i in range(12)}
    for u, v in edges:
        adj_map[u].add(v)
        adj_map[v].add(u)
        
    lengths = set()
    
    # Find all simple cycles
    # Use DFS from each node, only counting cycles where start is min vertex
    for s in range(12):
        visited = {s}
        def dfs(node, path):
            for neighbor in adj_map[node]:
                if neighbor == s:
                    if len(path) >= 4:
                        lengths.add(len(path))
                    return
                if neighbor in visited:
                    continue
                if neighbor < s:
                    continue
                visited.add(neighbor)
                dfs(neighbor, path + [neighbor])
                visited.remove(neighbor)
        
        dfs(s, [s])
        
    return lengths
