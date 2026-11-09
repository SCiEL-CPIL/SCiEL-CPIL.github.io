import time
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _diameter(added_edges, base_adj):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in base_adj:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    max_d = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if any(d == -1 for d in dist):
            return float('inf')
        max_d = max(max_d, max(dist))
    return max_d

def _is_valid(added_edges, base_adj):
    # Check triangle-free (bipartite constraint: even-odd)
    for u, v in added_edges:
        if u % 2 == v % 2:
            return False
    
    # Check diameter
    d = _diameter(added_edges, base_adj)
    return d <= 4

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    n = P['n']
    base_edges = set((min(u,v), max(u,v)) for u,v in P['edges'])
    
    # Generate all valid candidate edges (even-odd, not in base)
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if u % 2 != v % 2 and (min(u,v), max(u,v)) not in base_edges:
                candidates.append((u, v))
    
    best_edges = None
    best_len = float('inf')
    
    # Initial guess: Hub and spoke at 12 (current solution)
    initial_guess = [
        (12, 1), (12, 3), (12, 5), (12, 7), (12, 9),
        (12, 15), (12, 17), (12, 19), (12, 21), (12, 23)
    ]
    
    if _is_valid(initial_guess, base_edges):
        best_edges = list(initial_guess)
        best_len = len(initial_guess)
    
    # Local Search: Try to remove edges
    current_edges = list(best_edges)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(current_edges)
        for i in range(len(current_edges)):
            if time.time() > deadline:
                break
            edge_to_remove = current_edges[i]
            test_edges = current_edges[:i] + current_edges[i+1:]
            if _is_valid(test_edges, base_edges):
                current_edges = test_edges
                best_edges = current_edges
                best_len = len(current_edges)
                improved = True
                break
    
    # Try random restarts with smaller initial sets
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Try to find a valid set of size 6 or 7
        target_size = random.randint(6, 8)
        random_set = random.sample(candidates, target_size)
        
        if _is_valid(random_set, base_edges):
            # Try to reduce it
            current = list(random_set)
            imp = True
            while imp and time.time() < deadline:
                imp = False
                random.shuffle(current)
                for i in range(len(current)):
                    test = current[:i] + current[i+1:]
                    if _is_valid(test, base_edges):
                        current = test
                        imp = True
                        break
            if len(current) < best_len:
                best_edges = current
                best_len = len(current)
    
    if best_edges is None:
        return [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
                [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]
    
    return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-END
