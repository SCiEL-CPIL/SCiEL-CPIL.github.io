"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
edges = P['edges']

adj = [[] for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def has_bicolored_cycle(coloring, k):
    """Check if any pair of colors forms a cycle using union-find."""
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            parent = list(range(n))
            rank = [0] * n
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1
                return True
            
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                    if not union(u, v):
                        return True
    return False

def validate(coloring):
    """Check proper coloring and no bicolored cycles."""
    k = max(coloring) + 1
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    return not has_bicolored_cycle(coloring, k)

def solve_csp(k, deadline):
    """Solve CSP with k colors using backtracking with forward checking and MRV."""
    # domains[v] = set of available colors for vertex v
    domains = [set(range(k)) for _ in range(n)]
    assigned = [None] * n
    
    # Precompute adjacency for quick access
    adj_sets = [set(adj[v]) for v in range(n)]
    
    def is_valid_assignment(v, c):
        """Check if assigning color c to vertex v is valid given current assignments."""
        for u in adj[v]:
            if assigned[u] is not None and assigned[u] == c:
                return False
        return True
    
    def forward_check(v, c):
        """Apply forward checking after assigning color c to vertex v.
        Returns False if any domain becomes empty."""
        for u in adj[v]:
            if assigned[u] is None:
                if c in domains[u]:
                    domains[u].discard(c)
                    if not domains[u]:
                        return False
        return True
    
    def backtrack():
        """Backtracking search with MRV heuristic."""
        # Find unassigned vertex with MRV
        min_size = k + 1
        min_vertex = -1
        for v in range(n):
            if assigned[v] is None:
                size = len(domains[v])
                if size < min_size:
                    min_size = size
                    min_vertex = v
                    if size == 0:
                        return False
                    if size == 1:
                        break
        
        if min_vertex == -1:
            # All assigned
            return True
        
        v = min_vertex
        # Try colors in order (could randomize for different solutions)
        for c in sorted(domains[v]):
            if not is_valid_assignment(v, c):
                continue
            
            # Save state
            saved_domain = set(domains[v])
            neighbor_saved = {}
            for u in adj[v]:
                if assigned[u] is None:
                    neighbor_saved[u] = set(domains[u])
            
            # Assign
            assigned[v] = c
            domains[v] = {c}
            
            if forward_check(v, c):
                if backtrack():
                    return True
            
            # Undo
            assigned[v] = None
            domains[v] = saved_domain
            for u, saved in neighbor_saved.items():
                domains[u] = saved
        
        return False
    
    if backtrack():
        return assigned
    return None

def construct():
    start = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    # Try k = 3 first, then 4, 5, ...
    for k in range(3, 10):
        if time.time() - start > deadline:
            break
        
        result = solve_csp(k, deadline)
        if result is not None:
            # Verify
            if validate(result):
                return [int(c) for c in result]
    
    # Fallback: use greedy with many restarts
    best_coloring = None
    best_k = n + 1
    
    import random
    for k in range(3, 8):
        if time.time() - start > deadline:
            break
        
        for attempt in range(500):
            if time.time() - start > deadline:
                break
            
            random.seed(attempt * 7919 + k * 131)
            coloring = [-1] * n
            order = list(range(n))
            random.shuffle(order)
            
            success = True
            for v in order:
                colors = list(range(k))
                random.shuffle(colors)
                placed = False
                for c in colors:
                    valid = True
                    for neighbor in adj[v]:
                        if coloring[neighbor] == c:
                            valid = False
                            break
                    if valid:
                        coloring[v] = c
                        placed = True
                        break
                if not placed:
                    success = False
                    break
            
            if not success:
                continue
            
            if validate(coloring):
                k_used = max(coloring) + 1
                if k_used < best_k:
                    best_k = k_used
                    best_coloring = list(coloring)
                    break
    
    if best_coloring is not None:
        return [int(c) for c in best_coloring]
    
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
