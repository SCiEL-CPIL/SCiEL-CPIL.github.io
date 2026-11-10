"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
from itertools import combinations

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
edges = P['edges']

# Precompute adjacency
adj = [[] for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# Precompute edge index for fast lookup
edge_set = set()
for u, v in edges:
    edge_set.add((min(u,v), max(u,v)))

def has_bicolored_cycle(colors, k):
    """Check if any pair of colors forms a cycle in the induced subgraph."""
    # Group vertices by color
    color_groups = defaultdict(list)
    for i, c in enumerate(colors):
        color_groups[c].append(i)
    
    used_colors = list(color_groups.keys())
    
    for a, b in combinations(used_colors, 2):
        # Get all edges where both endpoints have color a or b
        relevant_edges = []
        for u, v in edges:
            cu, cv = colors[u], colors[v]
            if (cu == a or cu == b) and (cv == a or cv == b):
                relevant_edges.append((u, v))
        
        if len(relevant_edges) < 3:
            continue
        
        # Union-find
        parent = {}
        rank = {}
        for u, v in relevant_edges:
            if u not in parent:
                parent[u] = u
                rank[u] = 0
            if v not in parent:
                parent[v] = v
                rank[v] = 0
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        
        for u, v in relevant_edges:
            if not union(u, v):
                return True  # Cycle found
    
    return False

def is_valid(colors):
    """Check proper coloring and no bicolored cycles."""
    k = max(colors) + 1 if colors else 0
    
    # Proper coloring
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    
    # No bicolored cycles
    if not has_bicolored_cycle(colors, k):
        return True
    return False

def count_violations(colors):
    """Count violations: proper coloring conflicts + bicolored cycles."""
    violations = 0
    
    # Proper coloring
    for u, v in edges:
        if colors[u] == colors[v]:
            violations += 10
    
    # Bicolored cycles
    k = max(colors) + 1 if colors else 0
    if k < 2:
        return violations
    
    color_groups = defaultdict(list)
    for i, c in enumerate(colors):
        color_groups[c].append(i)
    
    used_colors = list(color_groups.keys())
    
    for a, b in combinations(used_colors, 2):
        relevant_edges = []
        for u, v in edges:
            cu, cv = colors[u], colors[v]
            if (cu == a or cu == b) and (cv == a or cv == b):
                relevant_edges.append((u, v))
        
        if len(relevant_edges) < 3:
            continue
        
        parent = {}
        rank = {}
        for u, v in relevant_edges:
            if u not in parent:
                parent[u] = u
                rank[u] = 0
            if v not in parent:
                parent[v] = v
                rank[v] = 0
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return 1
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return 0
        
        for u, v in relevant_edges:
            violations += union(u, v)
    
    return violations

def local_search(k, initial=None, max_iter=20000, seed=None, deadline=None):
    """Local search with proper coloring + bicolored cycle avoidance."""
    if seed is not None:
        random.seed(seed)
    
    if initial is None:
        colors = [random.randint(0, k-1) for _ in range(n)]
    else:
        colors = list(initial)
    
    current_cost = count_violations(colors)
    best_cost = current_cost
    best_colors = list(colors)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if current_cost == 0:
            return colors
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = list(colors)
        
        # Pick a random vertex
        v = random.randint(0, n-1)
        old = colors[v]
        best_new = old
        best_c = current_cost
        
        for new_c in range(k):
            if new_c == old:
                continue
            colors[v] = new_c
            nc = count_violations(colors)
            if nc < best_c:
                best_c = nc
                best_new = new_c
        colors[v] = best_new
        current_cost = best_c
    
    if best_cost == 0:
        return best_colors
    return None

def simulated_annealing(k, initial=None, max_iter=50000, seed=None, deadline=None):
    """Simulated annealing."""
    if seed is not None:
        random.seed(seed)
    
    if initial is None:
        colors = [random.randint(0, k-1) for _ in range(n)]
    else:
        colors = list(initial)
    
    current_cost = count_violations(colors)
    best_cost = current_cost
    best_colors = list(colors)
    
    T = 50.0
    cooling = 0.9995
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if current_cost == 0:
            return colors
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = list(colors)
        
        v = random.randint(0, n-1)
        old = colors[v]
        new_c = random.randint(0, k-1)
        while new_c == old:
            new_c = random.randint(0, k-1)
        
        colors[v] = new_c
        new_cost = count_violations(colors)
        
        delta = new_cost - current_cost
        if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 0.01)):
            current_cost = new_cost
        else:
            colors[v] = old
        
        T *= cooling
    
    if best_cost == 0:
        return best_colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_result = None
    best_k = 50
    
    # Try increasing k from 3
    for k in range(3, 15):
        if time.time() > deadline - 15:
            break
        
        found = None
        
        # Strategy 1: Local search with many restarts
        max_restarts = 100
        for attempt in range(max_restarts):
            if time.time() > deadline - 8:
                break
            seed = random.randint(0, 2**31)
            result = local_search(k, max_iter=5000, seed=seed, deadline=deadline - 5)
            if result is not None:
                found = result
                break
        
        if found is None:
            # Strategy 2: Simulated annealing
            for attempt in range(50):
                if time.time() > deadline - 8:
                    break
                seed = random.randint(0, 2**31)
                result = simulated_annealing(k, max_iter=20000, seed=seed, deadline=deadline - 5)
                if result is not None:
                    found = result
                    break
        
        if found is not None:
            # Verify
            if is_valid(found):
                best_result = found
                best_k = k
                break
    
    if best_result is None:
        # Fallback: use all distinct colors (always valid)
        return list(range(n))
    
    # Ensure Python ints
    return [int(c) for c in best_result]

# EVOLVE-BLOCK-END
