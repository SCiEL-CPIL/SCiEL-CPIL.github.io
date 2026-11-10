"""Few colors without any bicolored cycle – strong local search + SA polish."""
import time
import random
import sys

# Fixed graph data (degree-4 pinned graph, 50 vertices)
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']
ADJ = [[] for _ in range(N)]
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)

def count_distinct(colors):
    """Number of distinct color labels used."""
    return len(set(colors))

def has_bicolored_cycle(colors):
    """Check if any pair of colors induces a subgraph with a cycle (using union-find)."""
    # Group edges by color pair
    from collections import defaultdict
    pair_edges = defaultdict(list)
    for u, v in EDGES:
        cu, cv = colors[u], colors[v]
        if cu != cv:
            key = (min(cu, cv), max(cu, cv))
            pair_edges[key].append((u, v))
    
    # For each color pair, check if the induced subgraph has a cycle
    for key, edges in pair_edges.items():
        # Union-Find
        parent = list(range(N))
        rank = [0] * N
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return True  # cycle detected
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return False
        
        for u, v in edges:
            if union(u, v):
                return True
    return False

def is_valid(colors):
    """Check proper coloring and no bicolored cycles."""
    for u, v in EDGES:
        if colors[u] == colors[v]:
            return False
    return not has_bicolored_cycle(colors)

def cost(colors, k):
    """Cost function: proper coloring violations + bicolored cycle penalty."""
    # Proper coloring violations
    color_viol = 0
    for u, v in EDGES:
        if colors[u] == colors[v]:
            color_viol += 1
    
    # Bicolored cycles
    cycle_count = 0
    from collections import defaultdict
    pair_edges = defaultdict(list)
    for u, v in EDGES:
        cu, cv = colors[u], colors[v]
        if cu != cv:
            key = (min(cu, cv), max(cu, cv))
            pair_edges[key].append((u, v))
    
    for key, edges in pair_edges.items():
        parent = list(range(N))
        rank = [0] * N
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return True
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return False
        
        for u, v in edges:
            if union(u, v):
                cycle_count += 1
                break
    
    return color_viol * 10 + cycle_count

def local_search(colors, k, max_iter=2000):
    """Hill-climbing local search to minimize cost."""
    current_cost = cost(colors, k)
    best_colors = list(colors)
    best_cost = current_cost
    
    for _ in range(max_iter):
        improved = False
        # Try changing each vertex to a random color in [0, k)
        for _ in range(N * 2):
            v = random.randint(0, N - 1)
            old = colors[v]
            new = random.randint(0, k - 1)
            if new == old:
                continue
            colors[v] = new
            new_cost = cost(colors, k)
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = list(colors)
                improved = True
            else:
                colors[v] = old
        if not improved:
            break
    return best_colors, best_cost

def simulated_annealing(colors, k, t0=1.0, t_end=0.01, steps=5000):
    """Simulated annealing to escape local minima."""
    current = list(colors)
    current_cost = cost(current, k)
    best = list(current)
    best_cost = current_cost
    
    for i in range(steps):
        t = t0 * (t_end / t0) ** (i / steps)
        v = random.randint(0, N - 1)
        old = current[v]
        new = random.randint(0, k - 1)
        if new == old:
            continue
        current[v] = new
        new_cost = cost(current, k)
        delta = new_cost - current_cost
        if delta <= 0 or random.random() < __import__('math').exp(-delta / max(t, 1e-10)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best = list(current)
        else:
            current[v] = old
    return best, best_cost

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    best_result = None
    best_k = 50
    
    # Try k from 3 upward
    for k in range(3, 8):
        if time.time() > deadline:
            break
        
        found = False
        # Multiple random restarts
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Random initial coloring with k colors
            colors = [random.randint(0, k - 1) for _ in range(N)]
            
            # Local search
            colors, c = local_search(colors, k, max_iter=1000)
            
            if c == 0:
                # Verify
                if is_valid(colors):
                    best_k = k
                    best_result = colors
                    found = True
                    break
            
            # If not valid, try SA
            if time.time() > deadline:
                break
            colors, c = simulated_annealing(colors, k, t0=1.0, t_end=0.01, steps=2000)
            if c == 0 and is_valid(colors):
                best_k = k
                best_result = colors
                found = True
                break
        
        if found:
            break
    
    if best_result is not None:
        # Ensure Python int types
        return [int(c) for c in best_result]
    
    # Fallback: use seed program's answer (all distinct)
    return list(range(N))

# EVOLVE-BLOCK-END
