"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _has_bicolored_cycle(adj, colors, k):
    """
    Returns True if any pair of colors induces a cycle.
    Uses union-find with early exit.
    """
    n = P['n']
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            # Collect nodes with color c1 or c2
            nodes = set()
            for i in range(n):
                if colors[i] == c1 or colors[i] == c2:
                    nodes.add(i)
            if len(nodes) < 3:
                continue
            # Union-find
            parent = {i: i for i in nodes}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                parent[rx] = ry
                return True
            for u in nodes:
                for v in adj[u]:
                    if v in nodes and u < v:
                        if not union(u, v):
                            return True
    return False

def _count_proper_conflicts(adj, colors):
    return sum(1 for u, v in P['edges'] if colors[u] == colors[v])

def construct():
    start_time = time.time()
    deadline = start_time + 85
    adj = _build_adjacency()
    n = P['n']
    
    # Try increasing k until we find a valid coloring
    for k in range(3, 8):
        if time.time() > deadline - 5:
            break
        # Multiple random restarts
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            # Random initial coloring
            colors = [random.randint(0, k - 1) for _ in range(n)]
            
            # Simulated annealing / local search
            temp = 50.0
            min_temp = 0.1
            cool_rate = 0.98
            current_conflicts = _count_proper_conflicts(adj, colors)
            current_has_cycle = _has_bicolored_cycle(adj, colors, k)
            
            for step in range(3000):
                if time.time() > deadline - 5:
                    break
                
                # Check if valid
                if current_conflicts == 0 and not current_has_cycle:
                    return colors
                
                # Pick a random vertex to recolor
                v = random.randint(0, n - 1)
                old_color = colors[v]
                new_color = random.randint(0, k - 1)
                while new_color == old_color:
                    new_color = random.randint(0, k - 1)
                
                colors[v] = new_color
                new_conflicts = _count_proper_conflicts(adj, colors)
                new_has_cycle = _has_bicolored_cycle(adj, colors, k)
                
                # Score: conflicts + big penalty for cycles
                old_score = current_conflicts + (100 if current_has_cycle else 0)
                new_score = new_conflicts + (100 if new_has_cycle else 0)
                delta = new_score - old_score
                
                if delta <= 0 or random.random() < pow(2.71828, -delta / temp):
                    current_conflicts = new_conflicts
                    current_has_cycle = new_has_cycle
                else:
                    colors[v] = old_color
                
                temp *= cool_rate
                if temp < min_temp:
                    break
    
    # Fallback: greedy coloring with backtracking
    if time.time() < deadline - 10:
        colors = [-1] * n
        # Greedy with MRV (minimum remaining values)
        for i in range(n):
            used_colors = set()
            for v in adj[i]:
                if colors[v] != -1:
                    used_colors.add(colors[v])
            # Try colors 0, 1, 2, ...
            for c in range(50):
                if c not in used_colors:
                    colors[i] = c
                    break
            else:
                colors[i] = 0  # fallback
        
        # Verify and fix
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            # Check validity
            proper = all(colors[u] != colors[v] for u, v in P['edges'])
            if proper:
                k_used = max(colors) + 1
                if not _has_bicolored_cycle(adj, colors, k_used):
                    return colors
            # Random perturbation
            v = random.randint(0, n - 1)
            colors[v] = random.randint(0, 49)
    
    # Final fallback: distinct colors
    return list(range(n))
# EVOLVE-BLOCK-END
