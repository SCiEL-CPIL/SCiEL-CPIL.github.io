"""Rewritten program for bicolored-cycle-free coloring using local search."""
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_forest_fast(coloring, n, edges):
    """
    Fast check: for each pair of colors, check if the induced subgraph is a forest.
    Returns set of vertices in bicolored cycles, or empty set if all forests.
    """
    # Build color-pair -> list of edges mapping
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            key = (min(cu, cv), max(cu, cv))
            pair_edges[key].append((u, v))
    
    cycle_vertices = set()
    
    for (c1, c2), edge_list in pair_edges.items():
        # Union-find for this pair
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
        
        has_cycle = False
        for u, v in edge_list:
            if not union(u, v):
                has_cycle = True
                cycle_vertices.add(u)
                cycle_vertices.add(v)
        
    return cycle_vertices

def _count_violations_fast(coloring, n, edges):
    """
    Returns (proper_violation_count, bicolored_cycle_vertices).
    """
    proper_violations = 0
    for u, v in edges:
        if coloring[u] == coloring[v]:
            proper_violations += 1
    
    bicolored_cycle_vertices = _check_bicolored_forest_fast(coloring, n, edges)
    
    return proper_violations, bicolored_cycle_vertices

def _local_search(n, edges, k, deadline, adj):
    """
    Attempts to find a valid coloring with k colors using local search.
    """
    # Initialize randomly
    coloring = [random.randint(0, k - 1) for _ in range(n)]
    
    best_coloring = None
    best_score = float('inf')
    
    max_steps = 30000
    step = 0
    no_improve = 0
    max_no_improve = 2000
    
    while step < max_steps and time.time() < deadline:
        proper_violations, bicolored_cycle_vertices = _count_violations_fast(coloring, n, edges)
        
        score = proper_violations * 10 + len(bicolored_cycle_vertices)
        
        if score < best_score:
            best_score = score
            best_coloring = coloring[:]
            no_improve = 0
        else:
            no_improve += 1
            
        if score == 0:
            return best_coloring
            
        if no_improve > max_no_improve:
            # Restart with new random coloring
            coloring = [random.randint(0, k - 1) for _ in range(n)]
            no_improve = 0
            max_no_improve = max(1000, max_no_improve // 2)
            continue
        
        # Choose a vertex to recolor
        candidate_vertices = set(bicolored_cycle_vertices)
        if proper_violations > 0:
            # Add vertices from proper violations
            for u, v in edges:
                if coloring[u] == coloring[v]:
                    candidate_vertices.add(u)
                    candidate_vertices.add(v)
        
        if not candidate_vertices:
            candidate_vertices = set(range(n))
            
        v = random.choice(list(candidate_vertices))
        
        # Choose a new color
        neighbor_colors = set(coloring[u] for u in adj[v])
        available_colors = [c for c in range(k) if c not in neighbor_colors]
        
        if not available_colors:
            new_color = random.randint(0, k - 1)
        else:
            new_color = random.choice(available_colors)
            
        coloring[v] = new_color
        step += 1
        
    if best_coloring is not None:
        pv, cv = _count_violations_fast(best_coloring, n, edges)
        if pv == 0 and len(cv) == 0:
            return best_coloring
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try increasing number of colors
    for k in range(4, n + 1):
        if time.time() > deadline:
            break
        
        # Try multiple restarts for each k
        found = None
        num_restarts = 30 if k <= 6 else 15
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            result = _local_search(n, edges, k, deadline, adj)
            if result is not None:
                found = result
                break
        
        if found is not None:
            return list(found)
            
    # Fallback: use all different colors (always valid)
    return list(range(n))
