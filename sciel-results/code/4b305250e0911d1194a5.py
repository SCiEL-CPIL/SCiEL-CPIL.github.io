import time
import random
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
edges = P['edges']

adj = [[] for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def count_bicolored_cycles(coloring, k):
    """Count number of bicolored cycles using union-find per color pair."""
    total = 0
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
                        total += 1
                        break
    return total

def validate(coloring):
    k = max(coloring) + 1
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    return count_bicolored_cycles(coloring, k) == 0

def make_initial_3coloring():
    """Create a proper 3-coloring of the 5x10 grid-like graph."""
    coloring = [0] * n
    for i in range(5):
        for j in range(10):
            v = i * 10 + j
            coloring[v] = (i + j) % 3
    return coloring

def construct():
    start = time.time()
    deadline = 75.0  # safety margin before 90s kill
    
    random.seed(42)
    
    # Start with a valid proper 3-coloring
    best_coloring = make_initial_3coloring()
    best_cycles = count_bicolored_cycles(best_coloring, 3)
    
    if best_cycles == 0:
        return [int(c) for c in best_coloring]
    
    # Simulated annealing to break bicolored cycles
    k = 3
    T = 1.0
    cooling = 0.9995
    
    current = list(best_coloring)
    current_cycles = best_cycles
    current_proper = True
    
    # Track best valid solution
    best_valid = None
    best_valid_cycles = current_cycles
    
    iteration = 0
    while time.time() - start < deadline:
        iteration += 1
        
        # Pick a random vertex
        v = random.randint(0, n - 1)
        
        # Pick a new color (different from current)
        old_color = current[v]
        new_color = random.choice([c for c in range(k) if c != old_color])
        
        # Check if this creates an adjacency conflict
        proper_after = True
        for u in adj[v]:
            if current[u] == new_color:
                proper_after = False
                break
        
        if not proper_after:
            # Skip improper moves (or accept with very low prob)
            if random.random() < 0.01:
                pass  # allow occasional improper
            else:
                continue
        
        # Compute cycle count change
        current[v] = new_color
        new_cycles = count_bicolored_cycles(current, k)
        delta_cycles = new_cycles - current_cycles
        
        # Accept if better, or with probability exp(-delta/T)
        if delta_cycles <= 0:
            current_cycles = new_cycles
            if current_cycles == 0 and proper_after:
                best_valid = list(current)
                break
        else:
            if random.random() < 2.718281828 ** (-delta_cycles / T):
                current_cycles = new_cycles
        
        # Cool down
        T *= cooling
        
        # Periodically try to repair if we have a valid solution
        if best_valid is not None:
            break
    
    if best_valid is not None:
        if validate(best_valid):
            return [int(c) for c in best_valid]
    
    # If SA didn't find valid, try targeted repair
    # Find vertices in bicolored cycles and try recoloring them
    for attempt in range(200):
        if time.time() - start > deadline:
            break
        candidate = list(best_coloring)
        
        # Find all bicolored cycles
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
                
                cycle_vertices = []
                for u, v in edges:
                    cu, cv = candidate[u], candidate[v]
                    if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                        if not union(u, v):
                            cycle_vertices.extend([u, v])
                
                if cycle_vertices:
                    # Try recoloring one vertex in the cycle
                    v = random.choice(cycle_vertices)
                    old_c = candidate[v]
                    new_c = random.choice([c for c in range(k) if c != old_c])
                    proper = all(candidate[u] != new_c for u in adj[v])
                    if proper:
                        candidate[v] = new_c
                        if count_bicolored_cycles(candidate, k) == 0:
                            if validate(candidate):
                                return [int(c) for c in candidate]
                        candidate[v] = old_c
    
    # Fallback: try all pairs of vertex recolorings
    for v1 in range(n):
        if time.time() - start > deadline:
            break
        for v2 in range(v1 + 1, n):
            if time.time() - start > deadline:
                break
            candidate = list(best_coloring)
            c1_old = candidate[v1]
            c2_old = candidate[v2]
            
            for nc1 in range(k):
                if nc1 == c1_old:
                    continue
                proper1 = all(candidate[u] != nc1 for u in adj[v1])
                if not proper1:
                    continue
                candidate[v1] = nc1
                for nc2 in range(k):
                    if nc2 == c2_old:
                        continue
                    proper2 = all(candidate[u] != nc2 for u in adj[v2])
                    if not proper2:
                        continue
                    candidate[v2] = nc2
                    if count_bicolored_cycles(candidate, k) == 0:
                        if validate(candidate):
                            return [int(c) for c in candidate]
            candidate[v1] = c1_old
    
    # Final fallback: use 4 colors with a simple pattern
    coloring = [0] * n
    for i in range(5):
        for j in range(10):
            v = i * 10 + j
            coloring[v] = (i * 2 + j) % 4
    
    # Verify and return
    if validate(coloring):
        return [int(c) for c in coloring]
    
    # Absolute fallback
    return [int(i) for i in range(n)]
