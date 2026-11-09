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

def _count_conflicts(adj, colors, k):
    """
    Returns (proper_conflicts, bicolored_cycle_pairs).
    """
    n = P['n']
    proper_conflicts = 0
    for u, v in P['edges']:
        if colors[u] == colors[v]:
            proper_conflicts += 1
            
    bicolored_cycle_pairs = 0
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            nodes = [i for i in range(n) if colors[i] in (c1, c2)]
            if len(nodes) < 3:
                continue
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
            has_cycle = False
            for u in nodes:
                for v in adj[u]:
                    if v in nodes and u < v:
                        if not union(u, v):
                            has_cycle = True
                            break
                if has_cycle:
                    break
            if has_cycle:
                bicolored_cycle_pairs += 1
    return proper_conflicts, bicolored_cycle_pairs

def construct():
    start_time = time.time()
    deadline = start_time + 85
    adj = _build_adjacency()
    n = P['n']
    best_solution = None
    best_score = float('inf')
    
    for k in range(3, 10):
        if time.time() > deadline:
            break
        for r in range(50):
            if time.time() > deadline:
                break
            colors = [random.randint(0, k - 1) for _ in range(n)]
            temp = 100.0
            min_temp = 0.01
            cool_rate = 0.95
            current_pc, current_bc = _count_conflicts(adj, colors, k)
            current_score = current_pc + current_bc
            
            for step in range(2000):
                if time.time() > deadline:
                    break
                if current_score == 0:
                    return colors
                v = random.randint(0, n - 1)
                old_color = colors[v]
                new_color = random.randint(0, k - 1)
                while new_color == old_color:
                    new_color = random.randint(0, k - 1)
                colors[v] = new_color
                new_pc, new_bc = _count_conflicts(adj, colors, k)
                new_score = new_pc + new_bc
                delta = new_score - current_score
                if delta <= 0 or random.random() < pow(2.71828, -delta / temp):
                    current_score = new_score
                else:
                    colors[v] = old_color
                temp *= cool_rate
                if temp < min_temp:
                    break
            if current_score < best_score:
                best_score = current_score
                best_solution = colors[:]
        if best_score == 0:
            return best_solution
            
    return list(range(n))
# EVOLVE-BLOCK-END
