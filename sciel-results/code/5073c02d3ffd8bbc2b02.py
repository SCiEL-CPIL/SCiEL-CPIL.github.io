import time
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

# Try to use math.lcm if available, else use _lcm
try:
    from math import lcm
except ImportError:
    lcm = _lcm

def construct():
    start_time = time.time()
    deadline = time.time() + 80.0  # 80s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            val = lcm(i, j)
            lcm_table[i][j] = val
            lcm_table[j][i] = val

    # Find forbidden triples
    forbidden_triples = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l_ij = lcm_table[i][j]
            for k in range(j + 1, n + 1):
                if lcm_table[i][k] == l_ij and lcm_table[j][k] == l_ij:
                    forbidden_triples.append((i, j, k))
    
    # Degree count
    degree = [0] * (n + 1)
    for a, b, c in forbidden_triples:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1

    selected = set(range(1, n + 1))
    
    # Greedy Removal
    while True:
        if time.time() > deadline:
            break
            
        active_triple = None
        for t in forbidden_triples:
            if t[0] in selected and t[1] in selected and t[2] in selected:
                active_triple = t
                break
        
        if active_triple is None:
            break
        
        a, b, c = active_triple
        # Remove node with max degree
        max_deg = -1
        node_to_remove = a
        for node in (a, b, c):
            if degree[node] > max_deg:
                max_deg = degree[node]
                node_to_remove = node
        
        selected.remove(node_to_remove)

    # Local Search: Add back nodes
    conflict_pairs = {i: set() for i in range(1, n + 1)}
    for a, b, c in forbidden_triples:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))
    
    added_any = True
    while added_any:
        if time.time() > deadline:
            break
        added_any = False
        for v in range(1, n + 1):
            if v in selected:
                continue
            if time.time() > deadline:
                break
            
            can_add = True
            for a, b in conflict_pairs[v]:
                if a in selected and b in selected:
                    can_add = False
                    break
            
            if can_add:
                selected.add(v)
                added_any = True
                
    return list(selected)
