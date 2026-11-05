import random
import time
import math
from collections import deque

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Base graph: Petersen (10 vertices, 15 edges)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Generate all closed non-backtracking walks of length 3 to g-1 from every vertex
    walks = []
    for start_v in range(n_base):
        queue = deque()
        for next_v, e_id in adj[start_v]:
            u, v = edges[e_id]
            sign = 1 if u == start_v else -1
            queue.append((next_v, e_id, 1, [(e_id, sign)]))
        
        while queue:
            curr_v, prev_e, length, path = queue.popleft()
            if length >= g:
                continue
            if curr_v == start_v and length >= 3:
                walks.append(path)
            for next_v, e_id in adj[curr_v]:
                if e_id == prev_e:
                    continue
                u, v = edges[e_id]
                sign = 1 if u == curr_v else -1
                new_path = path + [(e_id, sign)]
                queue.append((next_v, e_id, length + 1, new_path))
    
    unique_walks = set()
    for w in walks:
        unique_walks.add(tuple(w))
    walks = list(unique_walks)
    
    # Pre-compute coefficient vectors (for Abelian fallback) and edge sets (for Non-Abelian)
    walk_coeffs = []
    walk_edge_sets = []
    for w in walks:
        coeffs = [0] * len(edges)
        for e_id, sign in w:
            coeffs[e_id] += sign
        nonzero = [(i, c) for i, c in enumerate(coeffs) if c != 0]
        if nonzero:
            walk_coeffs.append(nonzero)
            walk_edge_sets.append(w)

    # Define groups to try
    def get_dihedral_group(k):
        n = 2 * k
        table = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i < k:
                    if j < k:
                        table[i][j] = (i + j) % k
                    else:
                        table[i][j] = k + (j - k - i) % k
                else:
                    if j < k:
                        table[i][j] = k + (i - k + j) % k
                    else:
                        table[i][j] = (j - k - (i - k)) % k
        inv = [0]*n
        for i in range(n):
            for j in range(n):
                if table[i][j] == 0:
                    inv[i] = j
                    break
        return table, inv, 0

    def get_A5_group():
        import itertools
        perms = []
        for p in itertools.permutations(range(5)):
            inv = 0
            for i in range(5):
                for j in range(i+1, 5):
                    if p[i] > p[j]:
                        inv += 1
            if inv % 2 == 0:
                perms.append(p)
        n = len(perms)
        perm_to_idx = {p: i for i, p in enumerate(perms)}
        table = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                composed = tuple(perms[i][perms[j][x]] for x in range(5))
                table[i][j] = perm_to_idx[composed]
        inv = [0]*n
        for i in range(n):
            for j in range(n):
                if table[i][j] == 0:
                    inv[i] = j
                    break
        return table, inv, 0

    groups_to_try = []
    groups_to_try.append((6, "D3", get_dihedral_group(3)))
    groups_to_try.append((10, "D5", get_dihedral_group(5)))
    groups_to_try.append((12, "D6", get_dihedral_group(6)))
    groups_to_try.append((14, "D7", get_dihedral_group(7)))
    groups_to_try.append((16, "D8", get_dihedral_group(8)))
    groups_to_try.append((18, "D9", get_dihedral_group(9)))
    groups_to_try.append((20, "D10", get_dihedral_group(10)))
    groups_to_try.append((22, "D11", get_dihedral_group(11)))
    groups_to_try.append((24, "D12", get_dihedral_group(12)))
    groups_to_try.append((26, "D13", get_dihedral_group(13)))
    groups_to_try.append((28, "D14", get_dihedral_group(14)))
    groups_to_try.append((30, "D15", get_dihedral_group(15)))
    groups_to_try.append((60, "A5", get_A5_group()))
    groups_to_try.sort(key=lambda x: x[0])

    best_n = float('inf')
    best_edges = None

    for order, name, (table, inv, id_elem) in groups_to_try:
        if time.time() > deadline - 20.0:
            break
        
        candidate_n = 10 * order
        if candidate_n >= best_n:
            continue
            
        def solve_group(table, inv, id_elem):
            n_edges = 15
            n_group = len(table)
            
            def check_voltages(v):
                for w in walk_edge_sets:
                    prod = id_elem
                    for e_id, sign in w:
                        val = v[e_id]
                        if sign == -1:
                            val = inv[val]
                        prod = table[prod][val]
                    if prod == id_elem:
                        return False
                return True
            
            # Random Search with more attempts
            for _ in range(5000):
                if time.time() > deadline - 15.0:
                    return None
                v = [random.randint(0, n_group - 1) for _ in range(n_edges)]
                if check_voltages(v):
                    return v
            return None

        solution = solve_group(table, inv, id_elem)
        if solution:
            best_n = candidate_n
            best_edges = []
            for i, (u, v) in enumerate(edges):
                vol_idx = solution[i]
                for t in range(order):
                    u_lift = u * order + t
                    v_lift_idx = table[t][vol_idx]
                    v_lift = v * order + v_lift_idx
                    best_edges.append((u_lift, v_lift))
            break

    if best_edges is None:
        return _luw_fallback(k, g)
    
    return best_n, best_edges

def _luw_fallback(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw

if __name__ == "__main__":
    n, e = construct(3, 16)
    print(f"n={n}")
