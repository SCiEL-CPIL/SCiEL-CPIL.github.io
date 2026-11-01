import time
import random
import numpy as np

def _voltage_construct(k, g):
    """Voltage graph construction on K_6 base to get 5-regular graph of girth >= g."""
    n_base = 6
    edge_index = {}
    base_edges = []
    for u in range(n_base):
        for v in range(u + 1, n_base):
            edge_index[(u, v)] = len(base_edges)
            base_edges.append((u, v))
    n_edges = len(base_edges)  # 15
    
    # Generate all non-backtracking closed walks of length 3..(g-1) starting/ending at vertex 0
    walks_by_len = {L: [] for L in range(3, g)}
    
    def gen_walks(length):
        def dfs(pos, cur, prv, coeffs):
            if pos == length:
                if cur != 0 and prv != 0:
                    u, v = min(cur, 0), max(cur, 0)
                    idx = edge_index[(u, v)]
                    coeffs[idx] += -1
                    walks_by_len[length].append(tuple(coeffs))
                    coeffs[idx] -= -1
                return
            for nv in range(n_base):
                if nv == cur or nv == prv:
                    continue
                u, v = min(cur, nv), max(cur, nv)
                idx = edge_index[(u, v)]
                sign = 1 if cur < nv else -1
                coeffs[idx] += sign
                dfs(pos + 1, nv, cur, coeffs)
                coeffs[idx] -= sign
        for v1 in range(1, n_base):
            idx = edge_index[(0, v1)]
            coeffs = [0] * n_edges
            coeffs[idx] += 1
            dfs(1, v1, 0, coeffs)
    
    for L in range(3, g):
        gen_walks(L)
    
    walk_arrays = {}
    for L in range(3, g):
        if walks_by_len[L]:
            walk_arrays[L] = np.array(walks_by_len[L], dtype=np.int32)
        else:
            walk_arrays[L] = np.array([], dtype=np.int32).reshape(0, n_edges)
    
    # Try voltages for various m values
    # Moore bound for (5,9) is 428, so 6*m >= 428 => m >= 72
    m_candidates = [72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                    95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150,
                    155, 160, 165, 170, 175, 180, 186]
    
    for m in m_candidates:
        n_vertices = 6 * m
        if n_vertices < 428:
            continue
        
        for attempt in range(30000):
            voltages = np.random.randint(0, m, n_edges, dtype=np.int32)
            
            valid = True
            for L in range(3, g):
                arr = walk_arrays[L]
                if arr.size == 0:
                    continue
                sums = arr.dot(voltages)
                if np.any(sums % m == 0):
                    valid = False
                    break
            
            if valid:
                edges = []
                for u in range(n_base):
                    for v in range(u + 1, n_base):
                        idx = edge_index[(u, v)]
                        v_e = int(voltages[idx])
                        for i in range(m):
                            vid1 = u * m + i
                            j = (i + v_e) % m
                            vid2 = v * m + j
                            if vid1 < vid2:
                                edges.append((vid1, vid2))
                return n_vertices, edges
    
    return None

def _luw_construct(k, g):
    """LUW construction - fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    deadline = time.time() + 115
    
    if k != 5 or g != 9:
        return _luw_construct(k, g)
    
    # Try voltage construction first (fastest path to small n)
    result = _voltage_construct(k, g)
    if result is not None:
        n, edges = result
        if time.time() < deadline:
            return n, edges
    
    # Fallback to LUW
    return _luw_construct(k, g)
