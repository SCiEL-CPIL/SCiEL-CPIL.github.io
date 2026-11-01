"""Voltage graph lift approach for cage problem n(4,10)."""
import numpy as np
from itertools import product
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth >= g using voltage graph lifting
    with optimized voltage assignments over cyclic groups.
    
    Strategy: Use a small 4-regular base graph and lift it via voltage
    assignments over Z_m. Optimize voltages to maximize girth by ensuring
    no closed walk of length < g in the base has zero voltage sum.
    """
    if k != 4 or g != 10:
        return _luw_construct(k, g)
    
    # Try voltage lift with various base sizes and group orders
    # Target: minimize n = |V_base| * m while girth >= 10
    
    start_time = time.time()
    best_result = None
    
    # Try different base graph sizes and group orders
    # Base sizes: 8, 16, 20, 24
    # For each, try m values that give reasonable total vertex counts
    
    base_configs = [
        (16, [15, 16, 17, 18, 19, 20]),  # 16-vertex base
        (20, [12, 13, 14, 15, 16]),       # 20-vertex base
        (24, [10, 11, 12, 13]),           # 24-vertex base
    ]
    
    for base_size, m_values in base_configs:
        for m in m_values:
            if time.time() - start_time > 90:
                break
            result = _try_voltage_lift(base_size, m, g)
            if result is not None:
                n, edges = result
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
        if time.time() - start_time > 90:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to LUW
    return _luw_construct(k, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) construction."""
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _get_base_graph(n_base):
    """
    Get a 4-regular base graph on n_base vertices.
    Uses structured constructions for known sizes.
    """
    if n_base == 16:
        # 16-vertex 4-regular graph: use the 4-dimensional hypercube Q4
        # Q4 has vertices 0..15, edges between vertices differing in one bit
        edges = []
        for v in range(16):
            for bit in range(4):
                u = v ^ (1 << bit)
                if v < u:
                    edges.append((v, u))
        return edges
    elif n_base == 20:
        # 20-vertex 4-regular: use circulant C_20(1, 2, 5, 10) or similar
        # Let's use a structured 4-regular graph
        edges = []
        for i in range(20):
            for d in [1, 2, 7, 9]:
                j = (i + d) % 20
                if i < j:
                    edges.append((i, j))
        # Verify it's 4-regular
        degree = [0] * 20
        for i, j in edges:
            degree[i] += 1
            degree[j] += 1
        if all(d == 4 for d in degree):
            return edges
        # Fallback: use 16-vertex base
        return _get_base_graph(16)
    elif n_base == 24:
        # 24-vertex 4-regular: circulant C_24(1, 2, 6, 12)
        edges = []
        for i in range(24):
            for d in [1, 2, 6, 11]:
                j = (i + d) % 24
                if i < j:
                    edges.append((i, j))
        degree = [0] * 24
        for i, j in edges:
            degree[i] += 1
            degree[j] += 1
        if all(d == 4 for d in degree):
            return edges
        return _get_base_graph(16)
    else:
        # Fallback to 16
        return _get_base_graph(16)


def _voltage_lift_girth_check(base_edges, voltages, m, g):
    """
    Check if the voltage lift has girth >= g.
    A cycle of length L in the lift corresponds to a closed walk of length L
    in the base with voltage sum 0 mod m.
    
    We check all closed walks of length < g in the base.
    """
    n_base = max(max(e) for e in base_edges) + 1 if base_edges else 0
    if n_base == 0:
        return False
    
    # Build adjacency with voltages
    # adj[i] = list of (j, voltage)
    adj = [[] for _ in range(n_base)]
    for (i, j) in base_edges:
        v = voltages[(i, j) if (i, j) in voltages else (j, i)]
        adj[i].append((j, v))
        adj[j].append((i, (-v) % m))  # Reverse edge has negative voltage
    
    # Check for closed walks of length 2 to g-1 with zero voltage sum
    # Use BFS/DFS to find closed walks
    # For efficiency, use matrix power approach or iterative deepening
    
    # For small girth checks, we can enumerate walks
    # Walk of length L: sequence v0, v1, ..., vL=v0
    # Voltage sum: sum of voltages along the walk
    
    # This is expensive for large L, but g=10 is manageable
    # Use dynamic programming: dp[L][v][s] = True if there's a walk of length L ending at v with voltage sum s
    
    # But m can be large, so let's use a different approach
    # Check cycles directly using BFS from each vertex
    
    for start in range(n_base):
        # BFS to find all walks of length < g starting and ending at start
        # State: (current_vertex, voltage_sum, length)
        # We want to detect if any walk of length L < g returns to start with voltage_sum = 0
        
        # Use a set to track visited states to avoid infinite loops
        # State: (vertex, voltage_sum, length)
        visited = set()
        queue = [(start, 0, 0, None)]  # (vertex, vol_sum, length, prev_vertex)
        
        while queue:
            new_queue = []
            for (v, vol, length, prev) in queue:
                if length >= g - 1:
                    continue
                for (u, edge_vol) in adj[v]:
                    if u == prev and length > 0:
                        continue  # No backtracking
                    new_vol = (vol + edge_vol) % m
                    new_length = length + 1
                    state = (u, new_vol, new_length)
                    if state in visited:
                        continue
                    visited.add(state)
                    if u == start and new_length >= 2 and new_vol == 0:
                        return False  # Found a cycle of length < g
                    new_queue.append((u, new_vol, new_length, v))
            queue = new_queue
    
    return True


def _try_voltage_lift(n_base, m, g, time_limit=8):
    """
    Try to find voltage assignments for a base graph lifted over Z_m
    such that the resulting graph has girth >= g.
    
    Returns (n, edges) if successful, None otherwise.
    """
    start_time = time.time()
    base_edges = _get_base_graph(n_base)
    if not base_edges:
        return None
    
    n_total = n_base * m
    
    # Get all edges as a list for voltage assignment
    edge_list = list(base_edges)
    n_edges = len(edge_list)
    
    # We need to assign voltages to each edge in Z_m
    # Use greedy + local search approach
    
    # Initialize with random voltages
    np.random.seed(42)
    best_voltages = None
    best_girth = 0
    
    # Try multiple random initializations
    for trial in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        voltages = {}
        for (i, j) in edge_list:
            voltages[(i, j)] = np.random.randint(1, m)  # Avoid 0 for now
            voltages[(j, i)] = (-voltages[(i, j)]) % m
        
        # Local search: try to improve by changing one voltage at a time
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try changing each edge's voltage
            for idx, (i, j) in enumerate(edge_list):
                if time.time() - start_time > time_limit:
                    break
                old_vol = voltages[(i, j)]
                # Try a few random new voltages
                for _ in range(3):
                    new_vol = np.random.randint(0, m)
                    if new_vol == old_vol:
                        continue
                    voltages[(i, j)] = new_vol
                    voltages[(j, i)] = (-new_vol) % m
                    
                    if _voltage_lift_girth_check(base_edges, voltages, m, g):
                        best_voltages = dict(voltages)
                        best_girth = g
                        improved = True
                        break
                    else:
                        # Revert
                        voltages[(i, j)] = old_vol
                        voltages[(j, i)] = (-old_vol) % m
        
        if best_voltages is not None:
            break
    
    if best_voltages is None:
        return None
    
    # Construct the lifted graph
    edges = []
    for (i, j) in base_edges:
        v = best_voltages[(i, j)]
        for s in range(m):
            u = (s + v) % m
            # Vertex (i, s) connects to (j, u)
            vi = i * m + s
            vj = j * m + u
            if vi < vj:
                edges.append((vi, vj))
            else:
                edges.append((vj, vi))
    
    # Deduplicate edges
    edge_set = set()
    unique_edges = []
    for e in edges:
        key = (min(e), max(e))
        if key not in edge_set:
            edge_set.add(key)
            unique_edges.append(key)
    
    return n_total, unique_edges


# EVOLVE-BLOCK-END
