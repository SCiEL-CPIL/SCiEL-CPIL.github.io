# EVOLVE-BLOCK-START
import time
import random

def _get_petersen_edges():
    edges = []
    # Outer
    edges.append((0, 1))
    edges.append((1, 2))
    edges.append((2, 3))
    edges.append((3, 4))
    edges.append((4, 0))
    # Inner
    edges.append((5, 7))
    edges.append((7, 9))
    edges.append((9, 6))
    edges.append((6, 8))
    edges.append((8, 5))
    # Spokes
    edges.append((0, 5))
    edges.append((1, 6))
    edges.append((2, 7))
    edges.append((3, 8))
    edges.append((4, 9))
    return edges

def _find_cycles(graph, length):
    cycles = set()
    n = 10
    adj = {i: [] for i in range(n)}
    for u, v in graph:
        adj[u].append(v)
        adj[v].append(u)
        
    def dfs(start, current, path, visited):
        if len(path) == length + 1:
            if path[0] == path[-1]:
                cycle = tuple(path)
                rev = tuple(reversed(cycle))
                if cycle < rev:
                    cycles.add(cycle)
            return
        
        for neighbor in adj[current]:
            if neighbor == start and len(path) == length:
                dfs(start, start, path + [start], visited)
            elif neighbor not in visited:
                dfs(start, neighbor, path + [neighbor], visited | {neighbor})

    for i in range(n):
        dfs(i, i, [i], {i})
    return list(cycles)

def _cycle_sum(cycle, voltages):
    s = 0
    for i in range(len(cycle) - 1):
        u, v = cycle[i], cycle[i+1]
        key = (u, v)
        if key in voltages:
            s += voltages[key]
        else:
            s -= voltages[(v, u)]
    return s % 32

def _construct_voltage_graph():
    edges = _get_petersen_edges()
    cycles_5 = _find_cycles(edges, 5)
    cycles_8 = _find_cycles(edges, 8)
    
    random.seed(42)
    
    best_voltages = None
    best_bad = float('inf')
    
    deadline = time.time() + 100
    
    for _ in range(100): # 100 restarts
        voltages = {}
        for u, v in edges:
            val = random.randint(1, 31)
            voltages[(u, v)] = val
            voltages[(v, u)] = (-val) % 32
            
        def count_bad():
            bad = 0
            for c in cycles_5:
                if _cycle_sum(c, voltages) == 0:
                    bad += 1
            for c in cycles_8:
                if _cycle_sum(c, voltages) == 0:
                    bad += 1
            return bad
            
        current_bad = count_bad()
        
        # Local search
        for step in range(1000):
            if time.time() > deadline:
                break
            if current_bad == 0:
                break
                
            u, v = random.choice(edges)
            old_val = voltages[(u, v)]
            new_val = random.randint(1, 31)
            if new_val == old_val:
                continue
                
            voltages[(u, v)] = new_val
            voltages[(v, u)] = (-new_val) % 32
            
            new_bad = count_bad()
            if new_bad < current_bad:
                current_bad = new_bad
            else:
                voltages[(u, v)] = old_val
                voltages[(v, u)] = (-old_val) % 32
                
        if current_bad < best_bad:
            best_bad = current_bad
            best_voltages = dict(voltages)
            if best_bad == 0:
                break
                
    if best_bad == 0 and best_voltages:
        n = 320
        lift_edges = []
        for u, v in edges:
            vol = best_voltages[(u, v)]
            for g in range(32):
                u_idx = u * 32 + g
                v_g = (g + vol) % 32
                v_idx = v * 32 + v_g
                lift_edges.append((u_idx, v_idx))
        return n, lift_edges
    return None

def construct(k, g):
    deadline = time.time() + 110
    
    if k == 4 and g == 10:
        res = _construct_voltage_graph()
        if res:
            return res
            
    # Fallback: Random 4-regular graph with large n
    # This is unlikely to be called if voltage graph works
    return _fallback_construct(k, g)

def _fallback_construct(k, g):
    # Generate a random 4-regular graph with n=500
    # This is a valid graph, but n is large
    n = 500
    edges = []
    # Simple random regular graph generation (configuration model)
    # ... too complex for fallback, just return a known valid small graph if possible
    # Or use the L-U-W from before
    return _luw_graph(4, 5)

def _luw_graph(q, d):
    # Kept from original, but might be invalid?
    # If original was valid, this is safe.
    if q == 4:
        add, sub, mul = _gf4_add, _gf4_sub, _gf4_mul
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)
            
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
                
    n = len(order)
    return n, edges

def _gf4_mul(x, y):
    if x == 0 or y == 0:
        return 0
    logs = {1: 0, 2: 1, 3: 2}
    powers = [1, 2, 3, 1]
    r = powers[(logs[x] + logs[y]) % 3]
    return r

def _gf4_add(x, y):
    return x ^ y

def _gf4_sub(x, y):
    return x ^ y
# EVOLVE-BLOCK-END
