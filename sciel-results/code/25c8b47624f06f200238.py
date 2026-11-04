"""Voltage graph lift approach for the cage problem - optimized."""
import time
import random
from collections import deque

_START_TIME = None
_DEADLINE = 115

def _time_left():
    if _START_TIME is None:
        return _DEADLINE
    return max(0, _DEADLINE - (time.time() - _START_TIME))

def construct(k, g):
    global _START_TIME
    _START_TIME = time.time()
    
    # K_{3,3} voltage lift: 6*m vertices
    # Need girth >= 16, so eliminate all cycles of length < 16
    # K_{3,3} has cycles of length 4 and 6 only
    # So we need voltage sums != 0 for all 4-cycles and 6-cycles
    
    # 4-cycles: A_i1 -> B_j1 -> A_i2 -> B_j2 -> A_i1
    # Voltage: v[i1][j1] - v[i2][j1] + v[i2][j2] - v[i1][j2] != 0 mod m
    # 
    # 6-cycles: A_i1 -> B_j1 -> A_i2 -> B_j2 -> A_i3 -> B_j3 -> A_i1
    # where i1,i2,i3 are distinct and j1,j2,j3 are distinct
    # Voltage: v[i1][j1] - v[i2][j1] + v[i2][j2] - v[i3][j2] + v[i3][j3] - v[i1][j3] != 0 mod m
    
    # Precompute all 4-cycles and 6-cycles
    cycles_4 = []
    for i1 in range(3):
        for i2 in range(i1+1, 3):
            for j1 in range(3):
                for j2 in range(j1+1, 3):
                    cycles_4.append((i1, j1, i2, j2))
    
    cycles_6 = []
    for i1 in range(3):
        for i2 in range(3):
            if i2 == i1: continue
            for i3 in range(3):
                if i3 == i1 or i3 == i2: continue
                for j1 in range(3):
                    for j2 in range(3):
                        if j2 == j1: continue
                        for j3 in range(3):
                            if j3 == j1 or j3 == j2: continue
                            cycles_6.append((i1, j1, i2, j2, i3, j3))
    
    # Try m values, use random + local search
    best_result = None
    
    for m in range(150, 160):
        if _time_left() < 5:
            break
        
        for trial in range(200):
            if _time_left() < 3:
                break
            
            # Random initialization
            voltages = [[random.randint(0, m-1) for _ in range(3)] for _ in range(3)]
            
            # Check violations
            viol = _count_violations_fast(voltages, m, cycles_4, cycles_6)
            
            if viol == 0:
                n = 6 * m
                edges = _build_lift(voltages, m)
                if _verify_girth_fast(n, edges, g):
                    return n, edges
            
            # Local search: mutate one voltage at a time
            for step in range(50):
                if _time_left() < 3:
                    break
                
                i = random.randint(0, 2)
                j = random.randint(0, 2)
                old = voltages[i][j]
                voltages[i][j] = random.randint(0, m-1)
                
                new_viol = _count_violations_fast(voltages, m, cycles_4, cycles_6)
                
                if new_viol <= viol:
                    viol = new_viol
                else:
                    voltages[i][j] = old
                
                if viol == 0:
                    n = 6 * m
                    edges = _build_lift(voltages, m)
                    if _verify_girth_fast(n, edges, g):
                        return n, edges
                    break
    
    # Fallback: LUW construction
    return _construct_luw(k, g)


def _count_violations_fast(voltages, m, cycles_4, cycles_6):
    """Count violations for all short cycles."""
    violations = 0
    m_val = m
    
    for (i1, j1, i2, j2) in cycles_4:
        s = voltages[i1][j1] - voltages[i2][j1] + voltages[i2][j2] - voltages[i1][j2]
        if s % m_val == 0:
            violations += 1
    
    for (i1, j1, i2, j2, i3, j3) in cycles_6:
        s = (voltages[i1][j1] - voltages[i2][j1] + 
             voltages[i2][j2] - voltages[i3][j2] + 
             voltages[i3][j3] - voltages[i1][j3])
        if s % m_val == 0:
            violations += 1
    
    return violations


def _build_lift(voltages, m):
    """Build the voltage lift graph."""
    edges = []
    for i in range(3):
        for j in range(3):
            v = voltages[i][j]
            for gval in range(m):
                u_idx = i * m + gval
                v_idx = (3 + j) * m + (gval + v) % m
                edges.append((u_idx, v_idx))
    return edges


def _verify_girth_fast(n, edges, g):
    """Verify girth >= g using non-backtracking walks."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        # BFS with parent tracking
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] * 2 >= g:
                break
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True


def _construct_luw(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph."""
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
