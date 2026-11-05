import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    if k != 4:
        return 0, []
        
    # Priority 1: Voltage graph construction
    result = _voltage_search(k, g, deadline)
    if result is not None:
        n, edges = result
        if _verify(n, edges, k, g):
            return n, edges
            
    # Priority 2: Fallback to LUW
    result = _luw_construct(k, g)
    if result is not None:
        n, edges = result
        if _verify(n, edges, k, g):
            return n, edges
            
    return 0, []

def _voltage_search(k, g, deadline):
    if k != 4:
        return None
        
    base_graphs = [
        ("K44", _get_k44_edges()),
    ]
    
    best_result = None
    best_n = float('inf')
    
    for base_name, base_edges in base_graphs:
        base_n = len(set(v for e in base_edges for v in e))
        max_m = min(60, int(350 / base_n) + 5)
        
        for m in range(20, max_m):
            if time.time() > deadline - 25:
                break
                
            n = m * base_n
            if n >= best_n:
                continue
                
            voltages = _find_voltages_improved(base_edges, m, g, deadline - 10)
            if voltages is not None:
                lift_edges = _build_lift(base_edges, voltages, m)
                if _verify(n, lift_edges, k, g):
                    if n < best_n:
                        best_n = n
                        best_result = (n, lift_edges)
                        
    return best_result

def _get_k44_edges():
    edges = []
    for i in range(4):
        for j in range(4):
            edges.append((i, j + 4))
    return edges

def _enumerate_walks_optimized(base_edges, max_len):
    base_n = len(set(v for e in base_edges for v in e))
    adj = [[] for _ in range(base_n)]
    edge_map = {}
    for idx, (u, v) in enumerate(base_edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        edge_map[(u, v)] = (idx, 1)
        edge_map[(v, u)] = (idx, -1)
        
    walks = []
    
    for L in range(3, max_len):
        for u, v in base_edges:
            idx, sign = edge_map[(u, v)]
            path = [(idx, sign)]
            prev = u
            cur = v
            
            def dfs(prev, cur, path, target_u, target_L):
                if len(path) == target_L:
                    if cur == target_u:
                        walks.append(path[:])
                    return
                for (next_v, eidx) in adj[cur]:
                    if next_v == prev:
                        continue
                    e_u, e_v = base_edges[eidx]
                    if e_u == cur and e_v == next_v:
                        s = 1
                    else:
                        s = -1
                    dfs(cur, next_v, path + [(eidx, s)], target_u, target_L)
            
            dfs(prev, cur, path, u, L)
            
    return walks

def _find_voltages_improved(base_edges, m, g, deadline):
    n_edges = len(base_edges)
    walks = _enumerate_walks_optimized(base_edges, g)
    
    if not walks:
        return [0] * n_edges
        
    walk_coeffs = []
    for walk in walks:
        coeffs = {}
        for eidx, sign in walk:
            coeffs[eidx] = coeffs.get(eidx, 0) + sign
        walk_coeffs.append(coeffs)
    
    def count_violations_fast(vols):
        count = 0
        for coeffs in walk_coeffs:
            s = 0
            for eidx, coeff in coeffs.items():
                s = (s + coeff * vols[eidx]) % m
            if s == 0:
                count += 1
        return count
    
    random.seed(12345)
    
    for restart in range(8):
        if time.time() > deadline - 2:
            break
        vols = [random.randint(0, m - 1) for _ in range(n_edges)]
        current_violations = count_violations_fast(vols)
        if current_violations == 0:
            return vols[:]
        
        T = 5.0 + restart
        T_min = 0.01
        cooling = 0.97
        steps = 0
        max_steps = 1500
        
        while T > T_min and steps < max_steps and time.time() < deadline:
            e_idx = random.randint(0, n_edges - 1)
            old_v = vols[e_idx]
            new_v = random.randint(0, m - 1)
            if new_v == old_v:
                continue
                
            vols[e_idx] = new_v
            new_violations = count_violations_fast(vols)
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < math.exp(-delta / max(T, 0.01)):
                current_violations = new_violations
            else:
                vols[e_idx] = old_v
            
            T *= cooling
            steps += 1
            
            if current_violations == 0:
                return vols[:]
    
    return None

def _build_lift(base_edges, voltages, m):
    base_n = len(set(v for e in base_edges for v in e))
    edges = []
    
    for (u, v), vol in zip(base_edges, voltages):
        for s in range(m):
            su = u * m + s
            sv = v * m + (s + vol) % m
            if su < sv:
                edges.append((su, sv))
            else:
                edges.append((sv, su))
                
    return sorted(set(edges))

def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 5:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    else:
        return 0, []
        
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
                
    n = len(order)
    edges_list = sorted(set(edges))
    return n, edges_list

def _verify(n, edges, k, g):
    """Optimized verification using BFS from each vertex."""
    if n < 1 or len(edges) != len(set(edges)):
        return False
        
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n or a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
        
    for i in range(n):
        if len(adj[i]) != k:
            return False

    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            v = q.popleft()
            if dist[v] >= g - 1:
                continue
            for w in adj[v]:
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    q.append(w)
                elif dist[w] >= dist[v]:
                    if dist[v] + dist[w] + 1 < g:
                        return False
    return True
