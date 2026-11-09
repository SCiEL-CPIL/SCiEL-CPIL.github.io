"""Stochastic local search (Simulated Annealing) for k-regular graphs of large girth."""
import random
import math
import time
from collections import deque
import numpy as np

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Uses Simulated Annealing with 2-switches to maximize girth.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j).
    """
    # Moore bound: minimum possible vertices
    if g % 2 == 0:
        moore = 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        moore = 1 + k * sum((k - 1) ** i for i in range((g - 3) // 2 + 1))
    
    n = moore
    max_n = moore + 100
    
    def generate_k_regular(n, k):
        """Generate a random simple k-regular graph."""
        if n * k % 2 != 0:
            return None
        stubs = [i for i in range(n) for _ in range(k)]
        for _ in range(100):
            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                edge = (min(u, v), max(u, v))
                if edge in edges:
                    valid = False
                    break
                edges.add(edge)
            if valid:
                return list(edges)
        return None

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def has_girth_at_least(n, adj, g):
        """Check if girth is at least g."""
        for start in range(n):
            visited = set()
            queue = deque()
            for neighbor in adj[start]:
                queue.append((neighbor, start, 1))
                visited.add((neighbor, start))
            
            while queue:
                u, prev, dist = queue.popleft()
                if dist >= g - 1:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == start:
                        if dist + 1 < g:
                            return False
                    else:
                        state = (v, u)
                        if state not in visited:
                            visited.add(state)
                            queue.append((v, u, dist + 1))
        return True

    def count_3_4_cycles(n, adj):
        """Fast approximation of short cycles."""
        count = 0
        # 3-cycles
        for u in range(n):
            for i in range(len(adj[u])):
                for j in range(i+1, len(adj[u])):
                    if adj[u][i] in adj[adj[u][j]]:
                        count += 1
        # 4-cycles
        # For each pair of vertices, count common neighbors
        # This is O(n^2 * k)
        for u in range(n):
            for v in range(u+1, n):
                # Count common neighbors
                common = 0
                for x in adj[u]:
                    if x in adj[v]:
                        common += 1
                count += common
        return count

    time_limit = 110
    start_time = time.time()
    best_n = None
    best_edges = None
    
    current_n = n
    while current_n <= max_n and time.time() - start_time < time_limit:
        edges = generate_k_regular(current_n, k)
        if edges is None:
            current_n += 1
            continue
        
        adj = build_adj(current_n, edges)
        
        if has_girth_at_least(current_n, adj, g):
            best_n = current_n
            best_edges = edges
            break
        
        T = 100.0
        T_min = 0.01
        cooling = 0.999
        current_energy = count_3_4_cycles(current_n, adj)
        best_energy = current_energy
        best_edges_sa = edges[:]
        best_adj_sa = adj
        iter_count = 0
        
        while time.time() - start_time < time_limit:
            if len(edges) < 2:
                break
            i = random.randint(0, len(edges) - 1)
            j = random.randint(0, len(edges) - 1)
            if i == j:
                continue
            
            a, b = edges[i]
            c, d = edges[j]
            
            if len({a, b, c, d}) != 4:
                continue
            
            edge_set = set((min(u,v), max(u,v)) for u,v in edges)
            
            new_edges_candidate = None
            ne1 = (min(a,c), max(a,c))
            ne2 = (min(b,d), max(b,d))
            if ne1 not in edge_set and ne2 not in edge_set and ne1 != ne2:
                new_edges_candidate = [ne1, ne2]
            else:
                ne1 = (min(a,d), max(a,d))
                ne2 = (min(b,c), max(b,c))
                if ne1 not in edge_set and ne2 not in edge_set and ne1 != ne2:
                    new_edges_candidate = [ne1, ne2]
            
            if not new_edges_candidate:
                continue
                
            new_edges = edges[:]
            new_edges[i] = new_edges_candidate[0]
            new_edges[j] = new_edges_candidate[1]
            
            new_adj = build_adj(current_n, new_edges)
            new_energy = count_3_4_cycles(current_n, new_adj)
            
            if new_energy < current_energy:
                current_energy = new_energy
                edges = new_edges
                adj = new_adj
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_edges_sa = edges[:]
                    best_adj_sa = adj
            else:
                delta = new_energy - current_energy
                if random.random() < math.exp(-delta / T):
                    current_energy = new_energy
                    edges = new_edges
                    adj = new_adj
            
            T *= cooling
            if T < T_min:
                T = T_min
            
            iter_count += 1
            if iter_count % 100 == 0 and best_energy == 0:
                if has_girth_at_least(current_n, best_adj_sa, g):
                    best_n = current_n
                    best_edges = best_edges_sa
                    break
        
        if best_edges is not None:
            break
        current_n += 1
        
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback
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
    index, order, fallback_edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                fallback_edges.append((index[vertex], index[other]))
    
    return len(order), fallback_edges
