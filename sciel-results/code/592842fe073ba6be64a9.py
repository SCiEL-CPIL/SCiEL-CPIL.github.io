import time
import numpy as np
import random
from math import gcd

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Strategy: Voltage lift of Heawood graph (girth 7) to reduce constraints.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    def remaining():
        return deadline - time.time()
        
    # ---------------- Girth verification ----------------
    def check_girth(n, edges, g, time_limit):
        """Verify no cycle shorter than g using non-backtracking BFS from every vertex."""
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        min_cycle = float('inf')
        
        for s in range(n):
            if time.time() > time_limit:
                return -1
            
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            
            if min_cycle < g:
                return min_cycle
        
        return min_cycle

    # ---------------- Voltage lift of Heawood graph ----------------
    def voltage_lift_heawood():
        if k != 3 or g != 16:
            return None
        
        # Heawood graph edges (14 vertices, 21 edges)
        # Fano plane incidence graph
        # Vertices 0..6 (points), 7..13 (lines)
        heawood_edges = [
            (0, 7), (0, 8), (0, 9),
            (1, 7), (1, 10), (1, 11),
            (2, 8), (2, 10), (2, 12),
            (3, 9), (3, 11), (3, 12),
            (4, 7), (4, 11), (4, 13),
            (5, 8), (5, 12), (5, 13),
            (6, 9), (6, 10), (6, 13)
        ]
        
        n_base = 14
        edge_index = {}
        for i, (u, v) in enumerate(heawood_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i
            
        adj = [[] for _ in range(n_base)]
        for u, v in heawood_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Generate all non-backtracking closed walks of length < g
        max_L = g - 1
        walk_vectors = set()
        
        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0]*21)]
            while stack:
                current, prev, length, counts = stack.pop()
                
                if length >= max_L:
                    continue
                
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = heawood_edges[e_idx]
                    d = 1 if current == u else -1
                    
                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1
                    
                    if next_v == start_v and new_l >= 7: # Min girth of Heawood is 7
                        walk_vectors.add(tuple(new_counts))
                    
                    stack.append((next_v, current, new_l, new_counts))
        
        if not walk_vectors:
            return None
            
        # Filter out zero vectors and normalize
        unique_constraints = set()
        for c in walk_vectors:
            if all(x == 0 for x in c):
                continue
            first_non_zero = -1
            for i in range(21):
                if c[i] != 0:
                    first_non_zero = i
                    break
            if first_non_zero != -1:
                if c[first_non_zero] < 0:
                    c = tuple(-x for x in c)
                unique_constraints.add(c)
                
        C = np.array(list(unique_constraints), dtype=np.int64)
        if C.size == 0:
            return None
        
        # Precompute non-zero entries
        walk_coeffs = []
        for row in C:
            nonzero = [(i, c) for i, c in enumerate(row) if c != 0]
            walk_coeffs.append(nonzero)
        
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True
        
        def count_violations(volts, m):
            count = 0
            for nonzero in walk_coeffs:
                s = 0
                for idx, c in nonzero:
                    s += c * volts[idx]
                if s % m == 0:
                    count += 1
            return count

        def solve_voltage(m, max_iters=10000):
            """Local search to find valid voltage assignment."""
            v = np.random.randint(1, m, size=21).astype(np.int64)
            best_v = v.copy()
            best_cost = count_violations(v, m)
            
            T = 1.0
            cooling = 0.999
            
            for it in range(max_iters):
                if remaining() < 5:
                    break
                
                j = random.randint(0, 20)
                old_val = v[j]
                v[j] = random.randint(1, m-1)
                
                new_cost = count_violations(v, m)
                delta = new_cost - best_cost
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                    if new_cost == 0:
                        return v, 0
                else:
                    v[j] = old_val
                
                T *= cooling
            
            return best_v, best_cost

        # Try primes m such that 14*m < 936
        # m < 66.8
        primes = [p for p in range(61, 2, -1) if is_prime(p)]
        
        for m in primes:
            if remaining() < 10:
                break
            
            # Multiple restarts
            for restart in range(50):
                if remaining() < 5:
                    break
                
                v, cost = solve_voltage(m, max_iters=5000)
                
                if cost == 0:
                    # Construct graph
                    n = 14 * m
                    edges = []
                    for u, v_base in heawood_edges:
                        e_idx = edge_index[(u, v_base)]
                        vol = v[e_idx]
                        for layer in range(m):
                            u_lift = u * m + layer
                            v_lift = v_base * m + (layer + vol) % m
                            if u_lift < v_lift:
                                edges.append((u_lift, v_lift))
                            else:
                                edges.append((v_lift, u_lift))
                    
                    # Verify
                    gc = check_girth(n, edges, g, deadline - 2)
                    if gc >= g:
                        return (n, edges)
        
        return None
    
    # ---------------- Fallback: Petersen Lift ----------------
    def voltage_lift_petersen():
        if k != 3 or g != 16:
            return None
        # ... (Keep the existing Petersen logic as fallback)
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        n_base = 10
        edge_index = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i
        adj = [[] for _ in range(n_base)]
        for u, v in pet_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_L = g - 1
        walk_vectors = set()
        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0]*15)]
            while stack:
                current, prev, length, counts = stack.pop()
                if length >= max_L: continue
                for next_v in adj[current]:
                    if next_v == prev: continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    d = 1 if current == u else -1
                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1
                    if next_v == start_v and new_l >= 3:
                        walk_vectors.add(tuple(new_counts))
                    stack.append((next_v, current, new_l, new_counts))
        
        if not walk_vectors: return None
        unique_constraints = set()
        for c in walk_vectors:
            if all(x == 0 for x in c): continue
            first_non_zero = -1
            for i in range(15):
                if c[i] != 0:
                    first_non_zero = i
                    break
            if first_non_zero != -1:
                if c[first_non_zero] < 0:
                    c = tuple(-x for x in c)
                unique_constraints.add(c)
                
        C = np.array(list(unique_constraints), dtype=np.int64)
        if C.size == 0: return None
        walk_coeffs = []
        for row in C:
            nonzero = [(i, c) for i, c in enumerate(row) if c != 0]
            walk_coeffs.append(nonzero)
            
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True
            
        def count_violations(volts, m):
            count = 0
            for nonzero in walk_coeffs:
                s = 0
                for idx, c in nonzero:
                    s += c * volts[idx]
                if s % m == 0:
                    count += 1
            return count

        def solve_voltage(m, max_iters=5000):
            v = np.random.randint(1, m, size=15).astype(np.int64)
            best_v = v.copy()
            best_cost = count_violations(v, m)
            T = 1.0
            cooling = 0.9995
            for it in range(max_iters):
                if remaining() < 5: break
                j = random.randint(0, 14)
                old_val = v[j]
                v[j] = random.randint(1, m-1)
                new_cost = count_violations(v, m)
                delta = new_cost - best_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                    if new_cost == 0:
                        return v, 0
                else:
                    v[j] = old_val
                T *= cooling
            return best_v, best_cost

        m = 53
        while m < 500 and remaining() > 15:
            if is_prime(m):
                for restart in range(20):
                    if remaining() < 10: break
                    v, cost = solve_voltage(m, max_iters=3000)
                    if cost == 0:
                        n = 10 * m
                        edges = []
                        for u, v_base in pet_edges:
                            e_idx = edge_index[(u, v_base)]
                            vol = v[e_idx]
                            for layer in range(m):
                                u_lift = u * m + layer
                                v_lift = v_base * m + (layer + vol) % m
                                if u_lift < v_lift:
                                    edges.append((u_lift, v_lift))
                                else:
                                    edges.append((v_lift, u_lift))
                        gc = check_girth(n, edges, g, deadline - 2)
                        if gc >= g:
                            return (n, edges)
            m += 1
        return None

    # ---------------- Main ----------------
    # Try Heawood first (fewer constraints, potentially smaller n)
    res = voltage_lift_heawood()
    if res is not None:
        return res
        
    # Fallback to Petersen
    res = voltage_lift_petersen()
    if res is not None:
        return res
        
    # Fallback to LUW (from original)
    def luw_fallback(k, g):
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

    return luw_fallback(k, g)

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
