import time
import numpy as np
import random
from math import gcd

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Strategy: Voltage lift of Petersen graph with optimized constraint solving.
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
    
    # ---------------- Voltage lift of Petersen graph ----------------
    def voltage_lift_petersen():
        if k != 3 or g != 16:
            return None
        
        # Petersen graph edges (10 vertices, 15 edges)
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer cycle
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner star
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
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
        
        # Generate all non-backtracking closed walks of length < g
        # We represent a walk by the net count of each edge traversed.
        # A closed walk corresponds to a vector c in Z^15 such that sum(c_i * e_i) = 0 in the cycle space?
        # No, the voltage condition is sum(voltage(e_i) * direction) != 0 mod m.
        # The "direction" is encoded in the sign of the count in the vector.
        
        max_L = g - 1
        walk_vectors = set()
        
        # Efficient generation using BFS on states (current_node, prev_node, length)
        # State: (node, prev, length, counts_tuple)
        # To avoid explosion, we prune if length >= max_L
        # We only care about closed walks (node == start, prev is not start? No, just node == start)
        
        for start_v in range(n_base):
            # Stack: (current, prev, length, counts_list)
            stack = [(start_v, -1, 0, [0]*15)]
            while stack:
                current, prev, length, counts = stack.pop()
                
                if length >= max_L:
                    continue
                
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    # Determine direction: +1 if current==u, -1 if current==v
                    d = 1 if current == u else -1
                    
                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1
                    
                    if next_v == start_v and new_l >= 3:
                        # Found a closed walk
                        # Normalize the vector to handle symmetry (c and -c are equivalent constraints)
                        # We want to add the constraint: sum(c_i * v_i) != 0
                        # If c is a solution, -c is also a solution.
                        # We normalize by making the first non-zero element positive.
                        vec = tuple(new_counts)
                        if any(x != 0 for x in vec):
                            first_nz = -1
                            for i in range(15):
                                if vec[i] != 0:
                                    first_nz = i
                                    break
                            if first_nz != -1:
                                if vec[first_nz] < 0:
                                    vec = tuple(-x for x in vec)
                                walk_vectors.add(vec)
                    
                    stack.append((next_v, current, new_l, new_counts))
        
        if not walk_vectors:
            return None
            
        unique_constraints = list(walk_vectors)
        C = np.array(unique_constraints, dtype=np.int64)
        
        if C.size == 0:
            return None
        
        # Precompute non-zero entries for fast checking
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
        
        def check_valid(volts, m):
            """Check if voltage assignment avoids all short cycles."""
            for nonzero in walk_coeffs:
                s = 0
                for idx, c in nonzero:
                    s += c * volts[idx]
                if s % m == 0:
                    return False
            return True
        
        def solve_voltage(m, max_iters=10000):
            """Simulated annealing to find valid voltage assignment."""
            v = np.random.randint(1, m, size=15).astype(np.int64)
            
            def cost(v):
                # Vectorized cost calculation
                # C is (N, 15), v is (15,)
                # prod = C @ v
                # check prod % m == 0
                prod = C @ v
                residues = np.mod(prod, m)
                return np.sum(residues == 0)
            
            best_v = v.copy()
            best_cost = cost(v)
            
            T = 1.0
            cooling = 0.9998
            
            for it in range(max_iters):
                if remaining() < 5:
                    break
                
                # Pick a random variable to change
                j = random.randint(0, 14)
                old_val = v[j]
                v[j] = random.randint(1, m-1)
                
                new_cost = cost(v)
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
        
        # Try primes m starting from a reasonable lower bound
        # Moore bound suggests n >= 1020, so m >= 102.
        # But known graphs are smaller? 
        # Let's try a range of primes.
        # The current best known is 936, which is 10 * 93.6. So m=97 or 89.
        # Let's try primes from 53 upwards.
        
        primes_to_try = []
        m = 53
        while m < 200:
            if is_prime(m):
                primes_to_try.append(m)
            m += 1
        
        # Sort by distance to 93.6? Or just ascending?
        # Ascending is safer to find *any* solution, but we want *small* n.
        # So we want the *smallest* m that works.
        
        for m in primes_to_try:
            if remaining() < 10:
                break
                
            # Try multiple restarts for this m
            for restart in range(50):
                if remaining() < 5:
                    break
                    
                v, cost = solve_voltage(m, max_iters=2000)
                
                if cost == 0:
                    # Construct graph
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
                    
                    # Verify
                    gc = check_girth(n, edges, g, deadline - 2)
                    if gc >= g:
                        return (n, edges)
        
        return None
    
    # ---------------- LUW fallback (for other k values) ----------------
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
    
    # ---------------- Main ----------------
    res = voltage_lift_petersen()
    if res is not None:
        return res
    
    # Fallback
    return luw_fallback(k, g)

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
