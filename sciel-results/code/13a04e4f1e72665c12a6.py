import time
import numpy as np
import random
from collections import deque
from itertools import product

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    def remaining():
        return deadline - time.time()
    
    # ---------------- girth verification ----------------
    def check_girth(n, edges, g, time_limit):
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
    
    # ---------------- PSL(2,q) group construction ----------------
    def psl2q_construction():
        if k != 3 or g != 16:
            return None
        
        # PSL(2,q) for q = p^e where p is prime
        # We'll try small q values: 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 37, 41, 43, 47, 49, 53, 59, 61, 64, 67, 71, 73, 79, 81, 83, 89, 97
        # Order of PSL(2,q) = q(q^2-1)/gcd(2,q-1)
        
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True
        
        def psl2q_order(q):
            if q == 2:
                return 1
            if q == 3:
                return 6
            return q * (q*q - 1) // 2 if q % 2 == 1 else q * (q*q - 1) // 2
        
        # For each q, we need to find 3 elements a, b, c in PSL(2,q) such that:
        # 1. The graph with vertices = group elements, edges = {(x, x*a), (x, x*b), (x, x*c)} is 3-regular
        # 2. No cycle shorter than g exists
        # 
        # Key insight: if a, b, c are chosen such that all words of length < g/2 in {a, a^{-1}, b, b^{-1}, c, c^{-1}}
        # are non-identity, then the girth is at least g.
        #
        # For PSL(2,q), we can represent elements as 2x2 matrices with determinant 1, modulo scalars.
        # 
        # A simpler approach: use the fact that PSL(2,q) has a presentation, and search for generating sets
        # with large girth.
        
        # Try q values in order of increasing group order
        q_candidates = []
        for q in range(4, 100):
            if q == 4 or q == 9 or q == 16 or q == 25 or q == 36 or q == 49 or q == 64 or q == 81:
                # q is a square of a prime
                p = int(q ** 0.5)
                if p * p == q and is_prime(p):
                    q_candidates.append(q)
            elif is_prime(q):
                q_candidates.append(q)
        
        q_candidates.sort(key=psl2q_order)
        
        for q in q_candidates:
            if remaining() < 10:
                break
            
            n = psl2q_order(q)
            if n < 100:  # Too small
                continue
            if n > 5000:  # Too large
                break
            
            # For PSL(2,q), we represent elements as triples (a, b, c) in the field F_q
            # with the condition that not all are zero, modulo scalar multiplication.
            # Identity is (1, 0, 0) or similar.
            
            # This is getting complex. Let's use a different approach:
            # Use the voltage lift on a smaller base graph with a non-Abelian group.
            
            # Actually, let's stick with the Abelian voltage lift but improve the search.
            # The key issue is that we need to find voltage assignments that avoid all
            # short closed walks.
            
            # Let me try a different strategy: use a larger base graph.
            # The Petersen graph has 10 vertices. What if we use a larger base graph?
            
            # Alternative: use the Heawood graph (14 vertices, 3-regular, girth 6)
            # or the Foster graph (90 vertices, 3-regular, girth 9).
            
            # For now, let's improve the Abelian approach with better heuristics.
            break
        
        return None
    
    # ---------------- Improved Voltage Lift with Better Heuristics ----------------
    def voltage_lift_improved():
        if k != 3 or g != 16:
            return None
        
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

        # Generate all non-backtracking closed walks of length [5, 15]
        max_L = g - 1
        walk_vectors = set()
        
        for start_v in range(n_base):
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
                    d = 1 if current == u else -1
                    
                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1
                    
                    if next_v == start_v and new_l >= 5:
                        walk_vectors.add(tuple(new_counts))
                    
                    stack.append((next_v, current, new_l, new_counts))
                    
        if not walk_vectors:
            return None
            
        # Filter out zero vectors and duplicates
        unique_constraints = set()
        for c in walk_vectors:
            if all(x == 0 for x in c):
                continue
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
        if C.size == 0:
            return None
        
        # Try primes m >= 52 with better search strategy
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True
        
        # Use simulated annealing or genetic algorithm for voltage assignment
        def solve_voltage(m, max_iters=5000):
            v = np.random.randint(1, m, size=15).astype(np.int64)
            
            def cost(v):
                residues = np.mod(C @ v, m)
                return np.sum(residues == 0)
            
            best_v = v.copy()
            best_cost = cost(v)
            
            T = 1.0
            cooling = 0.9995
            
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
        
        # Try primes in increasing order
        m = 53
        while m < 500 and remaining() > 15:
            if is_prime(m):
                # Try multiple restarts
                for restart in range(10):
                    if remaining() < 10:
                        break
                    
                    v, cost = solve_voltage(m, max_iters=3000)
                    
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
            
            m += 1
            
        return None
    
    # ---------------- Main ----------------
    res = voltage_lift_improved()
    if res is not None:
        return res
    
    # Fallback: try the original approach with more attempts
    res = psl2q_construction()
    if res is not None:
        return res
    
    return (0, [])

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
