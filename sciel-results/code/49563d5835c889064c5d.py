import time
import random
import sys
import numpy as np

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
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

    if k == 4 and g == 10:
        # Try to construct a voltage graph lift of the Petersen graph
        # with group Z_32 to get n=320.
        
        # Petersen graph vertices 0..9
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        
        pet_adj = [[] for _ in range(10)]
        edge_to_idx = {}
        for i, (u, v) in enumerate(pet_edges):
            pet_adj[u].append((v, i))
            pet_adj[v].append((u, i))
            edge_to_idx[(u, v)] = i
            edge_to_idx[(v, u)] = i
            
        # Generate non-backtracking closed walks of length 5 to 9
        closed_walks = [] # List of tuples (edge_idx, sign)
        
        def gen_nb_walks(start, curr, prev, length, L, path):
            if length == L:
                if curr == start:
                    closed_walks.append(tuple(path))
                return
            for v, e_idx in pet_adj[curr]:
                if v == prev:
                    continue
                u_i, v_i = pet_edges[e_idx]
                if curr == u_i:
                    sign = 1
                else:
                    sign = -1
                gen_nb_walks(start, v, curr, length + 1, L, path + [(e_idx, sign)])
                
        for L in range(5, 10):
            for s in range(10):
                for v, e_idx in pet_adj[s]:
                    u_i, v_i = pet_edges[e_idx]
                    if s == u_i:
                        sign = 1
                    else:
                        sign = -1
                    gen_nb_walks(s, v, s, 1, L, [(e_idx, sign)])

        # Use a fast local search (Simulated Annealing) to find valid voltages
        n_edges = 15
        found = False
        best_voltages = None
        deadline = time.time() + 100
        
        # Precompute constraint coefficients for speed
        # constraints: list of lists of (edge_idx, coefficient)
        constraints = []
        for walk in closed_walks:
            coeff = [0] * n_edges
            for e_idx, sign in walk:
                coeff[e_idx] += sign
            constraints.append(coeff)
            
        def check_voltages(voltages):
            for coeff in constraints:
                total = 0
                for i in range(n_edges):
                    if coeff[i]:
                        total += coeff[i] * voltages[i]
                if total % 32 == 0:
                    return False
            return True
            
        def local_search():
            random.seed(42)
            # Initialize with random voltages
            voltages = [random.randint(0, 31) for _ in range(n_edges)]
            # Symmetry breaking: fix v0 = 1
            voltages[0] = 1
            
            current_energy = 0
            # Calculate initial energy (number of violated constraints)
            for coeff in constraints:
                total = sum(coeff[i] * voltages[i] for i in range(n_edges))
                if total % 32 == 0:
                    current_energy += 1
            
            if current_energy == 0:
                return voltages[:]
                
            temp = 10.0
            cooling = 0.95
            best_voltages_local = voltages[:]
            best_energy = current_energy
            
            while time.time() < deadline:
                if current_energy == 0:
                    return voltages[:]
                    
                # Pick a random edge to change
                i = random.randint(1, n_edges - 1) # Keep v0 fixed
                old_val = voltages[i]
                new_val = (old_val + random.randint(1, 31)) % 32
                
                # Calculate delta energy
                # Only constraints involving edge i can change
                delta = 0
                for coeff in constraints:
                    if coeff[i] == 0:
                        continue
                    old_total = sum(coeff[j] * voltages[j] for j in range(n_edges))
                    new_total = old_total + coeff[i] * (new_val - old_val)
                    old_violated = 1 if old_total % 32 == 0 else 0
                    new_violated = 1 if new_total % 32 == 0 else 0
                    delta += new_violated - old_violated
                    
                voltages[i] = new_val
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_voltages_local = voltages[:]
                    
                # Accept or reject
                if delta < 0:
                    pass # Always accept
                else:
                    if random.random() < np.exp(-delta / temp):
                        pass # Accept
                    else:
                        # Reject
                        voltages[i] = old_val
                        current_energy -= delta
                        
                temp *= cooling
                if temp < 0.1:
                    break
                    
            return best_voltages_local if best_energy == 0 else None

        # Run local search
        best_voltages = local_search()
        if best_voltages:
            found = True

        if found:
            # Construct the lift
            n = 10 * 32
            edges = []
            for u, v in pet_edges:
                idx = edge_to_idx[(u, v)]
                w = best_voltages[idx]
                for g_val in range(32):
                    u_idx = u * 32 + g_val
                    v_idx = v * 32 + (g_val + w) % 32
                    edges.append((u_idx, v_idx))
                    
            # Verify girth
            def has_girth_at_least(n, edges, g):
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                edge_set = set()
                for u, v in edges:
                    if u == v:
                        return False
                    if (v, u) in edge_set:
                        return False
                    edge_set.add((u, v))
                
                for start in range(n):
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[start] = 0
                    queue = [start]
                    head = 0
                    while head < len(queue):
                        u = queue[head]
                        head += 1
                        if dist[u] >= g // 2:
                            continue
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                queue.append(v)
                            else:
                                if v != parent[u] and u != parent[v]:
                                    cycle_len = dist[u] + dist[v] + 1
                                    if cycle_len < g:
                                        return False
                return True
                
            if has_girth_at_least(n, edges, g):
                return n, edges

    # Fallback to LUW
    return luw_construct(k, g)
