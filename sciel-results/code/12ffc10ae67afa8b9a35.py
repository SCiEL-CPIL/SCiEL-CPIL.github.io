import time
import numpy as np
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

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
            dist[s] = 0
            queue = [(s, -1)]
            
            while queue:
                node, prev = queue.pop(0)
                d = dist[node]
                
                if d >= g:
                    continue

                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if dist[nb] != -1:
                        cycle_len = d + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < g:
                            return min_cycle
                    else:
                        dist[nb] = d + 1
                        queue.append((nb, node))
                        
        return min_cycle

    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0: return False
            i += 6
        return True

    # Petersen Graph Definition
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

    # Generate all non-backtracking closed walks of length [3, g-1]
    # We need to ensure no cycle of length < g.
    # A cycle of length L is found if a closed walk of length L has sum 0.
    # We collect all closed walks of length < g.
    
    walk_vectors = []
    max_L = g - 1
    
    for start_v in range(n_base):
        # stack: (current_node, prev_node, length, vector)
        # We use a list of lists for vectors to allow in-place modification? 
        # No, we need to track the path to backtrack.
        # We'll use a list of (node, prev, len, vec_tuple)
        
        # To optimize, we can use a dict for state (node, prev) -> list of (len, vec)?
        # No, simple DFS with backtracking is fine for small L.
        
        def dfs(current, prev, length, vec):
            if length >= max_L:
                return
            
            for next_v in adj[current]:
                if next_v == prev:
                    continue
                e_idx = edge_index[(current, next_v)]
                u, v = pet_edges[e_idx]
                d = 1 if current == u else -1
                
                new_vec = vec.copy()
                new_vec[e_idx] += d
                new_l = length + 1
                
                if next_v == start_v and new_l >= 3:
                    # Check if this vector is non-zero
                    if any(x != 0 for x in new_vec):
                        walk_vectors.append(tuple(new_vec))
                
                dfs(next_v, current, new_l, new_vec)
        
        dfs(start_v, -1, 0, [0]*15)

    if not walk_vectors:
        return (0, [])

    # Filter duplicates and zero vectors
    unique_constraints = set()
    for c in walk_vectors:
        if all(x == 0 for x in c):
            continue
        # Normalize: make first non-zero element positive
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
        return (0, [])

    # Search for voltage assignment
    min_m = 40
    m = min_m
    best_result = None
    
    while m < 1000 and remaining() > 5:
        if is_prime(m):
            # Local search for voltage assignment
            found = False
            # Try a few random initializations
            for _ in range(50):
                if remaining() < 2: break
                v = np.random.randint(1, m, size=15).astype(np.int64)
                
                # Local repair
                for step in range(1000):
                    if remaining() < 2: break
                    residues = np.mod(C @ v, m)
                    violated = np.where(residues == 0)[0]
                    if len(violated) == 0:
                        found = True
                        break
                        
                    # Pick random violated constraint
                    idx = violated[random.randint(0, len(violated)-1)]
                    c_row = C[idx]
                    non_zero = np.where(c_row != 0)[0]
                    if len(non_zero) == 0: continue
                    j = non_zero[random.randint(0, len(non_zero)-1)]
                    v[j] = random.randint(1, m-1)
                
                if found:
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
        
    return (0, [])

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
