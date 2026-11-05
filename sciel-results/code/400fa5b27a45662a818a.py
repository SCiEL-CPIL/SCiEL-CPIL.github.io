import time
import numpy as np
import random
from collections import deque

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

    # ---------------- Voltage Lift of Petersen ----------------
    def voltage_lift_petersen():
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
        # Vector representation: count of each edge with sign
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
            
        # Filter out zero vectors and duplicates (negatives)
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
            return None

        # Try primes m >= 52
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True

        min_m = 52
        m = min_m
        best_result = None
        
        while m < 1000 and remaining() > 10:
            if is_prime(m):
                # Local search for voltage assignment
                found = False
                # Try a few random initializations
                for _ in range(50):
                    if remaining() < 5: break
                    v = np.random.randint(1, m, size=15).astype(np.int64)
                    
                    # Local repair
                    for step in range(200):
                        if remaining() < 5: break
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
                        
                # If not found, try next prime
            m += 1
            
        return None

    # ---------------- Greedy Growth ----------------
    def greedy_growth():
        if k != 3:
            return None
            
        # This is a heuristic search for a 3-regular graph of girth g
        # We grow a tree and close leaves.
        
        def grow_tree():
            # Start with a path of length 2
            # Vertices: 0-1-2
            # Degrees: 0:1, 1:2, 2:1
            # We want to reach n vertices.
            # Target n is unknown, but we can stop when we have enough "leaves" to close.
            # Actually, we just grow until we have a connected graph where we can close it.
            
            # Simpler: Start with a single vertex.
            # This is too slow.
            
            # Alternative: Start with a cycle of length g? No.
            
            # Let's use a standard heuristic:
            # 1. Generate a random 3-regular graph with n vertices (n = 512 + 2*i).
            # 2. Check girth.
            # 3. If girth < g, try to fix? Hard.
            
            # Better: Start with a tree of depth d.
            # For g=16, d=7.
            # n = 1 + 3 + 9 + ... + 3^7 = 1093.
            # This is larger than 936.
            # So we need to "prune" or "merge" to reduce size.
            
            return None # Placeholder, too complex to implement robustly in short time

        return None

    # ---------------- Main ----------------
    res = voltage_lift_petersen()
    if res is not None:
        return res
        
    # Fallback: Return a known valid graph if we have one hardcoded?
    # We don't have one.
    # Try a random 3-regular graph with large n?
    # n = 1000.
    # Random 3-regular graphs typically have small girth.
    # So this will likely fail.
    
    # If all else fails, return a trivially valid but large graph?
    # We can construct a cycle of length n (girth n) but it's 2-regular.
    # We need 3-regular.
    # Prism graph? Girth 4.
    
    # Let's try one more time with a larger m in voltage lift
    # The loop above went up to 1000.
    
    # If we fail, return (0, []) which scores 0.
    return (0, [])

if __name__ == "__main__":
    # Test
    n, edges = construct(3, 16)
    print(f"n={n}")
