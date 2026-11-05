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
        
        # For each vertex, do a non-backtracking BFS to find short cycles
        for s in range(n):
            if time.time() > time_limit:
                return -1
                
            # dist[node] = depth from s
            # parent[node] = previous node in the non-backtracking path
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [(s, -1)]
            
            while queue:
                node, prev = queue.pop(0)
                d = dist[node]
                
                # If we have already found a cycle shorter than g, we can stop early?
                # Actually, we update min_cycle as we go.
                if d >= g // 2: 
                    # We only need to check up to depth g//2 to find cycles of length < g?
                    # A cycle of length L is found when a back-edge connects depth d1 and d2 with d1+d2+1 = L.
                    # If L < g, then max(d1, d2) < g.
                    # Actually, if we stop at depth g//2, we might miss cycles where one side is long?
                    # No, in a tree, the path from root to node is unique.
                    # If a cycle exists, it closes at some point.
                    # Standard BFS finds the shortest cycle through s.
                    # We can stop expanding if d >= g, because any cycle formed would be length >= 2*d+1 > g.
                    # But we need to check edges from nodes at depth d to nodes at depth d' where d+d'+1 < g.
                    # So we must expand up to depth g-1?
                    # Actually, if d >= g-1, any new edge to a visited node creates a cycle of length >= 2*(g-1)+1? No.
                    # Let's just expand until queue is empty or d >= g.
                    # But for efficiency, if d >= g, we can skip adding neighbors?
                    # If d >= g, then any cycle closing here has length >= d + 1 + 0 = g+1?
                    # Wait, if we connect to parent, it's length 2.
                    # We are doing non-backtracking, so we don't go to parent.
                    # So if d >= g, any cycle closing has length > g.
                    # So we can stop expanding at depth g.
                    if d >= g:
                        continue

                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if dist[nb] != -1:
                        # Found a cycle
                        cycle_len = d + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < g:
                            return min_cycle
                    else:
                        dist[nb] = d + 1
                        parent[nb] = node
                        queue.append((nb, node))
                        
        return min_cycle

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
                for _ in range(100):
                    if remaining() < 5: break
                    v = np.random.randint(1, m, size=15).astype(np.int64)
                    
                    # Local repair
                    for step in range(500):
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

    res = voltage_lift_petersen()
    if res is not None:
        return res
        
    return (0, [])

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
