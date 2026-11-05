import time
import sys
from collections import deque
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before the 150s kill

    # --- Fast Girth Check using NumPy for BFS ---
    # For a graph with n vertices and girth g, we need to ensure no cycle < g.
    # This is equivalent to checking that the ball of radius (g-1)//2 around any vertex is a tree.
    # We can simulate BFS levels using numpy arrays for speed.
    
    def check_girth_numpy(adj_matrix, n, g):
        # adj_matrix is a boolean numpy array of shape (n, n)
        # We check if there is a cycle of length < g.
        # A cycle of length L exists if there is a path of length L-1 that returns to start?
        # No, simpler:
        # If girth >= g, then for any edge (u,v), the distance between u and v in G-(u,v) is >= g-1.
        # This is hard to check directly with numpy.
        
        # Alternative: BFS from each node up to depth (g-1)//2.
        # If we find a "cross edge" (edge connecting two nodes in the BFS tree that are not parent-child)
        # within this depth, we have a cycle of length <= 2*depth + 1.
        # If 2*depth + 1 < g, we fail.
        
        max_depth = (g - 1) // 2
        
        # Precompute neighbors as lists of arrays for fast access?
        # Or just use the adjacency matrix.
        
        for start in range(n):
            # BFS
            dist = np.full(n, -1, dtype=np.int32)
            parent = np.full(n, -1, dtype=np.int32)
            dist[start] = 0
            
            # Queue
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                
                # Get neighbors of u
                # Using matrix multiplication or slicing
                neighbors = np.where(adj_matrix[u])[0]
                
                for v in neighbors:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # v is already visited
                        # Check if it's a cross edge
                        # If parent[u] != v and parent[v] != u, it's a cross edge
                        if parent[u] != v and parent[v] != u:
                            # Both u and v are within max_depth
                            # Cycle length is at most dist[u] + dist[v] + 1
                            # Actually, the cycle formed by the tree paths and the edge (u,v)
                            # has length dist[u] + dist[v] + 1 ONLY IF the LCA is start.
                            # But even if LCA is not start, the cycle length is <= dist[u] + dist[v] + 1.
                            # Since dist[u] <= max_depth and dist[v] <= max_depth:
                            # Max cycle length found this way is 2*max_depth + 1.
                            # If g is even, 2*(g-1)//2 + 1 = g-1 < g.
                            # If g is odd, 2*(g-1)//2 + 1 = g-2 < g.
                            # So any cross edge within this radius implies girth < g.
                            return False
                            
        return True

    # --- Algebraic Construction using Non-Abelian Groups ---
    # We use the Petersen graph as the base.
    # The Petersen graph has 10 vertices and 15 edges.
    # We lift it using a group G.
    # The lifted graph has 10*|G| vertices.
    # We want 10*|G| to be small.
    # The smallest non-abelian group is S3 (order 6) or A4 (order 12) or D5 (order 10).
    # Let's try A5 (order 60) -> n=600.
    # Let's try D5 (order 10) -> n=100.
    # Let's try S3 (order 6) -> n=60.
    
    # However, the girth of the lift depends on the voltage assignment.
    # For a non-abelian group, the voltage assignment is a map from edges to group elements.
    # The girth is related to the length of the shortest word in the generators (voltages)
    # that equals the identity, subject to the constraints of the base graph cycles.
    
    # The Petersen graph has cycles of length 5 (outer), 5 (inner), and 10 (spokes+outer/inner).
    # Actually, the girth of Petersen is 5.
    # If we assign voltages such that the product of voltages around any cycle of length 5 in the base
    # is not the identity (and has large order), the girth of the lift increases.
    
    # Let's define a specific group and voltage assignment.
    # We will use the group of rotations of the icosahedron/dodecahedron (A5).
    # But implementing A5 arithmetic in Python is slow.
    # Let's use a simpler non-abelian group: The Dicyclic group or Dihedral group.
    # Dihedral group D_n (order 2n).
    # Let's try D_5 (order 10). n_lift = 100.
    
    # D_5 elements: {0, 1, ..., 9}
    # r = rotation (order 5), s = reflection (order 2).
    # r^5 = 1, s^2 = 1, srs = r^-1.
    # We can represent elements as (i, j) where i in Z_5, j in {0, 1}.
    # (i, 0) is r^i, (i, 1) is r^i s.
    # Multiplication:
    # (i, 0) * (j, 0) = (i+j, 0)
    # (i, 0) * (j, 1) = (i+j, 1)
    # (i, 1) * (j, 0) = (i-j, 1)
    # (i, 1) * (j, 1) = (i-j, 0)
    
    def build_dihedral_lift(n_base, base_edges, n_dih):
        # n_dih is the order of the dihedral group D_m, where m = n_dih // 2.
        # We need m to be an integer.
        if n_dih % 2 != 0:
            return None
        
        m = n_dih // 2
        n = n_base * n_dih
        
        # Voltage assignment:
        # We need to assign an element of D_m to each edge.
        # To maximize girth, we should avoid short products equaling identity.
        # Let's use a deterministic assignment based on edge index.
        # Edge i gets voltage (i % m, 0) ?
        # Or (i % m, i % 2)?
        
        # Let's try a few deterministic patterns.
        patterns = [
            lambda i: (i % m, 0),
            lambda i: (i % m, 1),
            lambda i: (i % m, (i // m) % 2),
            lambda i: ((i * 2) % m, 0),
            lambda i: ((i * 3) % m, 1),
        ]
        
        best_adj = None
        best_n = n
        
        for p_idx, pattern in enumerate(patterns):
            voltages = [pattern(i) for i in range(len(base_edges))]
            
            adj = [[] for _ in range(n)]
            
            for idx, (u, v) in enumerate(base_edges):
                vol = voltages[idx]
                # vol is (i, j)
                i_vol, j_vol = vol
                for k in range(n_dih):
                    # k represents the group element.
                    # We need to map k to (i_k, j_k).
                    # Let's use k = i + j*m.
                    i_k = k % m
                    j_k = k // m
                    
                    # Compute k + vol
                    # (i_k, j_k) * (i_vol, j_vol)
                    if j_k == 0 and j_vol == 0:
                        i_res = (i_k + i_vol) % m
                        j_res = 0
                    elif j_k == 0 and j_vol == 1:
                        i_res = (i_k + i_vol) % m
                        j_res = 1
                    elif j_k == 1 and j_vol == 0:
                        i_res = (i_k - i_vol) % m
                        j_res = 1
                    else: # j_k == 1 and j_vol == 1
                        i_res = (i_k - i_vol) % m
                        j_res = 0
                    
                    k_res = i_res + j_res * m
                    
                    u_lift = u * n_dih + k
                    v_lift = v * n_dih + k_res
                    
                    adj[u_lift].append(v_lift)
                    adj[v_lift].append(u_lift)
            
            # Check for simple graph
            # Since base is simple, and group action is free, no multiple edges unless
            # two different base edges map to the same lift edge.
            # This can happen if voltages are "bad".
            # We'll assume it's simple for now and check girth.
            
            # Convert to numpy matrix for fast check
            adj_matrix = np.zeros((n, n), dtype=bool)
            for u in range(n):
                for v in adj[u]:
                    adj_matrix[u, v] = True
            
            if check_girth_numpy(adj_matrix, n, g):
                # Found a valid graph
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
                
        return None

    def get_petersen_edges():
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        return edges

    # Fallback: LUW construction
    def construct_luw(k, g):
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

    # Get baseline
    baseline_n, baseline_edges = construct_luw(k, g)
    best_n = baseline_n
    best_edges = baseline_edges

    if k == 3:
        base_edges = get_petersen_edges()
        n_base = 10
        
        # Try different dihedral groups
        # n_lift = 10 * (2 * m) = 20 * m
        # We want n_lift < best_n.
        # If best_n is 936, we can try m up to 46.
        
        # Try m from 1 to 50
        for m in range(1, 51):
            if time.time() > deadline - 10.0:
                break
            
            n_dih = 2 * m
            result = build_dihedral_lift(n_base, base_edges, n_dih)
            if result:
                n, edges = result
                if n < best_n:
                    best_n = n
                    best_edges = edges
                    # Since n increases with m, if we find one, it's likely the smallest m that works
                    # But we should continue to check if a smaller m was missed?
                    # We are iterating m upwards. So the first success is the smallest m.
                    # We can break.
                    break

    return best_n, best_edges
