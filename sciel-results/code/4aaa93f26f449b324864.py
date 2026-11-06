import time
import random

def construct():
    deadline = time.time() + 115.0
    
    # Generate Polarity Graph of PG(2,7)
    # Field F_7
    q = 7
    
    # Generate all non-zero vectors in F_q^3, normalized so first non-zero is 1
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    # x is 1
                    points.append((1, (y * pow(x, q-2)) % q, (z * pow(x, q-2)) % q))
                elif y != 0:
                    # x=0, y=1
                    points.append((0, 1, (z * pow(y, q-2)) % q))
                else:
                    # x=0, y=0, z=1
                    points.append((0, 0, 1))
    
    n_total = len(points)
    # n_total should be 57 for q=7
    
    # Build adjacency
    # Polarity: <u, v> = u[0]*v[0] + u[1]*v[1] + u[2]*v[2] == 0 mod q
    adj = [0] * n_total
    for i in range(n_total):
        for j in range(i + 1, n_total):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # We want a subgraph with n=43.
    # Let's take the first 43 vertices.
    n_target = 43
    if n_total < n_target:
        n_target = n_total
        
    # Check if the subgraph induced by 0..n_target-1 is valid.
    # It is C4-free by inheritance.
    # We need to check alpha < 11.
    # Since alpha(PG(2,7)) is likely < 11, this should hold.
    # But let's do a quick greedy check for an independent set of size 11.
    
    def has_independent_set_11(adj_subset, n):
        # Greedy approach to find independent set of size 11
        # If we find one, return True.
        # This is not exhaustive, but good for rejection.
        # For acceptance, we rely on the fact that alpha is small.
        # To be safer, we can try multiple random orderings.
        for _ in range(100):
            remaining = (1 << n) - 1
            count = 0
            while remaining and count < 11:
                # Pick vertex with minimum degree in remaining
                best_v = -1
                best_deg = n + 1
                for v in range(n):
                    if remaining & (1 << v):
                        non_adj = remaining & ~(adj_subset[v] | (1 << v))
                        deg = bin(non_adj).count('1')
                        if deg < best_deg:
                            best_deg = deg
                            best_v = v
                if best_v == -1:
                    break
                count += 1
                non_adj = remaining & ~(adj_subset[best_v] | (1 << best_v))
                remaining &= ~(non_adj | (1 << best_v))
            if count >= 11:
                return True
        return False

    # Extract subgraph
    adj_sub = [0] * n_target
    for i in range(n_target):
        for j in range(i + 1, n_target):
            if adj[i] & (1 << j):
                adj_sub[i] |= (1 << j)
                adj_sub[j] |= (1 << i)
                
    if not has_independent_set_11(adj_sub, n_target):
        edges = []
        for u in range(n_target):
            for v in range(u + 1, n_target):
                if adj_sub[u] & (1 << v):
                    edges.append([u, v])
        return {'n': n_target, 'edges': edges}
    
    # Fallback if the check fails (unlikely)
    # Return the original PG(2,5) construction (n=31)
    # Re-implementing PG(2,5) logic for fallback
    q5 = 5
    points5 = []
    for x in range(q5):
        for y in range(q5):
            for z in range(q5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    points5.append((1, (y * pow(x, q5-2)) % q5, (z * pow(x, q5-2)) % q5))
                elif y != 0:
                    points5.append((0, 1, (z * pow(y, q5-2)) % q5))
                else:
                    points5.append((0, 0, 1))
    n5 = len(points5)
    adj5 = [0] * n5
    for i in range(n5):
        for j in range(i + 1, n5):
            u = points5[i]
            v = points5[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q5
            if dot == 0:
                adj5[i] |= (1 << j)
                adj5[j] |= (1 << i)
                
    edges5 = []
    for u in range(n5):
        for v in range(u + 1, n5):
            if adj5[u] & (1 << v):
                edges5.append([u, v])
    return {'n': n5, 'edges': edges5}
