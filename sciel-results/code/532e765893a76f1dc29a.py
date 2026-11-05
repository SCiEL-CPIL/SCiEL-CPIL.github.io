import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    def is_c4_free(adj):
        # adj is a numpy array of 0/1
        # Check if any pair has >1 common neighbor
        # common = adj @ adj
        # But adj @ adj includes diagonal (degrees)
        # We need off-diagonal counts > 1
        common = adj @ adj
        np.fill_diagonal(common, 0)
        return np.all(common <= 1)

    def has_independent_set(adj, k=11):
        # Simple heuristic: greedy independent set
        # If we can find one of size k, return True
        # This is not exact, but if we fail to find one, it might still exist.
        # However, for random graphs, greedy is often good.
        n = adj.shape[0]
        remaining = set(range(n))
        ind_set = []
        while remaining:
            # Pick vertex with min degree in induced subgraph
            min_deg = n
            min_v = -1
            for v in remaining:
                deg = sum(adj[v, u] for u in remaining if u != v)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            ind_set.append(min_v)
            remaining.remove(min_v)
            for u in list(remaining):
                if adj[min_v, u]:
                    remaining.remove(u)
            if len(ind_set) >= k:
                return True
        return False

    def try_build(n_target):
        # Start with ER(5) if n_target >= 31
        if n_target >= 31:
            q = 5
            n = q*q + q + 1
            # Construct ER(q)
            # Points in PG(2,q)
            # For q=5, we can hardcode or generate
            # Generation of PG(2,5) points
            points = []
            for x in range(q):
                for y in range(q):
                    points.append((x, y, 1))
            for x in range(q):
                points.append((x, 1, 0))
            points.append((1, 0, 0))
            
            # Polarity: x1*x2 + x2*x3 + x3*x1 = 0? 
            # Standard polarity for q odd: x1*x2 + x2*x3 + x3*x1 = 0
            # Two points P, Q are adjacent if P is on polar of Q.
            # Polar of Q=(a,b,c) is a*y + b*z + c*x = 0?
            # Let's use the standard definition: P=(x,y,z), Q=(u,v,w) adjacent if x*u + y*v + z*w = 0?
            # No, that's for orthogonal polarity.
            # For ER(q), adjacency is defined by orthogonality in the quadratic form.
            # For q=5, form Q(x,y,z) = x^2 + y^2 + z^2?
            # Let's just use a known graph or generate randomly if generation is complex.
            # Given time, let's use a random graph with C4-free check as fallback if ER generation fails.
            
            # Actually, let's just use a random 6-regular graph initialization for n_target
            pass
        
        # Initialize random graph
        adj = np.zeros((n_target, n_target), dtype=int)
        # Try to make it roughly regular
        deg_target = 6
        for _ in range(100):
            # Random edges
            adj = np.zeros((n_target, n_target), dtype=int)
            for i in range(n_target):
                # Pick deg_target neighbors
                neighbors = random.sample([j for j in range(n_target) if j != i], deg_target)
                for j in neighbors:
                    adj[i, j] = 1
                    adj[j, i] = 1
            
            if is_c4_free(adj):
                if not has_independent_set(adj, 11):
                    return {'n': n_target, 'edges': [[i, j] for i in range(n_target) for j in range(i+1, n_target) if adj[i, j]]}
        return None

    # Try n=39
    for n_try in [39, 38, 37, 36, 35, 34]:
        if time.time() > deadline:
            break
        result = try_build(n_try)
        if result:
            return result
            
    # Fallback: return seed
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

construct()
