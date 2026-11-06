import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 105  # Safety margin for 120s limit
    
    # Fallback result
    best_result = {'n': 39, 'edges': []}
    
    # --- Helper to build Polarity Graph of PG(2, 7) ---
    def build_polarity_graph():
        q = 7
        # Points in PG(2, q) represented as tuples (x, y, z) in GF(q)^3, not all zero
        # Normalize: if x != 0, scale to x=1. Else if y != 0, scale to y=1. Else z=1.
        points = []
        for x in range(q):
            for y in range(q):
                points.append((1, x, y))
        for z in range(q):
            points.append((0, 1, z))
        points.append((0, 0, 1))
        
        n = len(points) # 57
        # Map point to index
        p_to_idx = {p: i for i, p in enumerate(points)}
        
        # Bilinear form B(u, v) = u1*v3 + u2*v2 + u3*v1 (Symmetric)
        # Wait, u1*v3 + u3*v1 = 2*u1*v3.
        # Let's use A = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        # B(u, v) = u1*v2 + u2*v1 + u2*v3 + u3*v2
        
        def b(u, v):
            return (u[0]*v[1] + u[1]*v[0] + u[1]*v[2] + u[2]*v[1]) % q
        
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i+1, n):
                if b(points[i], points[j]) == 0:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj, points

    adj_57, _ = build_polarity_graph()
    n_full = 57
    deg_full = adj_57.sum(axis=1)
    
    # --- Search for subset ---
    # Target n=46, min_deg=7
    # Target n=45, min_deg=6
    # Target n=44, min_deg=5
    
    targets = [
        (46, 7),
        (45, 6),
        (44, 5),
        (43, 4)
    ]
    
    for n_target, min_deg_req in targets:
        k_remove = n_full - n_target
        if k_remove < 0: continue
        
        # We need to remove k_remove vertices such that remaining min degree >= min_deg_req
        # Use local search / simulated annealing to find R
        best_R = None
        best_score = float('inf')
        
        # Initial random R
        R = set(random.sample(range(n_full), k_remove))
        
        # Calculate score: sum of deficits in min degree
        # Deficit for v in V' is max(0, min_deg_req - deg_V'(v))
        # We want score = 0
        
        def calc_score(R_set):
            V_prime = [v for v in range(n_full) if v not in R_set]
            if not V_prime: return float('inf')
            min_deg = min(deg_full[v] - np.sum(adj_57[v, list(R_set)]) for v in V_prime)
            # Actually we need min degree >= min_deg_req
            # Score is how far we are from satisfying it
            # Let's just return min_deg, and we want max min_deg
            return min_deg
        
        current_min_deg = calc_score(R)
        
        # Local search
        max_iter = 2000
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a vertex in R to remove and a vertex in V' to add (swap)
            # Or just move a vertex
            # Let's try swaps
            v_in_R = random.choice(list(R))
            v_out_R = random.choice([v for v in range(n_full) if v not in R])
            
            new_R = set(R)
            new_R.remove(v_in_R)
            new_R.add(v_out_R)
            
            new_min_deg = calc_score(new_R)
            
            # Accept if better or equal (greedy)
            # Or with probability if worse (annealing) - but we want to be fast
            if new_min_deg >= current_min_deg:
                R = new_R
                current_min_deg = new_min_deg
                
                if current_min_deg >= min_deg_req:
                    best_R = R
                    break
        
        if best_R is not None:
            # Construct edge list
            V_prime = [v for v in range(n_full) if v not in best_R]
            # Map old indices to new 0..n-1
            idx_map = {old: new for new, old in enumerate(V_prime)}
            edges = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    u = V_prime[i]
                    v = V_prime[j]
                    if adj_57[u, v]:
                        edges.append([i, j])
                        
            best_result = {'n': n_target, 'edges': edges}
            return best_result

        # If we didn't break, we might still have a good R?
        # The loop above breaks on success. If it finishes without success, we try next target.
        # But maybe we should keep the best R found?
        # For simplicity, we just move to next target which has relaxed constraints.
        
    return best_result
