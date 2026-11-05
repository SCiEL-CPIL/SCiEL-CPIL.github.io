import time
import random
import numpy as np

def construct():
    """Return 256 colours in {0,1} for integers 1..256 minimizing monochromatic 4-APs."""
    n = 256
    deadline = time.time() + 110  # 40s buffer for 150s limit
    
    # Precompute all 4-term APs: a, a+d, a+2d, a+3d with 0-based indexing
    ap_indices_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_indices_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    ap_indices = np.array(ap_indices_list, dtype=np.int32)
    num_aps = len(ap_indices)
    
    # Precompute neighbor triples for each position
    # neighbors[p] is an array of shape (deg, 3)
    neighbors = []
    for p in range(n):
        triples = []
        for idx in range(num_aps):
            a, b, c, d_pos = ap_indices[idx]
            if p == a:
                triples.append((b, c, d_pos))
            elif p == b:
                triples.append((a, c, d_pos))
            elif p == c:
                triples.append((a, b, d_pos))
            elif p == d_pos:
                triples.append((a, b, c))
        if triples:
            neighbors.append(np.array(triples, dtype=np.int32))
        else:
            neighbors.append(np.empty((0, 3), dtype=np.int32))

    def count_mono_vec(x):
        """Count monochromatic 4-APs using vectorized operations."""
        # x is np.array of 0/1 (int8 or float)
        cols = x[ap_indices]
        mono = (cols[:, 0] == cols[:, 1]) & (cols[:, 1] == cols[:, 2]) & (cols[:, 2] == cols[:, 3])
        return np.sum(mono)

    def compute_deltas(x):
        """Compute marginal gain of flipping each position."""
        # x is float array
        # delta[p] = (2*x[p] - 1) * sum_{(q,r,s) in N(p)} (x[q]*x[r]*x[s] - (1-x[q])*(1-x[r])*(1-x[s]))
        # Let y = 2*x - 1. Then x = (y+1)/2.
        # Term = x_q x_r x_s - (1-x_q)(1-x_r)(1-x_s)
        # = 0.5 * (y_q y_r y_s + y_q + y_r + y_s)
        # So delta[p] = y_p * 0.5 * sum (y_q y_r y_s + y_q + y_r + y_s)
        
        y = 2.0 * x - 1.0
        deltas = np.zeros(n, dtype=np.float64)
        
        for p in range(n):
            neigh = neighbors[p]
            if len(neigh) == 0:
                continue
            
            # Extract q, r, s indices
            q_idx = neigh[:, 0]
            r_idx = neigh[:, 1]
            s_idx = neigh[:, 2]
            
            yq = y[q_idx]
            yr = y[r_idx]
            ys = y[s_idx]
            
            # Sum of (yq*yr*ys + yq + yr + ys)
            term = yq * yr * ys + yq + yr + ys
            sum_term = np.sum(term)
            
            deltas[p] = y[p] * 0.5 * sum_term
        
        return deltas

    def sa_local_search(initial, max_steps=50000):
        """Simulated annealing with fast delta computation."""
        x = initial.copy().astype(np.float64)
        current_cost = count_mono_vec(x.astype(np.int8))
        best_x = x.copy()
        best_cost = current_cost
        
        temp = 1.0
        cooling = 0.9995
        start_time = time.time()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if (step % 1000) == 0 and time.time() > deadline - 2:
                break
            
            p = random.randint(0, n - 1)
            
            # Compute delta for position p
            neigh = neighbors[p]
            if len(neigh) == 0:
                delta = 0.0
            else:
                q_idx = neigh[:, 0]
                r_idx = neigh[:, 1]
                s_idx = neigh[:, 2]
                
                yq = 2.0 * x[q_idx] - 1.0
                yr = 2.0 * x[r_idx] - 1.0
                ys = 2.0 * x[s_idx] - 1.0
                yp = 2.0 * x[p] - 1.0
                
                term = yq * yr * ys + yq + yr + ys
                sum_term = np.sum(term)
                delta = yp * 0.5 * sum_term
            
            # Acceptance
            if delta < 0:
                accept = True
            else:
                # Avoid overflow in exp
                if delta > temp * 100:
                    accept = False
                else:
                    accept = random.random() < np.exp(-delta / max(temp, 1e-10))
            
            if accept:
                x[p] = 1.0 - x[p]
                current_cost += int(delta)
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
            
            temp *= cooling
        
        return best_x.astype(np.int8), best_cost

    def local_search(initial, max_iter=1000):
        """Hill climbing: flip single best improving bit."""
        x = initial.copy().astype(np.float64)
        current_cost = count_mono_vec(x.astype(np.int8))
        best_x = x.copy()
        best_cost = current_cost
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            deltas = compute_deltas(x)
            min_delta = np.min(deltas)
            
            if min_delta >= 0:
                break
            
            # Flip the bit with most negative delta
            p = np.argmin(deltas)
            x[p] = 1.0 - x[p]
            current_cost += int(min_delta)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_x = x.copy()
        
        return best_x.astype(np.int8), best_cost

    def thue_morse():
        tm = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            tm[i] = tm[i >> 1] ^ (i & 1)
        return tm
    
    def random_coloring():
        return np.random.randint(0, 2, size=n).astype(np.int8)
    
    # Initialize with several strategies
    candidates = [thue_morse(), random_coloring(), random_coloring(), random_coloring()]
    
    best_overall = None
    best_overall_cost = float('inf')
    
    # First pass: Local search from each candidate
    for candidate in candidates:
        if time.time() > deadline - 20:
            break
        result, cost = local_search(candidate)
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall = result
    
    # Second pass: SA polish on best result
    if best_overall is not None and time.time() < deadline - 10:
        result, cost = sa_local_search(best_overall, max_steps=100000)
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall = result
    
    # Additional restarts if time permits
    if best_overall is not None and time.time() < deadline - 5:
        while time.time() < deadline - 2:
            restart = random_coloring()
            result, cost = local_search(restart)
            if cost < best_overall_cost:
                best_overall_cost = cost
                best_overall = result
            
            # Quick SA polish on improved result
            if cost < best_overall_cost + 5 and time.time() < deadline - 2:
                result2, cost2 = sa_local_search(result, max_steps=20000)
                if cost2 < best_overall_cost:
                    best_overall_cost = cost2
                    best_overall = result2
    
    return [int(c) for c in best_overall]
