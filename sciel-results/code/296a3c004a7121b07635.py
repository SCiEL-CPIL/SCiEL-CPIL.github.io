# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115.0
    
    # Initial random permutation
    p = np.random.permutation(n).astype(np.int64)
    
    def compute_cost(perm):
        b_perm = b[np.ix_(perm, perm)]
        return np.sum(a * b_perm)

    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute for delta
    # We need fast access to b rows and cols
    
    # Simulated Annealing Parameters
    T = 1000.0
    # Cooling factor: aim to cool to ~1 in ~50,000 iterations
    # T_k = T0 * exp(-alpha * k)
    # 1 = 1000 * exp(-alpha * 50000)
    # exp(-50000 * alpha) = 0.001
    # -50000 * alpha = ln(0.001) = -6.9
    # alpha = 6.9 / 50000 = 0.000138
    # cooling_factor = exp(-0.000138) = 0.999862
    cooling_factor = 0.999862
    
    def compute_delta(i, j, perm):
        u = perm[i]
        v = perm[j]
        if u == v:
            return 0
        
        # Vectors
        row_v = b[v, :]
        row_u = b[u, :]
        col_v = b[:, v]
        col_u = b[:, u]
        
        diff_row = row_v - row_u
        diff_col = col_v - col_u
        
        # S1: Sum_{k} a[i,k] * (b[v, perm[k]] - b[u, perm[k]])
        s1 = np.dot(a[i, :], diff_row[perm])
        # Subtract k=i and k=j contributions
        s1 -= a[i, i] * (b[v, u] - b[u, u])
        s1 -= a[i, j] * (b[v, v] - b[u, v])
        
        # S2: Sum_{k} a[k,i] * (b[perm[k], v] - b[perm[k], u])
        s2 = np.dot(a[:, i], diff_col[perm])
        # Subtract k=i and k=j contributions
        s2 -= a[i, i] * (b[u, v] - b[u, u])
        s2 -= a[j, i] * (b[v, v] - b[v, u])
        
        # S3: Sum_{k} a[j,k] * (b[u, perm[k]] - b[v, perm[k]])
        # = - Sum_{k} a[j,k] * (b[v, perm[k]] - b[u, perm[k]])
        s3 = -np.dot(a[j, :], diff_row[perm])
        # Add back k=i and k=j contributions
        s3 += a[j, i] * (b[v, u] - b[u, u])
        s3 += a[j, j] * (b[v, v] - b[u, v])
        
        # S4: Sum_{k} a[k,j] * (b[perm[k], u] - b[perm[k], v])
        # = - Sum_{k} a[k,j] * (b[perm[k], v] - b[perm[k], u])
        s4 = -np.dot(a[:, j], diff_col[perm])
        # Add back k=i and k=j contributions
        s4 += a[i, j] * (b[u, v] - b[u, u])
        s4 += a[j, j] * (b[v, v] - b[v, u])
        
        # Diagonal / Cross terms
        delta = s1 + s2 + s3 + s4
        delta += a[i, i] * (b[v, v] - b[u, u])
        delta += a[i, j] * (b[v, u] - b[u, v])
        delta += a[j, i] * (b[u, v] - b[v, u])
        delta += a[j, j] * (b[u, u] - b[v, v])
        
        return delta

    # SA Loop
    last_time_check = start_time
    iter_count = 0
    
    while time.time() - start_time < time_budget - 2.0:
        if iter_count % 1000 == 0 and time.time() - last_time_check > 0.1:
            last_time_check = time.time()
            if time.time() - start_time > time_budget - 2.0:
                break
        
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        delta = compute_delta(i, j, p)
        
        # Acceptance criterion
        if delta < 0:
            accept = True
        else:
            # Accept with probability exp(-delta/T)
            # Use log comparison for speed and stability
            # log(random) < -delta / T
            # random() is in (0, 1], log is in (-inf, 0]
            # If T is very small, -delta/T is large negative, condition fails.
            # If delta is 0, -delta/T is 0, log(r) < 0 is always true (r<1).
            # We should be careful with r=0. np.random.random() returns [0.0, 1.0).
            # If r=0, log(0) is -inf. -inf < x is True.
            # This is fine.
            try:
                # If T is extremely small, we might get division by zero or overflow
                if T < 1e-9:
                    accept = False
                else:
                    # Compare log(r) < -delta / T
                    # r = np.random.random()
                    # np.log(r) < -delta / T
                    # To avoid log(0) issues, we can use a small epsilon or just handle it.
                    # np.log(0) raises warning but returns -inf.
                    # Let's use a direct probability check if T is large enough, else log.
                    # Actually, log method is standard.
                    accept = np.log(np.random.random()) < (-delta / T)
            except Exception:
                accept = False
        
        if accept:
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cool down
        T *= cooling_factor
        
        iter_count += 1
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
