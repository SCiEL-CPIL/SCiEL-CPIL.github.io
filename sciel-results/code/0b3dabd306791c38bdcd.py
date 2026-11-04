import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Heuristic Rank Matching
    # Assign facilities with high flow to locations with low distance sum (central)
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    fac_order = np.argsort(-flow_sums, kind="stable")
    loc_order = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    
    # Calculate initial cost
    B_perm = b[np.ix_(p, p)]
    cost = np.sum(a * B_perm)
    
    best_p = p.copy()
    best_cost = cost
    
    start_time = time.time()
    time_limit = 115  # Leave buffer for overhead
    
    # Simulated Annealing Parameters
    # Initial temperature should be high enough to accept worse solutions
    # Heuristic: T0 ~ (avg cost improvement per step) / (-ln(0.5)) or similar
    # For QAP, costs are large. Let's estimate initial T based on typical delta magnitude.
    # A safe heuristic: T0 = 1000000.0 or scale with n.
    T = 500000.0 
    T_min = 1e-5
    # We want to cool down over the duration.
    # Let's estimate number of iterations.
    # Each iteration is O(n) for delta. n=150. 
    # We can do many iterations. Let's aim for ~500,000 iterations.
    # Cooling factor alpha such that T * alpha^iter < T_min
    # alpha = (T_min / T)^(1/iter)
    # Let's just use a fixed alpha and check time.
    alpha = 0.99995
    
    # To ensure we finish, we can pre-calculate max iterations or just loop until time.
    # Looping until time is safer.
    
    while time.time() - start_time < time_limit:
        # Pick two random distinct indices
        i = random.randint(0, n - 1)
        k = random.randint(0, n - 1)
        
        if i == k:
            continue
            
        u = p[i]
        v = p[k]
        
        # Vectorized Delta Calculation for swap (i, k)
        # S = sum_{j} (A[i,j] - A[k,j]) * (B[v, p[j]] - B[u, p[j]])
        diff_a = a[i] - a[k]
        diff_b = b[v, p] - b[u, p]
        S = np.dot(diff_a, diff_b)
        
        # Subtract self and cross terms included in S
        # j=i term: (a[i,i]-a[k,i]) * (b[v, p[i]] - b[u, p[i]]) = (a[i,i]-a[k,i]) * (b[v, u] - b[u, u])
        # j=k term: (a[i,k]-a[k,k]) * (b[v, p[k]] - b[u, p[k]]) = (a[i,k]-a[k,k]) * (b[v, v] - b[u, v])
        
        S -= (a[i, i] - a[k, i]) * (b[v, u] - b[u, u])
        S -= (a[i, k] - a[k, k]) * (b[v, v] - b[u, v])
        
        # Add correct new terms for i, k interaction and self loops
        # New - Old:
        # (a[i,k]*b[v,u] + a[k,i]*b[u,v] + a[i,i]*b[v,v] + a[k,k]*b[u,u])
        # - (a[i,k]*b[u,v] + a[k,i]*b[v,u] + a[i,i]*b[u,u] + a[k,k]*b[v,v])
        
        delta = S + (a[i, k] * (b[v, u] - b[u, v]) + 
                     a[k, i] * (b[u, v] - b[v, u]) + 
                     a[i, i] * (b[v, v] - b[u, u]) + 
                     a[k, k] * (b[u, u] - b[v, v]))
        
        # Acceptance Criterion
        if delta < 0:
            # Always accept improvement
            p[i], p[k] = p[k], p[i]
            cost += delta
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        else:
            # Accept worse solution with probability exp(-delta / T)
            # If T is very small, this probability is near 0
            if T > 1e-10:
                # Optimization: if delta is very large compared to T, exp is 0
                # log_prob = -delta / T
                # if log_prob < -50, prob is ~0
                if delta / T < 50:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        p[i], p[k] = p[k], p[i]
                        cost += delta
        
        # Cooling
        T *= alpha
        
        # If T becomes too small, we can stop or reset? 
        # Usually, we just stop accepting non-improvements effectively.
        # But we must return a valid permutation.
        # We can break if T < T_min to save time, but we might still want to do some local search?
        # No, stick to SA. If T is tiny, it behaves like greedy hill climbing.
        # We can accelerate cooling if we detect stagnation? 
        # For simplicity, fixed schedule.
        
        # To prevent infinite loop if alpha is very close to 1, we rely on time limit.
        # However, if T drops to 0, we are just doing greedy swaps.
        # We can add a check: if T < T_min, we could switch to a few local search steps?
        # No, keep it pure SA. The time limit will stop it.
        
        # Optional: Speed up by breaking if T is negligible and no improvement for a while?
        # Let's just let it run. The random.random() check is fast.

    return best_p.tolist()
