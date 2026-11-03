import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Heuristic Rank Matching
    # This provides a good starting point for SA
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
    time_limit = 115.0
    
    # Simulated Annealing Parameters
    # Initial temperature should be high enough to accept bad moves initially
    # Estimate average delta magnitude to set T0
    # We can estimate by running a few random swaps
    T = 10000.0  # Initial temperature
    cooling_rate = 0.9995  # Slow cooling to explore more
    min_T = 1e-6
    
    # To make it more robust, we can use a "restart" strategy if stuck
    # But standard SA with slow cooling is a distinct family from ILS.
    
    # Precompute indices for speed?
    # The delta calculation is the bottleneck.
    
    while time.time() - start_time < time_limit and T > min_T:
        # Pick two random distinct facilities to swap
        i = random.randint(0, n - 1)
        k = random.randint(0, n - 1)
        
        if i == k:
            continue
            
        u = p[i]
        v = p[k]
        
        # Calculate Delta Cost for swapping i and k
        # Delta = sum_j (A[i,j] - A[k,j]) * (B[v, p[j]] - B[u, p[j]])
        # Correction for j=i and j=k terms
        diff_a = a[i] - a[k]
        diff_b = b[v, p] - b[u, p]
        
        S = np.dot(diff_a, diff_b)
        
        # Subtract self and cross terms included in S
        # j=i term: (a[i,i]-a[k,i]) * (b[v, u] - b[u, u])
        # j=k term: (a[i,k]-a[k,k]) * (b[v, v] - b[u, v])
        S -= (a[i, i] - a[k, i]) * (b[v, u] - b[u, u])
        S -= (a[i, k] - a[k, k]) * (b[v, v] - b[u, v])
        
        # Add correct new terms for i, k interaction and self loops
        delta = S + (a[i, k] * (b[v, u] - b[u, v]) + 
                     a[k, i] * (b[u, v] - b[v, u]) + 
                     a[i, i] * (b[v, v] - b[u, u]) + 
                     a[k, k] * (b[u, u] - b[v, v]))
        
        # Acceptance Step
        if delta < 0:
            # Accept improvement
            p[i], p[k] = p[k], p[i]
            cost += delta
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        else:
            # Accept deterioration with probability exp(-delta/T)
            # Optimization: if T is very small, exp(-delta/T) is 0
            if T > 0:
                # Use random() < exp(-delta/T)
                # To avoid overflow/underflow issues, compare log(random) < -delta/T
                # But random() is (0,1), log is negative.
                # Simpler: random() < np.exp(-delta / T)
                # For speed, if delta/T > 50, exp is ~0
                if delta / T < 50:
                    if random.random() < np.exp(-delta / T):
                        p[i], p[k] = p[k], p[i]
                        cost += delta
                        if cost < best_cost:
                            best_cost = cost
                            best_p = p.copy()
        
        # Cool down
        T *= cooling_rate
        
        # Recalculate cost periodically to avoid floating point drift?
        # Since we are using integer arithmetic for cost and delta, drift is minimal.
        # But p is int64, a,b are int64. cost is int64.
        # np.dot returns float64 if inputs are float, but a,b are int64.
        # Wait, a and b are int64. diff_a is int64. diff_b is int64.
        # np.dot(int64, int64) returns int64? No, usually float64 or int64 depending on version.
        # Let's ensure we cast to int to be safe, or rely on Python int if we convert.
        # Actually, numpy dot on int64 arrays returns int64.
        # So cost remains int64. No drift.
        
    return best_p.tolist()
