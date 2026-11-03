import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0
    
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    def swap_delta(p, i, j):
        """Compute cost delta for swapping positions i and j."""
        u, v = p[i], p[j]
        if u == v:
            return 0
        # For swapping facilities u and v at positions i and j:
        # delta = sum_{k != i,j} [(a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])]
        #       + a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u])
        #       - terms where k=i or k=j in the sums above
        # 
        # Let's compute it more carefully:
        # Old cost contribution involving i or j:
        #   sum_k a[i,k]*b[u,p[k]] + sum_k a[k,i]*b[p[k],u] + sum_k a[j,k]*b[v,p[k]] + sum_k a[k,j]*b[p[k],v]
        #   (but this double counts a[i,j]*b[u,v] and a[j,i]*b[v,u])
        # 
        # New cost after swap (i gets v, j gets u):
        #   sum_k a[i,k]*b[v,p[k]] + sum_k a[k,i]*b[p[k],v] + sum_k a[j,k]*b[u,p[k]] + sum_k a[k,j]*b[p[k],u]
        #   (double counts a[i,j]*b[v,u] and a[j,i]*b[u,v])
        #
        # Delta = new - old
        # For k not in {i,j}:
        #   (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
        # For k=i: 
        #   old has a[i,i]*b[u,u] + a[i,j]*b[u,v] (from row i) and a[i,i]*b[u,u] + a[j,i]*b[v,u] (from col i) -- wait this is getting complex
        # Let me just use the standard formula:
        
        diff_a_ir = a[i, :] - a[j, :]
        diff_a_ci = a[:, i] - a[:, j]
        diff_b_ur = b[u, :] - b[v, :]
        diff_b_uc = b[:, u] - b[:, v]
        
        # Sum over all k
        s1 = np.dot(diff_a_ir, diff_b_ur[p])
        s2 = np.dot(diff_a_ci, diff_b_uc[p])
        
        # Subtract terms where k=i and k=j
        # When k=i: term in s1 is diff_a_ir[i]*diff_b_ur[p[i]] = (a[i,i]-a[j,i])*(b[u,u]-b[v,u])
        #           term in s2 is diff_a_ci[i]*diff_b_uc[p[i]] = (a[i,i]-a[i,j])*(b[u,u]-b[v,u])
        # When k=j: term in s1 is diff_a_ir[j]*diff_b_ur[p[j]] = (a[i,j]-a[j,j])*(b[u,v]-b[v,v])
        #           term in s2 is diff_a_ci[j]*diff_b_uc[p[j]] = (a[j,i]-a[j,j])*(b[u,v]-b[v,v])
        
        s1 -= (a[i,i]-a[j,i])*(b[u,u]-b[v,u]) + (a[i,j]-a[j,j])*(b[u,v]-b[v,v])
        s2 -= (a[i,i]-a[i,j])*(b[u,u]-b[v,u]) + (a[j,i]-a[j,j])*(b[u,v]-b[v,v])
        
        # The cross terms: a[i,j]*b[u,v] appears in old (row i, col j) and new has a[i,j]*b[v,u]
        # a[j,i]*b[v,u] appears in old (row j, col i) and new has a[j,i]*b[u,v]
        cross = a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u])
        
        return s1 + s2 + cross
    
    rng = np.random.RandomState(42)
    
    # Initial solution: greedy by row sums
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Simulated Annealing with restarts
    T_start = max(1.0, abs(current_cost) * 0.005)
    T_end = max(0.1, T_start * 0.0001)
    alpha = 0.997  # slower cooling
    
    T = T_start
    step_count = 0
    no_improve_steps = 0
    restart_threshold = 5000
    
    while time.time() < deadline - 5.0:
        i, j = rng.choice(n, size=2, replace=False)
        if i == j:
            continue
        delta = swap_delta(p, i, j)
        
        # Acceptance criterion
        if delta < 0 or rng.random() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            step_count += 1
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        
        # Cool down
        T = max(T_end, T * alpha)
        
        # Restart if stuck or if temperature is very low
        if (no_improve_steps > restart_threshold or T < T_start * 0.001) and time.time() < deadline - 15.0:
            # Perturb the best solution
            p = best_p.copy()
            current_cost = best_cost
            # Apply random swaps
            num_swaps = rng.randint(5, 15)
            for _ in range(num_swaps):
                i2, j2 = rng.choice(n, size=2, replace=False)
                if i2 != j2:
                    p[i2], p[j2] = p[j2], p[i2]
            current_cost = compute_cost(p)
            T = T_start * 0.5
            no_improve_steps = 0
            restart_threshold = int(restart_threshold * 1.2)
        
        # Periodic local search refinement (best-improvement 2-swap)
        if step_count % 2000 == 0 and time.time() < deadline - 10:
            for _ in range(2):
                improved = False
                for i in range(n):
                    if time.time() > deadline - 10:
                        break
                    # Vectorized: compute all deltas for swapping i with j > i
                    diff_a_ir = a[i, :] - a[i, :]  # placeholder
                    # Actually let's just do it sequentially but efficiently
                    for j in range(i + 1, n):
                        delta = swap_delta(p, i, j)
                        if delta < 0:
                            p[i], p[j] = p[j], p[i]
                            current_cost += delta
                            improved = True
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_p = p.copy()
                            break
                    if improved:
                        break
                if not improved:
                    break
    
    return best_p.tolist()
