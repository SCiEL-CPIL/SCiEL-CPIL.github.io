import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def compute_value(h):
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def project(h):
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(200):
        s = h.sum()
        if abs(s - target) < 1e-14:
            break
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if abs(s - target) > 1e-12:
        diff = target - s
        free = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)
    
    # Precompute correlation lags for efficiency? 
    # np.correlate is fast enough for N=100.
    
    def local_search(h_init, max_iters=5000, time_limit=10.0):
        h = h_init.copy()
        h = project(h)
        current_val = compute_value(h)
        start_time = time.time()
        
        # We want to minimize max(c). 
        # Let's use a steepest descent style swap search.
        # Try all pairs (i, j) and swap if it improves.
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            # To speed up, we can sample swaps or do full scan.
            # Full scan is O(N^2) per iter. N=100 -> 10k swaps.
            # Each swap requires recomputing value O(N) or updating O(1)?
            # Recomputing value is O(N) for correlate? No, correlate is O(N^2) generally, but numpy is fast.
            # Let's just recompute. 10k * 100 ops is 1M ops. Very fast.
            
            # Randomize order to avoid bias
            indices = np.random.permutation(N)
            
            for i in indices:
                if (time.time() - start_time) > time_limit:
                    break
                for j in indices[i+1:]:
                    if i == j: continue
                    # Swap h[i] and h[j]
                    h[i], h[j] = h[j], h[i]
                    new_val = compute_value(h)
                    if new_val < current_val - 1e-12:
                        current_val = new_val
                        improved = True
                    else:
                        # Revert
                        h[i], h[j] = h[j], h[i]
                        
        return h, current_val

    # Discrete Grid Search
    # We need a multiset of 100 values from a grid that sums to 50.
    # Symmetric grids around 0.5 are good.
    # e.g., 50 values at 0.5.
    # Let's try mixing 0.0 and 1.0. 50 zeros, 50 ones. Sum=50.
    # Let's try mixing 0.25 and 0.75. 
    # Let's try a finer grid.
    
    grids = [
        [0.0, 1.0],
        [0.25, 0.75],
        [0.0, 0.5, 1.0],
        [0.2, 0.5, 0.8],
        [0.1, 0.5, 0.9],
        [0.0, 0.25, 0.5, 0.75, 1.0]
    ]
    
    for grid in grids:
        if remaining() < 20: break
        
        # Generate random multisets from this grid that sum to 50
        # Since we need exactly 100 items, and sum 50, average is 0.5.
        # We can just pick 100 random values and adjust? 
        # Better: Pick counts for each grid value such that sum(counts)=100 and sum(counts*val)=50.
        
        # For simple grids, we can construct specific patterns.
        # Pattern 1: 50 of 0.5
        # Pattern 2: 50 of 0, 50 of 1
        # Pattern 3: 25 of 0, 50 of 0.5, 25 of 1 (Sum = 25*1 = 50)
        # Pattern 4: 50 of 0.25, 50 of 0.75
        
        candidates = []
        
        # Add specific balanced multisets
        if 0.5 in grid:
            candidates.append([0.5]*100)
        if 0.0 in grid and 1.0 in grid:
            candidates.append([0.0]*50 + [1.0]*50)
            candidates.append([0.0]*25 + [0.5]*50 + [1.0]*25) # if 0.5 in grid
        if 0.25 in grid and 0.75 in grid:
            candidates.append([0.25]*50 + [0.75]*50)
        
        # Random multisets
        # We can sample 100 values from grid, then project? No, projection changes values.
        # We want to keep values on grid for the discrete phase.
        # Let's just generate random permutations of valid multisets.
        
        for multiset in candidates:
            if remaining() < 10: break
            
            # Shuffle
            np.random.shuffle(multiset)
            h_init = np.array(multiset)
            
            # Run local search
            h_opt, val_opt = local_search(h_init, max_iters=1000, time_limit=5.0)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
                
            # Try a few random restarts with this multiset
            for _ in range(5):
                if remaining() < 10: break
                np.random.shuffle(multiset)
                h_init = np.array(multiset)
                h_opt, val_opt = local_search(h_init, max_iters=1000, time_limit=2.0)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()

    # Continuous Refinement
    # Use the best discrete solution as a starting point for a continuous optimizer.
    # We minimize a smooth approximation of the max correlation: 
    # f(h) = max(c) + (1/beta) * log(sum(exp(beta * (c - max(c)))))
    # This is the LogSumExp trick.
    
    def objective_and_grad(x, beta=50.0):
        h = x + 0.5
        c = np.correlate(h, 1.0 - h, mode="full")
        c_max = np.max(c)
        c_shifted = c - c_max
        exp_c = np.exp(beta * c_shifted)
        sum_exp = np.sum(exp_c)
        obj = c_max + np.log(sum_exp) / beta
        weights = exp_c / sum_exp

        grad_h = np.zeros(N)
        for m in range(2 * N - 1):
            w = weights[m]
            if w < 1e-10:
                continue
            if m < N:
                # Contribution to gradient from lag m
                # c_m = sum_k h[k] (1-h[k+m]) for m>0? 
                # correlate(h, 1-h) definition:
                # c[k] = sum_n h[n] (1-h[n+k]) for k>=0? 
                # np.correlate(a, v, 'full') returns c where c[k] = sum_n a[n] v[n+k]
                # Here a=h, v=1-h.
                # c[m] = sum_{n=0}^{N-1-m} h[n] (1-h[n+m]) for m >= 0.
                # c[-m] = sum_{n=m}^{N-1} h[n] (1-h[n-m]) for m > 0.
                
                # Let's just use the finite difference or analytical deriv.
                # d c_m / d h[i]
                # If i is in the first sum (n=i), term is (1-h[i+m]).
                # If i is in the second part (n+m=i => n=i-m), term is -h[i-m].
                
                # So grad_h[i] += w * ( (1-h[i+m]) if i+m<N else 0 ) - w * ( h[i-m] if i-m>=0 else 0 )
                
                # Vectorized update:
                # Part 1: w * (1-h[i+m]) for i in [0, N-1-m]
                if m < N:
                    # i ranges 0 to N-1-m
                    grad_h[:N-m] += w * (1.0 - h[m:])
                    # i ranges m to N-1
                    grad_h[m:] -= w * h[:N-m]
        
        return obj, grad_h

    # Run SLSQP or L-BFGS-B
    if remaining() > 20:
        x0 = best_h - 0.5
        bounds = [(-0.5, 0.5)] * N
        
        # Add a constraint for sum(h) = 50 => sum(x) = 0
        # SLSQP handles equality constraints well.
        # L-BFGS-B doesn't. We used projection before.
        # Let's use SLSQP for the final polish.
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 0.0, 'jac': lambda x: np.ones(N)}]
        
        try:
            res = minimize(
                objective_and_grad,
                x0,
                args=(100.0,),
                method='SLSQP',
                jac=True,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res.success:
                h_ref = res.x + 0.5
                h_ref = project(h_ref) # Just in case
                val_ref = compute_value(h_ref)
                if val_ref < best_val:
                    best_val = val_ref
                    best_h = h_ref.copy()
        except Exception:
            pass
            
        # Try L-BFGS-B with projection if SLSQP failed or for diversity
        if remaining() > 10:
            def obj_lbfgs(x):
                h = x + 0.5
                c = np.correlate(h, 1.0 - h, mode="full")
                c_max = np.max(c)
                c_shifted = c - c_max
                exp_c = np.exp(50.0 * c_shifted)
                sum_exp = np.sum(exp_c)
                obj = c_max + np.log(sum_exp) / 50.0
                weights = exp_c / sum_exp
                grad_h = np.zeros(N)
                for m in range(2 * N - 1):
                    w = weights[m]
                    if w < 1e-10: continue
                    if m < N:
                        grad_h[:N-m] += w * (1.0 - (x[m:] + 0.5))
                        grad_h[m:] -= w * (x[:N-m] + 0.5)
                return obj, grad_h

            try:
                res2 = minimize(
                    obj_lbfgs,
                    best_h - 0.5,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                h_ref2 = res2.x + 0.5
                h_ref2 = project(h_ref2)
                val_ref2 = compute_value(h_ref2)
                if val_ref2 < best_val:
                    best_val = val_ref2
                    best_h = h_ref2.copy()
            except Exception:
                pass

    return best_h
