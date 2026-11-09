# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Minimize the area of the union of 64 Kakeya needle triangles using gradient-based optimization.
    """
    n = 64
    k = int(np.log2(n))
    
    # Initial guess: Keich construction
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    # Parameters
    a = (np.arange(n) + 1) / n
    M = 20000
    y = (np.arange(M) + 0.5) / M
    
    # Precompute constants for speed
    # L_i(y) = x_i + a_i * y
    # w(y) = (1 - y) / n
    # Union length = w + sum_{j=1}^{n-1} min(L_{(j+1)} - L_{(j)}, w)
    
    # We use a vectorized approach for the objective and gradient.
    # To save memory and time, we process in chunks or rely on NumPy efficiency.
    # M=20000, n=64 -> 1.28M floats for L, which is fine.
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 500.0 # Leave 100s buffer
    
    def objective_and_grad(x):
        """
        Computes area (mean union length) and its gradient.
        x: shape (n,)
        """
        # L shape (M, n)
        L = x[None, :] + a[None, :] * y[:, None]
        
        # Sort L along axis 1
        # We need indices to compute gradient w.r.t x
        # Let S be the sorted L. S[:, j] = L[:, perm[j]]
        # Gradient of S[:, j] w.r.t x_i is 1 if perm[j] == i else 0.
        # Actually, we can compute the gradient by back-propagating through the sort.
        
        # For each row m:
        # U_m = w_m + sum_{j=0}^{n-2} min(S_m[j+1] - S_m[j], w_m)
        # Let D_m[j] = S_m[j+1] - S_m[j]
        # Let G_m[j] = min(D_m[j], w_m)
        # U_m = w_m + sum G_m[j]
        
        # Gradient of U_m w.r.t x_i:
        # dU_m/dx_i = sum_{j} dG_m[j]/dx_i
        # dG_m[j]/dx_i = d(min(S[j+1]-S[j], w))/dx_i
        # If S[j+1]-S[j] < w:
        #   dG/dx_i = d(S[j+1]-S[j])/dx_i = dS[j+1]/dx_i - dS[j]/dx_i
        # Else:
        #   dG/dx_i = 0 (since w is constant w.r.t x)
        
        # Note: dS[j]/dx_i is 1 if the i-th element of L is at position j in the sorted order, else 0.
        
        S = np.sort(L, axis=1)
        w = (1.0 - y) / n
        
        D = np.diff(S, axis=1) # shape (M, n-1)
        G = np.minimum(D, w[:, None]) # shape (M, n-1)
        
        area = np.mean(w + np.sum(G, axis=1))
        
        # Gradient computation
        # We need to know for each row m and each gap j, which indices i and k are the ones
        # forming the gap (i is at pos j, k is at pos j+1).
        # We only need the gradient if D < w.
        
        # Create a matrix H of shape (M, n) where H[m, i] accumulates the gradient contribution.
        # H[m, i] = sum_{j where D_m[j] < w_m} (1_{perm_m[j+1]==i} - 1_{perm_m[j]==i})
        
        # This is effectively:
        # For each gap j in row m:
        #   if D_m[j] < w_m:
        #     H[m, perm_m[j+1]] += 1
        #     H[m, perm_m[j]] -= 1
        
        # We can do this efficiently using advanced indexing or sparse logic.
        # However, doing it in a loop over M might be slow in Python.
        # Let's try to vectorize.
        
        # Get permutation indices
        # np.argsort(L, axis=1) gives the indices that would sort L.
        # Let P be the inverse permutation? No, we need to map sorted positions back to original indices.
        # If S[:, j] = L[:, P[:, j]], then P[:, j] is the original index.
        P = np.argsort(L, axis=1)
        
        # P shape (M, n). P[:, j] is the index in x that corresponds to the j-th smallest value.
        
        # We need to scatter updates to a gradient array grad_x of shape (n,).
        # grad_x[i] = sum_{m} (1/M) * sum_{j=0}^{n-2} (1_{P[m, j+1]==i} - 1_{P[m, j]==i}) * 1_{D[m, j] < w[m]}
        
        # Let's compute the term T[m, j] = 1_{D[m, j] < w[m]}
        T = (D < w[:, None]).astype(np.float64) # shape (M, n-1)
        
        # We want to compute:
        # grad_x[i] = (1/M) * sum_{m, j} T[m, j] * (1_{P[m, j+1]==i} - 1_{P[m, j]==i})
        
        # This can be viewed as:
        # grad_x = (1/M) * (Sum_{m, j} T[m, j] * OneHot(P[m, j+1]) - Sum_{m, j} T[m, j] * OneHot(P[m, j]))
        
        # Let's use np.bincount or scatter.
        # Flatten P and T.
        
        # Term 1: +1 at P[m, j+1]
        # We need to flatten P[:, 1:] and T.
        # P_plus = P[:, 1:].ravel()
        # T_flat = T.ravel()
        # grad_plus = np.bincount(P_plus, weights=T_flat, minlength=n)
        
        # Term 2: -1 at P[m, j]
        # P_minus = P[:, :-1].ravel()
        # grad_minus = np.bincount(P_minus, weights=T_flat, minlength=n)
        
        # grad_x = (grad_plus - grad_minus) / M
        
        P_plus = P[:, 1:].ravel()
        P_minus = P[:, :-1].ravel()
        T_flat = T.ravel()
        
        grad_plus = np.bincount(P_plus, weights=T_flat, minlength=n)
        grad_minus = np.bincount(P_minus, weights=T_flat, minlength=n)
        
        grad_x = (grad_plus - grad_minus) / M
        
        return area, grad_x

    # Optimization
    # We can run multiple restarts if time permits, but L-BFGS-B is quite robust.
    # Let's run one main optimization with a tight maxiter to stay within time.
    
    best_x = x0
    best_area = np.inf
    
    # Define objective function for scipy
    def obj_func(x):
        val, _ = objective_and_grad(x)
        return val
    
    def jac_func(x):
        _, grad = objective_and_grad(x)
        return grad
    
    # Combine for efficiency if possible, but scipy calls them separately by default.
    # We can use a wrapper that caches the last computation if x is the same, 
    # but scipy L-BFGS-B usually doesn't call them with the same x in immediate succession 
    # in a way that simple caching helps much without overhead.
    # However, we can just define a single function that returns both? No, scipy expects separate.
    
    # Let's just use the separate functions.
    
    try:
        res = minimize(
            obj_func, 
            x0, 
            method='L-BFGS-B', 
            jac=jac_func, 
            options={'maxiter': 2000, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() < deadline:
             best_x = res.x
             best_area = res.fun
             
    except Exception:
        pass
    
    # If we have time, we can do a second pass or local refinement, but usually 
    # L-BFGS-B converges well. 
    # Let's check if we should do a restart from a different point?
    # The Keich point is a good local minimum. 
    # Maybe try a perturbed start?
    
    if time.time() < deadline - 100:
        # Try a random restart to see if we can find a better basin
        np.random.seed(42)
        x_rand = x0 + np.random.normal(0, 0.01, n)
        
        try:
            res2 = minimize(
                obj_func, 
                x_rand, 
                method='L-BFGS-B', 
                jac=jac_func, 
                options={'maxiter': 500, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() < deadline:
                 if res2.fun < best_area:
                     best_x = res2.x
                     best_area = res2.fun
        except Exception:
            pass

    return np.array(best_x, dtype=np.float64)

# EVOLVE-BLOCK-END
