import numpy as np
from scipy.optimize import minimize, basinhopping

def kakeya_needle_2d_positions64() -> np.ndarray:
        n = 64
        w = 1.0 / n
        a = (np.arange(1, n + 1)) / n
        
        # Keich construction
        k = int(np.log2(n))
        j = np.arange(1, k + 1)
        x_keich = np.zeros(n)
        for i in range(n):
            eps = np.array(list(str(bin(i))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            x_keich[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
        
        # Quadrature points
        M = 20000
        y = (np.arange(M) + 0.5) / M
        
        def objective(x):
            # x: shape (n,)
            # L: shape (n, M)
            L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
            # Sort L along axis 0
            L_sorted = np.sort(L, axis=0)
            # Differences
            d = L_sorted[1:, :] - L_sorted[:-1, :]
            # w_y
            w_y = w * (1 - y)
            # Union length: w_y + sum(min(d, w_y))
            # min(d, w_y) is element-wise
            union_len = w_y + np.sum(np.minimum(d, w_y), axis=0)
            area = np.mean(union_len)
            return area
        
        # Optimization
        # Start from Keich
        x0 = x_keich.copy()
        
        # Nelder-Mead settings
        nm_opts = {
            'maxiter': 2000,
            'maxfev': 2000,
            'xatol': 1e-8,
            'fatol': 1e-8
        }
        
        # Basin Hopping
        # Define a perturbation method
        def perturb(x, T):
            # Gaussian perturbation
            return x + T * np.random.randn(*x.shape)
        
        # We might need to be careful with the time limit.
        # Let's run Basin Hopping with a time limit check if possible, or just limit iterations.
        # Since we can't easily interrupt scipy, we limit maxiter.
        
        # Let's try a few restarts manually to be safe with time.
        best_x = x0
        best_f = objective(x0)
        
        # Run a few Nelder-Mead steps from different perturbations of Keich
        # Or just one Basin Hopping run with limited iterations.
        
        # Given the 600s limit, and ~50ms per eval, we can afford ~10000 evals.
        # BH with maxiter=50, local_maxiter=100 -> 50*100 = 5000 evals approx.
        
        bh_res = basinhopping(
            objective, 
            x0, 
            niter=20, 
            minimizer_kwargs={'method': 'Nelder-Mead', 'options': nm_opts},
            stepsize=0.1,
            interval=5,
            disp=False
        )
        
        if bh_res.fun < best_f:
            best_x = bh_res.x
            best_f = bh_res.fun
            
        # Ensure valid output
        if not np.all(np.isfinite(best_x)):
            return x_keich
            
        return best_x
