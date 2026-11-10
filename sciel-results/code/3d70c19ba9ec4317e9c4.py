import numpy as np
import scipy.optimize
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    start_time = time.time()
    deadline = start_time + 628  # 2s safety margin before 630s limit
    
    # Precompute constants
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n  # Shape (M,)
    A = a[:, None] * y[None, :]  # Shape (n, M)
    
    def calc_area(x_vals):
        """Compute area only (faster, no gradient)."""
        L = x_vals[:, None] + A
        L_sorted = np.sort(L, axis=0)
        d = L_sorted[1:, :] - L_sorted[:-1, :]
        mins = np.minimum(d, w[None, :])
        union_lens = w + np.sum(mins, axis=0)
        return np.mean(union_lens)
    
    def calc_area_and_grad(x_vals):
        """Compute area and gradient."""
        L = x_vals[:, None] + A
        I = np.argsort(L, axis=0)
        L_sorted = np.take_along_axis(L, I, axis=0)
        d = L_sorted[1:, :] - L_sorted[:-1, :]
        overlap_mask = d < w[None, :]
        
        grad_L_sorted = np.zeros_like(L)
        for k in range(n - 1):
            mask = overlap_mask[k, :]
            grad_L_sorted[k, mask] -= 1.0
            grad_L_sorted[k + 1, mask] += 1.0
        
        grad_x = np.zeros(n)
        np.add.at(grad_x, I, grad_L_sorted)
        grad_x /= M
        
        mins = np.minimum(d, w[None, :])
        union_lens = w + np.sum(mins, axis=0)
        area = np.mean(union_lens)
        
        return area, grad_x
    
    def smooth_calc_area(x_vals, eps):
        """Smooth surrogate using soft-min approximation."""
        L = x_vals[:, None] + A
        L_sorted = np.sort(L, axis=0)
        d = L_sorted[1:, :] - L_sorted[:-1, :]
        d_clipped = np.maximum(d, 1e-10)
        max_d = np.max(d_clipped, axis=0)
        # Soft-min: -eps*log(mean(exp(-d/eps)))
        # Using stable log-sum-exp
        exp_terms = np.exp(-(d_clipped - max_d[None, :]) / eps)
        mean_exp = np.mean(exp_terms, axis=0)
        soft_min = -eps * np.log(mean_exp) - max_d
        union_lens = w + np.sum(soft_min, axis=0)
        return np.mean(union_lens)
    
    def smooth_calc_area_and_grad(x_vals, eps):
        """Compute smooth surrogate area and gradient."""
        L = x_vals[:, None] + A
        I = np.argsort(L, axis=0)
        L_sorted = np.take_along_axis(L, I, axis=0)
        d = L_sorted[1:, :] - L_sorted[:-1, :]
        d_clipped = np.maximum(d, 1e-10)
        
        max_d = np.max(d_clipped, axis=0)
        exp_terms = np.exp(-(d_clipped - max_d[None, :]) / eps)
        sum_exp = np.sum(exp_terms, axis=0)
        
        # grad of soft_min w.r.t. d_k = -exp_term_k / sum_exp
        grad_d = -exp_terms / sum_exp[None, :]
        
        # Chain rule: d_k = L_sorted[k+1] - L_sorted[k]
        grad_L_sorted = np.zeros_like(L)
        for k in range(n - 1):
            grad_L_sorted[k, :] -= grad_d[k, :]
            grad_L_sorted[k + 1, :] += grad_d[k, :]
        
        grad_x = np.zeros(n)
        np.add.at(grad_x, I, grad_L_sorted)
        grad_x /= M
        
        mean_exp = np.mean(exp_terms, axis=0)
        soft_min = -eps * np.log(mean_exp) - max_d
        union_lens = w + np.sum(soft_min, axis=0)
        area = np.mean(union_lens)
        
        return area, grad_x

    # Generate candidate initial points
    def keich_initial(i, n):
        k = int(np.log2(n))
        j = np.arange(1, k + 1)
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        return np.sum((1 - j) * eps * (2.0 ** -j) / k)

    def random_initial(n, seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(-0.5, 1.5, n)

    def structured_initial(n, seed):
        """Try a structured pattern based on sine/cosine."""
        rng = np.random.RandomState(seed)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return rng.uniform(-0.5, 1.5, n) * 0.3 + 0.5 * np.sin(t) + 0.5 * np.cos(2 * t)

    def sine_pattern(n, seed, freq_mult=1.0):
        """Pure sine/cosine pattern without randomness."""
        rng = np.random.RandomState(seed)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        amp = rng.uniform(0.3, 0.8)
        phase = rng.uniform(0, 2 * np.pi)
        return amp * np.sin(freq_mult * t + phase) + 0.5 * np.cos(2 * freq_mult * t)

    candidates = []
    
    # Keich construction
    x_keich = np.array([keich_initial(i, n) for i in range(n)])
    candidates.append(x_keich)
    
    # Perturbed Keich
    rng_keich = np.random.RandomState(0)
    for i in range(10):
        noise = rng_keich.normal(0, 0.05, n)
        candidates.append(x_keich + noise)
        
    # Multiple random starts
    for seed in range(20):
        candidates.append(random_initial(n, seed))
    
    # Some structured starts
    for seed in range(10):
        candidates.append(structured_initial(n, seed))
    
    # Sine patterns with different frequencies
    for seed in range(10):
        for fm in [1.0, 2.0, 3.0]:
            candidates.append(sine_pattern(n, seed + 100 * int(fm), fm))
    
    # Evaluate all candidates
    best_x = candidates[0].copy()
    best_area = calc_area(best_x)
    
    for i, x_cand in enumerate(candidates):
        if time.time() > deadline:
            break
        area = calc_area(x_cand)
        if area < best_area:
            best_area = area
            best_x = x_cand.copy()
            
    # Optimization phases
    bounds = [(-2.0, 3.0) for _ in range(n)]
    
    def obj_func(x_vals):
        return calc_area(x_vals)
    
    def grad_func(x_vals):
        _, grad = calc_area_and_grad(x_vals)
        return grad
        
    # Phase 1: Sharp optimization from best candidate
    try:
        res = scipy.optimize.minimize(
            obj_func,
            best_x,
            method='L-BFGS-B',
            jac=grad_func,
            bounds=bounds,
            options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-7, 'disp': False}
        )
        if res.fun < best_area:
            best_area = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x

    # Phase 2: Annealed Smooth Surrogate Optimization
    # Start with large eps, decrease it
    eps_schedule = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for eps_val in eps_schedule:
        if time.time() > deadline - 30:
            break
            
        try:
            res_smooth = scipy.optimize.minimize(
                lambda x: smooth_calc_area(x, eps=eps_val),
                best_x,
                method='L-BFGS-B',
                jac=lambda x: smooth_calc_area_and_grad(x, eps=eps_val)[1],
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
            )
            # Check if the smooth solution is actually better on the sharp objective
            sharp_area = calc_area(res_smooth.x)
            if sharp_area < best_area:
                best_area = sharp_area
                best_x = res_smooth.x.copy()
        except Exception:
            pass

    if time.time() > deadline:
        return best_x

    # Phase 3: Multi-restart with perturbation and re-optimization
    if time.time() < deadline - 40:
        rng = np.random.RandomState(42)
        for restart in range(10):
            if time.time() > deadline - 15:
                break
            
            # Perturb with decreasing noise
            noise_scale = 0.05 * (1.0 - restart * 0.08)
            x_perturbed = best_x + rng.normal(0, noise_scale, n)
            
            try:
                res2 = scipy.optimize.minimize(
                    obj_func,
                    x_perturbed,
                    method='L-BFGS-B',
                    jac=grad_func,
                    bounds=bounds,
                    options={'maxiter': 250, 'ftol': 1e-10, 'gtol': 1e-7, 'disp': False}
                )
                if res2.fun < best_area:
                    best_area = res2.fun
                    best_x = res2.x.copy()
            except Exception:
                pass
            
            # Quick coordinate descent after each restart
            if time.time() < deadline - 10:
                # Simple coordinate descent
                for _ in range(1):
                    for i in range(n):
                        if time.time() > deadline:
                            break
                        x_i_old = best_x[i]
                        
                        def obj_1d(xi):
                            x_new = best_x.copy()
                            x_new[i] = xi
                            return calc_area(x_new)
                        
                        lo = x_i_old - 0.1
                        hi = x_i_old + 0.1
                        # Golden section search
                        gr = (np.sqrt(5) - 1) / 2
                        a1 = hi - gr * (hi - lo)
                        b1 = lo + gr * (hi - lo)
                        fa = obj_1d(a1)
                        fb = obj_1d(b1)
                        
                        for _ in range(20):
                            if time.time() > deadline:
                                break
                            if fa < fb:
                                hi = b1
                                b1 = a1
                                fb = fa
                                a1 = hi - gr * (hi - lo)
                                fa = obj_1d(a1)
                            else:
                                lo = a1
                                a1 = b1
                                fa = fb
                                b1 = lo + gr * (hi - lo)
                                fb = obj_1d(b1)
                        
                        best_1d = (lo + hi) / 2
                        f_best = obj_1d(best_1d)
                        if f_best < obj_1d(x_i_old) - 1e-12:
                            best_x[i] = best_1d

    # Final tight polish if time remains
    if time.time() < deadline - 5:
        try:
            res_final = scipy.optimize.minimize(
                obj_func,
                best_x,
                method='L-BFGS-B',
                jac=grad_func,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9, 'disp': False}
            )
            if res_final.fun < best_area:
                best_area = res_final.fun
                best_x = res_final.x.copy()
        except Exception:
            pass
    
    return best_x
