import numpy as np
import scipy.optimize
import time

N = 100
TARGET = N / 2.0

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for the 150s limit

    def get_correlations(h):
        """
        Compute linear correlations of h and 1-h for all non-zero shifts.
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # Exclude shift 0 (index N-1)
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero

    def project_to_feasible(h):
        """
        Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2.
        Uses bisection on a Lagrange multiplier for exact projection.
        """
        h = np.clip(h, 0.0, 1.0).copy()
        target = N / 2.0
        current = np.sum(h)
        
        if abs(current - target) < 1e-12:
            return h
        
        lo = -2.0
        hi = 2.0
        
        def sum_at_lambda(lam):
            return np.sum(np.clip(h - lam, 0.0, 1.0))
        
        for _ in range(60):
            mid = (lo + hi) / 2.0
            s_mid = sum_at_lambda(mid)
            if abs(s_mid - target) < 1e-12:
                break
            if s_mid > target:
                lo = mid
            else:
                hi = mid
        
        lam = (lo + hi) / 2.0
        h = np.clip(h - lam, 0.0, 1.0)
        
        # Final adjustment for any residual error due to floating point
        residual = target - np.sum(h)
        if abs(residual) > 1e-14:
            interior = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.sum(interior) > 0:
                h[interior] += residual / np.sum(interior)
                h = np.clip(h, 0.0, 1.0)
        
        return h

    def lse_objective(h, beta):
        """
        LogSumExp of correlations for smooth approximation of max.
        """
        c_all = get_correlations(h)
        max_c = np.max(c_all)
        if max_c > 500:
            max_c = 500.0
        lse = max_c + np.log(np.sum(np.exp(beta * (c_all - max_c)))) / beta
        return lse

    def epigraph_objective(h, t, beta):
        """
        Epigraph formulation: minimize t subject to c_i <= t for all i.
        Uses smooth penalty for constraint violations.
        """
        c_all = get_correlations(h)
        violations = np.maximum(0, c_all - t)
        penalty = np.sum(violations ** 2) * beta
        return t + penalty

    def optimize_instance(h0, beta, max_iter, ftol=1e-10, gtol=1e-10):
        """
        Optimize using L-BFGS-B.
        """
        bounds = [(0, 1) for _ in range(N)]
        
        def obj(h):
            h_proj = project_to_feasible(h)
            return lse_objective(h_proj, beta)

        try:
            res = scipy.optimize.minimize(
                obj, 
                h0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0
        
        h_opt = project_to_feasible(h_opt)
        return h_opt

    def optimize_epigraph(h0, t0, beta, max_iter, ftol=1e-10, gtol=1e-10):
        """
        Optimize using epigraph formulation with L-BFGS-B.
        """
        bounds = [(0, 1) for _ in range(N)] + [(0, N)]
        
        def obj(x):
            h = x[:N]
            t = x[N]
            h_proj = project_to_feasible(h)
            return epigraph_objective(h_proj, t, beta)

        try:
            res = scipy.optimize.minimize(
                obj, 
                np.concatenate([h0, [t0]]), 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            h_opt = res.x[:N]
        except Exception:
            h_opt = h0
        
        h_opt = project_to_feasible(h_opt)
        return h_opt

    def score_h(h):
        """Calculate the true max overlap score."""
        c_all = get_correlations(h)
        return np.max(c_all)

    # GMM-based parameterization: h_i = F^{-1}(i/N) where F is CDF of a GMM
    def gmm_cdf(x, mu, sigma, weights):
        """Compute CDF of a Gaussian Mixture Model at points x."""
        cdf = np.zeros_like(x, dtype=float)
        for m in range(len(mu)):
            w = weights[m]
            s = sigma[m]
            if s < 1e-10:
                # Degenerate: step function
                cdf += w * (x >= mu[m]).astype(float)
            else:
                z = (x - mu[m]) / s
                # Standard normal CDF via erf
                cdf += w * 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
        return cdf

    def gmm_quantile(u, mu, sigma, weights, grid_size=10000):
        """Compute inverse CDF (quantile function) of a GMM at points u in [0,1]."""
        # Create a grid in [0,1] for u
        u_grid = np.linspace(0, 1, grid_size)
        # Create a grid in x-space for evaluating CDF
        x_min = mu.min() - 3 * sigma.max()
        x_max = mu.max() + 3 * sigma.max()
        x_grid = np.linspace(x_min, x_max, grid_size)
        # Compute CDF on x_grid
        cdf_vals = gmm_cdf(x_grid, mu, sigma, weights)
        # Interpolate to get quantile
        # Ensure cdf_vals is strictly increasing
        for i in range(1, len(cdf_vals)):
            if cdf_vals[i] <= cdf_vals[i-1]:
                cdf_vals[i] = cdf_vals[i-1] + 1e-10
        # Normalize to [0,1]
        cdf_vals = (cdf_vals - cdf_vals[0]) / (cdf_vals[-1] - cdf_vals[0])
        # For each u, find the x such that cdf(x) = u
        quantiles = np.interp(u, cdf_vals, x_grid)
        return quantiles

    def gmm_to_sequence(params, n_components):
        """Convert GMM parameters to a sequence h of length N."""
        # params: [mu_0, mu_1, ..., mu_{K-1}, sigma_0, ..., sigma_{K-1}, w_0, ..., w_{K-1}]
        K = n_components
        mu = params[:K].copy()
        sigma = params[K:2*K].copy()
        weights = params[2*K:].copy()
        
        # Ensure valid parameters
        mu = np.clip(mu, -0.5, 1.5)
        sigma = np.maximum(sigma, 0.01)
        weights = np.maximum(weights, 1e-6)
        weights = weights / weights.sum()
        
        # Generate quantiles at u_i = (i + 0.5) / N for i = 0, ..., N-1
        u = (np.arange(N) + 0.5) / N
        h = gmm_quantile(u, mu, sigma, weights, grid_size=5000)
        
        # Clip to [0,1] and project to feasible
        h = np.clip(h, 0.0, 1.0)
        h = project_to_feasible(h)
        
        return h

    def optimize_gmm(params0, n_components, beta, max_iter, ftol=1e-10, gtol=1e-10):
        """Optimize GMM parameters to minimize LSE of correlations."""
        K = n_components
        bounds = [(-0.5, 1.5)] * K + [(0.01, 2.0)] * K + [(1e-6, 1.0)] * K
        
        def obj(params):
            h = gmm_to_sequence(params, K)
            return lse_objective(h, beta)
        
        try:
            res = scipy.optimize.minimize(
                obj,
                params0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            params_opt = res.x
        except Exception:
            params_opt = params0
        
        h_opt = gmm_to_sequence(params_opt, K)
        return h_opt, params_opt

    best_h = np.full(N, 0.5)
    best_score = score_h(best_h)  # N/4 = 25.0

    rng = np.random.default_rng(42)
    
    # Multi-restart strategy
    restarts = []
    
    # 1. Uniform start
    restarts.append((np.full(N, 0.5), 5.0, 800))
    
    # 2-6. Random perturbations
    for i in range(5):
        h_rand = rng.uniform(0.1, 0.9, size=N)
        h_rand = project_to_feasible(h_rand)
        beta = 5.0 + i * 5.0
        restarts.append((h_rand, beta, 800))
        
    # 7-12. Bimodal initializations
    for i in range(6):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        h_rand = np.where(h_rand > 0.5, 0.8, 0.2)
        h_rand = project_to_feasible(h_rand)
        beta = 20.0 + i * 5.0
        restarts.append((h_rand, beta, 600))

    # 13-15. Sturmian word candidates (quasi-periodic sequences)
    for i, alpha in enumerate([0.6180339887, 0.7071067812, 0.75]):
        n_vals = np.arange(N)
        h_sturm = np.floor((n_vals + 1) * alpha) - np.floor(n_vals * alpha)
        h_sturm = h_sturm.astype(float)
        h_sturm = project_to_feasible(h_sturm)
        beta = 10.0 + i * 10.0
        restarts.append((h_sturm, beta, 800))

    # Run all restarts
    for idx, (h0, beta, max_iter) in enumerate(restarts):
        if time.time() - start_time > time_limit:
            break
            
        h_opt = optimize_instance(h0, beta, max_iter)
        current_score = score_h(h_opt)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h_opt.copy()
            
    # GMM-based optimization
    if time.time() - start_time < time_limit - 30.0:
        # Try different numbers of components
        for K in [2, 3, 4]:
            if time.time() - start_time > time_limit - 15.0:
                break
            
            # Initialize GMM parameters
            for init_seed in range(3):
                if time.time() - start_time > time_limit - 15.0:
                    break
                
                rng_gmm = np.random.default_rng(100 + init_seed * 1000 + K)
                mu0 = rng_gmm.uniform(0.0, 1.0, size=K)
                sigma0 = rng_gmm.uniform(0.05, 0.5, size=K)
                weights0 = rng_gmm.dirichlet(np.ones(K))
                
                params0 = np.concatenate([mu0, sigma0, weights0])
                
                # Optimize GMM
                h_gmm, params_gmm = optimize_gmm(params0, K, 10.0, 500, ftol=1e-10, gtol=1e-10)
                gmm_score = score_h(h_gmm)
                
                if gmm_score < best_score:
                    best_score = gmm_score
                    best_h = h_gmm.copy()
                    
                    # Polish with direct optimization
                    if time.time() - start_time < time_limit - 15.0:
                        h_polished = optimize_instance(h_gmm, 50.0, 1000, ftol=1e-12, gtol=1e-12)
                        polished_score = score_h(h_polished)
                        if polished_score < best_score:
                            best_score = polished_score
                            best_h = h_polished.copy()

    # Epigraph optimization on best solution
    if time.time() - start_time < time_limit - 25.0:
        t0 = best_score
        h_epigraph = optimize_epigraph(best_h, t0, 50.0, 1500, ftol=1e-12, gtol=1e-12)
        epigraph_score = score_h(h_epigraph)
        
        if epigraph_score < best_score:
            best_score = epigraph_score
            best_h = h_epigraph.copy()
            
    # Binary swap local search for discrete refinement
    if time.time() - start_time < time_limit - 15.0:
        h_swap = best_h.copy()
        improved = True
        max_swap_iters = 50
        for _ in range(max_swap_iters):
            if not improved:
                break
            if time.time() - start_time > time_limit - 10.0:
                break
            improved = False
            # Try random swaps
            for _ in range(1000):
                i, j = rng.integers(0, N, size=2)
                if i == j:
                    continue
                # Swap values
                h_swap[i], h_swap[j] = h_swap[j], h_swap[i]
                h_swap = project_to_feasible(h_swap)
                new_score = score_h(h_swap)
                if new_score < best_score - 1e-10:
                    best_score = new_score
                    best_h = h_swap.copy()
                    improved = True
                else:
                    # Revert
                    h_swap[i], h_swap[j] = h_swap[j], h_swap[i]
                    h_swap = project_to_feasible(h_swap)

    # High-precision polish pass on the best solution
    if time.time() - start_time < time_limit - 10.0:
        polish_beta = 100.0
        h_polished = optimize_instance(best_h, polish_beta, 2000, ftol=1e-12, gtol=1e-12)
        polished_score = score_h(h_polished)
        
        if polished_score < best_score:
            best_score = polished_score
            best_h = h_polished.copy()
            
    # Final constraint verification
    best_h = project_to_feasible(best_h)
    
    return best_h
