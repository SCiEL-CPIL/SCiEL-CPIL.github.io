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
        # Use log-sum-exp with numerical stability
        shifted = beta * (c_all - max_c)
        # Clip to avoid overflow
        shifted = np.clip(shifted, -500, 500)
        lse = max_c + np.log(np.sum(np.exp(shifted))) / beta
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
        Optimize using L-BFGS-B with adaptive beta.
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
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0
        
        h_opt = project_to_feasible(h_opt)
        return h_opt

    def optimize_slsqp(h0, beta, max_iter, ftol=1e-10, gtol=1e-10):
        """
        Optimize using SLSQP for high-precision polish.
        """
        def obj(h):
            h_proj = project_to_feasible(h)
            return lse_objective(h_proj, beta)

        def constraint_sum(h):
            return np.sum(h) - TARGET

        constraints = [{'type': 'eq', 'fun': constraint_sum}]
        bounds = [(0, 1) for _ in range(N)]
        
        try:
            res = scipy.optimize.minimize(
                obj, 
                h0, 
                method='SLSQP', 
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': ftol, 'disp': False}
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
    
    # 16-18. Three-level initializations
    for i, levels in enumerate([[0.0, 0.5, 1.0], [0.1, 0.5, 0.9], [0.0, 0.33, 0.67]]):
        h_3level = np.random.choice(levels, size=N, p=[0.25, 0.5, 0.25])
        h_3level = project_to_feasible(h_3level)
        beta = 15.0 + i * 10.0
        restarts.append((h_3level, beta, 700))
    
    # 19-21. Sine wave initializations with different frequencies
    for i, freq in enumerate([0.5, 1.0, 2.0]):
        n_vals = np.arange(N)
        h_sine = 0.5 + 0.4 * np.sin(2 * np.pi * freq * n_vals / N)
        h_sine = project_to_feasible(h_sine)
        beta = 10.0 + i * 10.0
        restarts.append((h_sine, beta, 700))
    
    # 22-24. Block pattern initializations
    for i, block_size in enumerate([5, 10, 20]):
        h_block = np.zeros(N)
        for b in range(0, N, block_size):
            end = min(b + block_size, N)
            h_block[b:end] = 0.0 if (b // block_size) % 2 == 0 else 1.0
        h_block = project_to_feasible(h_block)
        beta = 20.0 + i * 5.0
        restarts.append((h_block, beta, 600))

    # Run all restarts
    for idx, (h0, beta, max_iter) in enumerate(restarts):
        if time.time() - start_time > time_limit - 50.0:
            break
            
        h_opt = optimize_instance(h0, beta, max_iter)
        current_score = score_h(h_opt)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h_opt.copy()
            
    # Epigraph optimization on best solution
    if time.time() - start_time < time_limit - 45.0:
        t0 = best_score
        h_epigraph = optimize_epigraph(best_h, t0, 50.0, 1500, ftol=1e-12, gtol=1e-12)
        epigraph_score = score_h(h_epigraph)
        
        if epigraph_score < best_score:
            best_score = epigraph_score
            best_h = h_epigraph.copy()
            
    # Coordinate descent local search for discrete refinement
    def coordinate_descent_refine(h, max_rounds=30, time_budget=20.0):
        """
        Coordinate descent: try adjustments to each coordinate with adaptive step sizes.
        """
        h = h.copy()
        best_score_local = score_h(h)
        start_cd = time.time()
        
        for round_num in range(max_rounds):
            if time.time() - start_cd > time_budget:
                break
            improved = False
            # Adaptive step sizes: start larger, shrink over rounds
            base_step = 0.05 * (0.8 ** round_num)
            for i in range(N):
                if time.time() - start_cd > time_budget:
                    break
                orig_val = h[i]
                # Try multiple step sizes
                for delta in [-2*base_step, -base_step, base_step, 2*base_step, 
                             -0.5*base_step, 0.5*base_step]:
                    new_val = orig_val + delta
                    if new_val < 0 or new_val > 1:
                        continue
                    h[i] = new_val
                    h = project_to_feasible(h)
                    new_score = score_h(h)
                    if new_score < best_score_local - 1e-12:
                        best_score_local = new_score
                        improved = True
                    else:
                        h[i] = orig_val
                        h = project_to_feasible(h)
                if not improved:
                    h[i] = orig_val
                    h = project_to_feasible(h)
            if not improved:
                break
                
        return h, best_score_local

    if time.time() - start_time < time_limit - 40.0:
        h_cd, cd_score = coordinate_descent_refine(best_h, max_rounds=30, time_budget=20.0)
        if cd_score < best_score:
            best_score = cd_score
            best_h = h_cd.copy()

    # Pair swap local search for discrete refinement
    if time.time() - start_time < time_limit - 30.0:
        h_swap = best_h.copy()
        improved = True
        max_swap_iters = 100
        for _ in range(max_swap_iters):
            if not improved:
                break
            if time.time() - start_time > time_limit - 25.0:
                break
            improved = False
            # Try systematic pair swaps
            for i in range(N):
                if time.time() - start_time > time_limit - 25.0:
                    break
                for j in range(i+1, min(i+10, N)):  # Only nearby pairs for efficiency
                    orig_i, orig_j = h_swap[i], h_swap[j]
                    # Try swapping values
                    h_swap[i], h_swap[j] = orig_j, orig_i
                    h_swap = project_to_feasible(h_swap)
                    new_score = score_h(h_swap)
                    if new_score < best_score - 1e-10:
                        best_score = new_score
                        best_h = h_swap.copy()
                        improved = True
                    else:
                        # Revert
                        h_swap[i], h_swap[j] = orig_i, orig_j
                        h_swap = project_to_feasible(h_swap)
                    if improved:
                        break
                if improved:
                    break

    # High-precision polish pass on the best solution
    if time.time() - start_time < time_limit - 25.0:
        # Use very high beta and tight tolerances
        polish_beta = 100.0
        h_polished = optimize_instance(best_h, polish_beta, 2000, ftol=1e-12, gtol=1e-12)
        polished_score = score_h(h_polished)
        
        if polished_score < best_score:
            best_score = polished_score
            best_h = h_polished.copy()
    
    # SLSQP high-precision polish
    if time.time() - start_time < time_limit - 20.0:
        slsqp_beta = 50.0
        h_slsqp = optimize_slsqp(best_h, slsqp_beta, 500, ftol=1e-12)
        slsqp_score = score_h(h_slsqp)
        
        if slsqp_score < best_score:
            best_score = slsqp_score
            best_h = h_slsqp.copy()
            
    # Final epigraph polish
    if time.time() - start_time < time_limit - 15.0:
        t0 = best_score
        h_final_epigraph = optimize_epigraph(best_h, t0, 100.0, 1000, ftol=1e-12, gtol=1e-12)
        final_epigraph_score = score_h(h_final_epigraph)
        
        if final_epigraph_score < best_score:
            best_score = final_epigraph_score
            best_h = h_final_epigraph.copy()
    
    # Final coordinate descent polish
    if time.time() - start_time < time_limit - 10.0:
        h_cd_final, cd_final_score = coordinate_descent_refine(best_h, max_rounds=20, time_budget=8.0)
        if cd_final_score < best_score:
            best_score = cd_final_score
            best_h = h_cd_final.copy()
            
    # Final constraint verification
    best_h = project_to_feasible(best_h)
    
    return best_h
