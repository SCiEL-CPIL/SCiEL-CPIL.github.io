import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 600
DEADLINE = 148

def construct():
    start_time = time.time()

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h (assumed sum=1)."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return np.max(conv)

    def smooth_max_obj_grad(h, beta):
        """Smooth max autoconvolution with gradient, using temperature beta."""
        h = np.maximum(h, 1e-10)
        h = h / np.sum(h)

        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real

        m = np.max(conv)
        exp_conv = np.exp((conv - m) * beta)
        weights = exp_conv / np.sum(exp_conv)

        w_full = np.zeros(size)
        w_full[:len(conv)] = weights
        W = np.fft.fft(w_full, n=size)
        grad_h = 2.0 * np.fft.ifft(H * W).real[:N]

        val = m + np.log(np.sum(exp_conv)) / beta
        return val, grad_h

    i = np.arange(N, dtype=float)
    center = (N - 1) / 2.0

    # --- GMM (Gaussian Mixture Model) parameterization ---
    K = 4

    def gmm_to_h(params):
        """Convert GMM parameters to discrete heights on N intervals."""
        h = np.zeros(N)
        for k in range(K):
            w_k = max(params[3*k], 0.01)
            mu_k = params[3*k+1]
            sigma_k = max(params[3*k+2], 1.0)
            h += w_k * np.exp(-((i - mu_k)**2) / (2 * sigma_k**2))
        h = np.maximum(h, 1e-10)
        h /= np.sum(h)
        return h

    def gmm_objective(params):
        """Objective: max autoconvolution of GMM-generated h."""
        h = gmm_to_h(params)
        return compute_autoconv_max(h)

    # DE bounds for GMM parameters
    bounds = []
    for k in range(K):
        bounds.append((0.1, 5.0))
        bounds.append((0, N-1))
        bounds.append((5.0, N/3.0))

    # Initial best: uniform
    best_h = np.ones(N) / N
    best_val = compute_autoconv_max(best_h)

    # Stage 1: Differential Evolution on GMM parameters
    if time_left() > 60:
        try:
            de_result = differential_evolution(
                gmm_objective,
                bounds=bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='deferred',
                callback=None
            )
            if de_result.fun < best_val:
                h_de = gmm_to_h(de_result.x)
                best_val = de_result.fun
                best_h = h_de.copy()
        except Exception:
            pass

    # Stage 2: L-BFGS-B polish on discrete heights with annealing temperature
    if time_left() > 30:
        betas = [10.0, 50.0, 200.0, 500.0]
        
        try:
            current_h = best_h.copy()
            current_val = best_val
            
            for beta in betas:
                if time_left() < 5:
                    break
                
                result = minimize(
                    lambda h: smooth_max_obj_grad(h, beta)[0],
                    current_h,
                    jac=lambda h: smooth_max_obj_grad(h, beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 200}
                )
                
                if result.success:
                    final_h = np.maximum(result.x, 0)
                    if np.sum(final_h) > 0:
                        final_h = final_h / np.sum(final_h)
                        final_val = compute_autoconv_max(final_h)
                        if final_val < current_val:
                            current_val = final_val
                            current_h = final_h.copy()
            
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        except Exception:
            pass

    # Stage 3: Multi-restart with GMM perturbations + Annealed L-BFGS-B polish
    if time_left() > 20:
        n_restarts = int(min(4, time_left() // 5))
        for r in range(n_restarts):
            if time_left() < 8:
                break

            base_params = np.zeros(3*K)
            for k in range(K):
                base_params[3*k] = 1.0
                if k < K//2:
                    base_params[3*k+1] = N * (0.15 + 0.1*k)
                else:
                    base_params[3*k+1] = N * (0.85 - 0.1*(k-K//2))
                base_params[3*k+2] = N/8

            perturbation = np.random.normal(0, 0.3, size=3*K)
            params_perturbed = base_params + perturbation
            for k in range(K):
                params_perturbed[3*k] = max(0.1, min(5.0, params_perturbed[3*k]))
                params_perturbed[3*k+1] = max(0, min(N-1, params_perturbed[3*k+1]))
                params_perturbed[3*k+2] = max(5.0, min(N/3.0, params_perturbed[3*k+2]))

            h_init = gmm_to_h(params_perturbed)

            try:
                betas_restart = [50.0, 200.0]
                current_h = h_init
                current_val = compute_autoconv_max(current_h)
                
                for beta in betas_restart:
                    if time_left() < 4:
                        break
                    result = minimize(
                        lambda h: smooth_max_obj_grad(h, beta)[0],
                        current_h,
                        jac=lambda h: smooth_max_obj_grad(h, beta)[1],
                        method='L-BFGS-B',
                        options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 100}
                    )
                    if result.success:
                        final_h = np.maximum(result.x, 0)
                        if np.sum(final_h) > 0:
                            final_h = final_h / np.sum(final_h)
                            final_val = compute_autoconv_max(final_h)
                            if final_val < current_val:
                                current_val = final_val
                                current_h = final_h.copy()

                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            except Exception:
                pass

    # Stage 4: Coordinate descent fine-tuning
    if time_left() > 10:
        current_h = best_h.copy()
        current_val = best_val
        max_rounds = int(min(5, time_left() // 2))

        for round_num in range(max_rounds):
            if time_left() < 4:
                break

            improved = False
            for _ in range(10):
                if time_left() < 3:
                    break
                idx = np.random.choice(N, size=10, replace=False)
                h_trial = current_h.copy()
                for j in idx:
                    h_trial[j] *= (1.0 + np.random.uniform(-0.1, 0.1))
                h_trial = np.maximum(h_trial, 0)
                if np.sum(h_trial) > 0:
                    h_trial = h_trial / np.sum(h_trial)
                    val_trial = compute_autoconv_max(h_trial)
                    if val_trial < current_val:
                        current_val = val_trial
                        current_h = h_trial.copy()
                        improved = True

            if not improved:
                break

        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()

    # Final normalization and return
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) > 0:
        best_h = best_h / np.sum(best_h)
    else:
        best_h = np.ones(N) / N

    return [float(h) for h in best_h]
