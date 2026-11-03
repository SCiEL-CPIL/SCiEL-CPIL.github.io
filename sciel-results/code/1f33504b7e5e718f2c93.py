import numpy as np
import time
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    dx = 1.0 / (2 * N)  # interval width: [-1/4, 1/4] has length 1/2, N intervals

    def autoconv_max_ratio(h):
        """Compute max(autoconv) / ||h||_2^2 for step function h on N equal intervals."""
        h = np.asarray(h, dtype=np.float64)
        if np.any(h < 0):
            h = np.maximum(h, 0)
        l2_sq = np.sum(h**2) * dx
        if l2_sq < 1e-15:
            return float('inf')
        
        # Use FFT for efficient autoconvolution
        n_fft = 2 * N
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        
        H = np.fft.fft(h_padded)
        autoconv_circ = np.fft.ifft(H * H).real
        
        # The linear autoconvolution has support from -(N-1)*dx to (N-1)*dx
        # With circular conv of size 2N, indices 0..N-1 correspond to lags 0..(N-1)
        # and indices N..2N-1 correspond to lags -(N-1)..-1
        # By symmetry, we can just take max over all 2N-1 valid entries
        
        autoconv_vals = autoconv_circ[:2*N-1] * dx**2
        max_autoconv = np.max(autoconv_vals)
        
        return max_autoconv / l2_sq

    def logsumexp_autoconv_ratio(params, num_harmonics, temperature=10.0):
        """
        Smooth surrogate for autoconv_max_ratio using Log-Sum-Exp on the autoconvolution.
        params: [log_amp_0, log_amp_1, ..., log_amp_M, phase_1, ..., phase_M]
        """
        # Reconstruct h from Fourier params
        log_amps = params[:num_harmonics + 1]
        phases = params[num_harmonics + 1:]
        
        # Compute h in spatial domain
        # h(x) = sum_{m=0}^M exp(log_amp_m) * cos(4*pi*m*x + phase_m)
        # Note: We use cos for m=0 as well (phase 0) to keep it real and positive if possible,
        # but actually for m=0, cos(0)=1, so it's just exp(log_amp_0).
        
        # To ensure h >= 0, we might need to be careful. 
        # However, the optimization will drive it. 
        # A safer bet is to optimize h directly? No, Fourier is the structural change.
        # Let's stick to Fourier but ensure positivity by taking max(0, h) in the objective?
        # That breaks differentiability. 
        # Better: Constrain log_amps to be large enough or use a softplus?
        # Let's just compute h, and if it goes negative, the L2 norm and autoconv might be weird.
        # Standard trick: h = exp(sum(...))? No, that's not linear superposition.
        
        # Let's use the linear superposition: h = sum a_m cos(w_m x + p_m)
        # We want h >= 0.
        # If we optimize log_amps, they can be any real.
        # If h becomes negative, we clip it in the objective calculation for the ratio,
        # but the gradient will be wrong.
        
        # Alternative: Optimize h directly but use a smooth max surrogate?
        # The directive says "different representation". Fourier is a good one.
        # Let's assume the optimal h is positive and the optimizer stays in the positive region
        # or we use a penalty.
        
        # Let's try a simpler structural change: 
        # Parametrize h by its log-values in a low-dimensional basis?
        # Or just use L-BFGS on h directly with a smooth max surrogate?
        # The current code uses coordinate descent on h.
        # L-BFGS on h with LogSumExp(max) is a strong candidate.
        
        # Let's implement L-BFGS on h directly with a smooth objective.
        # This avoids the non-negativity constraint issues of Fourier superposition.
        pass

    def smooth_objective(h, temperature=10.0):
        """
        Compute a smooth approximation of max(autoconv) / ||h||_2^2.
        h is a vector of length N.
        """
        h = np.maximum(h, 0) # Enforce non-negativity softly? No, hard clip for now, 
                             # but L-BFGS might struggle if it hits 0.
                             # Actually, if we bound h >= 0 in minimize, it's fine.
        
        l2_sq = np.sum(h**2) * dx
        if l2_sq < 1e-15:
            return 1e10 # Large penalty
        
        n_fft = 2 * N
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        
        H = np.fft.fft(h_padded)
        autoconv_circ = np.fft.ifft(H * H).real
        
        autoconv_vals = autoconv_circ[:2*N-1] * dx**2
        
        # Log-Sum-Exp approximation of max
        # max(x) approx 1/T * log(sum(exp(T*x)))
        # We want to minimize max(autoconv) / l2_sq
        # Let y = autoconv_vals / l2_sq
        # We want min max(y)
        # Surrogate: (1/T) * log(sum(exp(T*y)))
        
        # Scale to avoid overflow in exp
        # y values are roughly O(1) or smaller. T=10 is safe.
        max_y = np.max(autoconv_vals)
        # If max_y is very small, T*max_y is small.
        # Let's use a stable logsumexp implementation
        shifted = autoconv_vals - max_y
        log_sum_exp = max_y + np.log(np.sum(np.exp(temperature * shifted)))
        
        # The surrogate is log_sum_exp / temperature / l2_sq ?
        # Wait, max(y) = max(autoconv_vals) / l2_sq
        # logsumexp(T * autoconv_vals) = T * max(autoconv_vals) + log(sum(exp(T*(autoconv_vals - max))))
        # So max(autoconv_vals) approx (1/T) * logsumexp(T * autoconv_vals)
        
        max_autoconv_approx = (1.0 / temperature) * log_sum_exp
        ratio = max_autoconv_approx / l2_sq
        
        return ratio

    def coordinate_descent_refine(h, max_iters=100, step_decay=0.9):
        """Refine h using coordinate descent to minimize autoconv_max_ratio."""
        h = np.asarray(h, dtype=np.float64).copy()
        current_score = autoconv_max_ratio(h)
        step_size = 0.05
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Randomize order to avoid bias
            indices = np.random.permutation(N)
            for i in indices:
                if time.time() > deadline:
                    break
                # Try increasing and decreasing h[i]
                for direction in [1, -1]:
                    h_trial = h.copy()
                    h_trial[i] = max(0, h[i] + direction * step_size)
                    trial_score = autoconv_max_ratio(h_trial)
                    if trial_score < current_score:
                        h = h_trial
                        current_score = trial_score
                        improved = True
                        break
            if not improved:
                step_size *= step_decay
                if step_size < 1e-6:
                    break
            else:
                # Decay step size slightly if we improved a lot? Or keep it?
                # Let's decay slowly to allow fine tuning
                step_size *= 0.98
        
        return h, current_score

    best_score = float('inf')
    best_h = None

    # Initialize with a Gaussian
    centers = np.arange(N) - N // 2
    h_init = np.exp(-centers**2 / (2 * 50**2))
    
    # Run L-BFGS-B on h directly with the smooth objective
    # Bounds: h >= 0. Upper bound? Let's say 10.
    bounds = [(0, 10.0)] * N
    
    try:
        res = minimize(
            smooth_objective, 
            h_init, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        h_opt = res.x
        # Evaluate true score
        score_opt = autoconv_max_ratio(h_opt)
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
    except Exception as e:
        print(f"L-BFGS-B failed: {e}")
        best_h = h_init.copy()
        best_score = autoconv_max_ratio(best_h)

    # Polish with coordinate descent
    if time.time() < deadline - 10:
        refined_h, refined_score = coordinate_descent_refine(best_h, max_iters=200, step_decay=0.9)
        if refined_score < best_score:
            best_score = refined_score
            best_h = refined_h

    # Try a few other initializations if time permits
    if time.time() < deadline - 30:
        inits = [
            np.exp(-centers**2 / (2 * 10**2)),
            np.exp(-centers**2 / (2 * 100**2)),
            np.ones(N) * 0.5
        ]
        for h_i in inits:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    smooth_objective, 
                    h_i, 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                h_opt = res.x
                score_opt = autoconv_max_ratio(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
                
                # Quick polish
                if time.time() < deadline - 5:
                    refined_h, refined_score = coordinate_descent_refine(best_h, max_iters=50, step_decay=0.9)
                    if refined_score < best_score:
                        best_score = refined_score
                        best_h = refined_h
            except:
                pass

    if best_h is None:
        return [1.0] * N

    return list(best_h)

# EVOLVE-BLOCK-END
