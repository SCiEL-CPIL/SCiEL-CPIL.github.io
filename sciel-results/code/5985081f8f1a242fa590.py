"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def compute_autocorr(h):
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        R = np.zeros(N)
        R[0] = acf[0]
        R[1:N] = acf[1:N]
        return R

    def get_max_autocorr(R):
        return np.max(np.abs(R[1:N]))

    def hill_climb(h, R, current_val, max_passes=300, time_limit=None):
        """Greedy local search with incremental O(N) updates."""
        if time_limit is None:
            time_limit = deadline - 5.0
        h = h.copy()
        R = R.copy()
        k = np.arange(1, N)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > time_limit:
                break
            improved = False
            passes += 1
            order = np.random.permutation(N)
            for i in order:
                if time.time() > time_limit:
                    break
                # Delta for flipping h[i]
                idx1 = i + k
                valid1 = idx1 < N
                term1 = np.zeros(N-1)
                term1[valid1] = h[idx1[valid1]]
                idx2 = i - k
                valid2 = idx2 >= 0
                term2 = np.zeros(N-1)
                term2[valid2] = h[idx2[valid2]]
                delta = -2.0 * h[i] * (term1 + term2)
                R[1:N] += delta
                new_val = get_max_autocorr(R)
                if new_val < current_val:
                    h[i] *= -1.0
                    current_val = new_val
                    improved = True
                else:
                    R[1:N] -= delta
        return h, R, current_val

    def soft_max_sidelobe(x, beta=10.0):
        """
        Smooth approximation of max |C_k(x)|.
        x is the continuous variable in [-1, 1].
        """
        h = x
        R = compute_autocorr(h)
        sidelobes = np.abs(R[1:N])
        max_val = np.max(sidelobes)
        if max_val < 1e-10:
            return max_val, np.zeros(N)
        
        # Log-sum-exp for stability
        shifted = beta * (sidelobes - max_val)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        obj = max_val + np.log(sum_exp) / beta
        
        # Gradient calculation
        # d(obj)/dx_i = sum_k w_k * sign(C_k) * (h[i+k] + h[i-k])
        # w_k = exp(beta * |C_k|) / sum_exp
        # Note: sign(C_k) * (h[i+k] + h[i-k]) is the derivative of C_k w.r.t x_i multiplied by sign
        # Actually, d|C_k|/dx_i = sign(C_k) * dC_k/dx_i
        # dC_k/dx_i = h[i+k] + h[i-k] (if valid)
        
        weights = exp_shifted / sum_exp
        grad = np.zeros(N)
        
        # Efficient gradient computation
        # grad = sum_k w_k * sign(C_k) * (shifted_h_plus + shifted_h_minus)
        # We can do this with loops over k or vectorized operations.
        # Since N=400, a loop over k (399 iterations) with vectorized inner ops is fast enough for a few L-BFGS steps.
        
        C_k = R[1:N]
        signs = np.sign(C_k)
        
        # w_k * sign(C_k)
        weighted_signs = weights * signs
        
        for k_idx in range(N - 1):
            wk = weighted_signs[k_idx]
            if abs(wk) < 1e-12:
                continue
            lag = k_idx + 1
            # Contribution to grad[i] from term h[i+lag]
            # grad[0:N-lag] += wk * h[lag:N]
            # Contribution to grad[i] from term h[i-lag]
            # grad[lag:N] += wk * h[0:N-lag]
            grad[:N-lag] += wk * h[lag:]
            grad[lag:] += wk * h[:N-lag]
            
        return obj, grad

    def continuous_optimize(x0, time_limit):
        """Run L-BFGS-B to minimize soft max sidelobe."""
        try:
            res = minimize(soft_max_sidelobe, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 50, 'ftol': 1e-4, 'gtol': 1e-3, 'maxfun': 200})
            return res.x
        except Exception:
            return x0

    rng = np.random.RandomState(42)
    best_h = None
    best_R = None
    best_val = float('inf')

    # Phase 1: Continuous Relaxation + Discrete Refinement
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Initialize with random binary values scaled to [-1, 1]
        x0 = rng.choice([-1.0, 1.0], N) * 0.8
        
        # Run continuous optimization
        x_opt = continuous_optimize(x0, deadline - 20.0)
        
        # Round to binary
        h_bin = np.sign(x_opt)
        h_bin[h_bin == 0] = 1.0  # Arbitrary choice for zeros
        
        # Evaluate
        R_bin = compute_autocorr(h_bin)
        val_bin = get_max_autocorr(R_bin)
        
        # Refine with discrete hill climbing
        h_ref, R_ref, val_ref = hill_climb(h_bin, R_bin, val_bin, max_passes=200, time_limit=deadline - 10.0)
        
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref.copy()
            best_R = R_ref.copy()

    # Phase 2: Pure Discrete Random Restarts (if time permits)
    if time.time() < deadline - 20.0:
        for r in range(5):
            if time.time() > deadline - 10.0:
                break
            h = rng.choice([1, -1], N).astype(np.float64)
            R = compute_autocorr(h)
            val = get_max_autocorr(R)
            h, R, val = hill_climb(h, R, val, max_passes=100, time_limit=deadline - 5.0)
            if val < best_val:
                best_val = val
                best_h = h.copy()
                best_R = R.copy()

    if best_h is None:
        best_h = rng.choice([1, -1], N).astype(np.float64)

    return best_h
# EVOLVE-BLOCK-END
