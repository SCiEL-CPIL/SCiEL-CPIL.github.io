import numpy as np
from scipy import optimize
import time

N = 600
DELTA = (1.0/2.0) / N  # interval width = 1/1200
# Midpoints of the intervals
x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5 * DELTA

def compute_autoconv(h):
    """Compute autoconvolution of step function with heights h on N equal intervals."""
    h = np.asarray(h, dtype=np.float64)
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    autoconv = DELTA * conv
    return autoconv

def compute_score(h):
    """Compute the normalized max autoconvolution score."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return np.inf
    autoconv = compute_autoconv(h)
    max_ac = np.max(autoconv)
    integral = DELTA * np.sum(h)
    if integral <= 0:
        return np.inf
    score = max_ac * 0.5 / (integral ** 2)
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Number of Fourier modes for log(h)
    n_modes = 25
    
    # Precompute Fourier basis matrix
    # f(x) = sum_{k=1}^{n_modes} (a_k cos(2*pi*k*x) + b_k sin(2*pi*k*x))
    # We optimize the vector of coefficients theta = [a_1, b_1, ..., a_n, b_n]
    # h(x) = exp(f(x)) / integral(exp(f(x)))
    
    # Basis functions
    # x is shape (N,)
    # We want F shape (N, 2*n_modes)
    F = np.zeros((N, 2 * n_modes))
    for k in range(1, n_modes + 1):
        F[:, 2*k-2] = np.cos(2 * np.pi * k * x)
        F[:, 2*k-1] = np.sin(2 * np.pi * k * x)
        
    def get_h_from_theta(theta):
        f = F @ theta
        # Softmax-like normalization to ensure integral 1 and h >= 0
        # h_i = exp(f_i) / sum(exp(f_j)) * (1/DELTA) ? 
        # Wait, integral = DELTA * sum(h_i) = 1 => sum(h_i) = 1/DELTA.
        # Let w_i = exp(f_i) / sum(exp(f)). Then h_i = w_i / DELTA.
        # This ensures h_i >= 0 and DELTA * sum(h_i) = 1.
        max_f = np.max(f)
        exp_f = np.exp(f - max_f)
        sum_exp_f = np.sum(exp_f)
        w = exp_f / sum_exp_f
        h = w / DELTA
        return h

    def objective_and_grad(theta, beta):
        h = get_h_from_theta(theta)
        autoconv = compute_autoconv(h)
        
        # Smooth max
        max_ac = np.max(autoconv)
        exp_ac = np.exp(beta * (autoconv - max_ac))
        log_sum_exp = max_ac + np.log(np.sum(exp_ac)) / beta
        score = 0.5 * log_sum_exp
        
        # Gradient calculation
        # This is tricky. We need d(score)/d(theta).
        # score = 0.5 * LSE(beta * autoconv)
        # d(score)/d(autoconv) = 0.5 * beta * exp(beta * autoconv) / sum(exp(beta * autoconv))
        # Let p = exp(beta * (autoconv - max_ac)) / sum(exp(beta * (autoconv - max_ac)))
        # Then d(score)/d(autoconv) = 0.5 * p
        
        # autoconv = DELTA * conv
        # conv is linear in h? No, conv is quadratic in h.
        # conv_k = sum_{i,j} h_i h_j K_{ij,k}
        # d(conv_k)/d(h_m) = sum_j h_j K_{mj,k} + sum_i h_i K_{im,k}
        # By symmetry of autoconvolution, this is 2 * (h * h_shifted)_m ?
        # Actually, the gradient of the quadratic form x^T A x is 2 A x.
        # Here, autoconv is a vector of quadratic forms.
        # Let's use finite differences for the gradient of theta?
        # 2*n_modes = 50 variables.
        # 50 evaluations of objective per gradient step.
        # Each objective eval is ~0.5ms. 50 * 0.5ms = 25ms.
        # L-BFGS-B might take 100-200 iterations. 200 * 25ms = 5s.
        # This is feasible.
        
        # However, implementing analytic gradient for log-sum-exp of autoconv is complex.
        # Let's use scipy.optimize.minimize with numerical gradient (BFGS or L-BFGS-B).
        # To be safe and fast, we can implement the analytic gradient if we can.
        
        # Let's try to derive the analytic gradient.
        # S = 0.5 * (1/beta) * log( sum_k exp(beta * C_k) )
        # where C = autoconv.
        # dS/dC_k = 0.5 * exp(beta * C_k) / sum_j exp(beta * C_j) = 0.5 * p_k
        # C = DELTA * (h * h)
        # Let Q = h * h (convolution). C = DELTA * Q.
        # dS/dh_m = sum_k (dS/dC_k) * (dC_k/dh_m)
        # dC_k/dh_m = DELTA * (dQ_k/dh_m)
        # Q_k = sum_{i,j} h_i h_j delta_{i+j, k} (in index space)
        # dQ_k/dh_m = sum_j h_j delta_{m+j, k} + sum_i h_i delta_{i+m, k}
        # This is essentially 2 * (h * h) but evaluated at the shift corresponding to m?
        # No, the gradient of the convolution with respect to one input is the convolution of the other input with the kernel?
        # Actually, if Q = h * h, then dQ/dh is an operator.
        # In the frequency domain:
        # H = FFT(h)
        # Q_freq = H * H
        # C_freq = DELTA * Q_freq
        # S = 0.5 * LSE(beta * IFFT(C_freq))
        # This path is hard to differentiate directly in time domain without inverse FFTs of gradients.
        
        # Let's stick to numerical gradient for the theta optimization.
        # It's robust.
        
        return score

    def objective_fourier(theta):
        return objective_and_grad(theta, current_beta)[0]

    # Optimization strategy
    # 1. Start with theta = 0 (uniform h)
    # 2. Increase beta from 10 to 1000
    # 3. At each beta, run L-BFGS-B for a few iterations
    
    best_h = np.ones(N) / DELTA # Normalized uniform
    best_score = compute_score(best_h)
    
    # Multi-start for theta
    np.random.seed(42)
    theta_best = np.zeros(2 * n_modes)
    
    betas = [10, 50, 100, 200, 500, 1000]
    
    for beta in betas:
        if time.time() > deadline - 30:
            break
        
        current_beta = beta
        
        # Refine theta starting from previous best
        # We might need multiple restarts for early betas
        n_restarts = 3 if beta < 100 else 1
        
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
                
            if r == 0:
                x0 = theta_best.copy()
            else:
                x0 = np.random.randn(2 * n_modes) * 0.5
            
            # Use L-BFGS-B with numerical gradient
            # maxiter needs to be small to fit in time
            # 50 vars, num grad ~ 50 evals.
            # 100 iters * 50 evals * 0.5ms = 2.5s per restart.
            # 3 restarts * 6 betas = 18 * 2.5s = 45s. Safe.
            
            try:
                res = optimize.minimize(objective_fourier, x0, method='L-BFGS-B',
                                        options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5})
                if res.fun < best_score:
                    best_score = res.fun
                    best_h = get_h_from_theta(res.x).copy()
                    theta_best = res.x.copy()
            except Exception:
                pass

    # Polish phase: Direct optimization on 600 bins
    # Start from best_h
    # Use SLSQP or COBYLA? SLSQP is better for constraints.
    # But 600 vars is large for SLSQP.
    # Let's try a simple projected gradient step on the bins using the subgradient of the max.
    # Or just a few iterations of L-BFGS-B on bins with bounds.
    
    if time.time() < deadline - 15:
        h_start = best_h.copy()
        
        def objective_direct(h_vec):
            if np.any(h_vec < 0):
                return 1e10
            return compute_score(h_vec)
            
        # L-BFGS-B on 600 vars with num grad is 600 evals per step.
        # 600 * 0.5ms = 0.3s per step.
        # 20 steps = 6s.
        # This is doable.
        
        try:
            res_direct = optimize.minimize(objective_direct, h_start, method='L-BFGS-B',
                                           bounds=[(0, None)] * N,
                                           options={'maxiter': 20, 'ftol': 1e-8, 'gtol': 1e-4})
            if res_direct.fun < best_score:
                best_score = res_direct.fun
                best_h = res_direct.x.copy()
        except Exception:
            pass

    return best_h.tolist()
