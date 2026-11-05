import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    def objective_and_grad(h):
        """
        Differentiable proxy for the objective.
        We use a soft-max to approximate the max of the autoconvolution.
        Objective: J = 2*N * max(c) / S^2
        Proxy: J_smooth = 2*N * C_soft / S^2
        where C_soft = logsumexp(t * c) / t (for large t, approx max(c))
        
        Note: The original objective uses abs().
        If we assume the optimal solution has S > 0 and max(c) > 0 (which is true for seed),
        we minimize 2*N*max(c)/S^2.
        If max(c) could be negative, we would want to maximize it? 
        But max(c) is at least c[0] = sum(h^2) >= 0. 
        Wait, c is the autoconvolution. c[0] = sum(h_i^2).
        So max(c) is always >= 0.
        And S^2 is always >= 0.
        So the term inside abs() is always >= 0.
        Thus we can drop the abs().
        
        Let's verify:
        c = convolve(h, h).
        c[0] = h[0]^2. No, np.convolve(h, h) full mode.
        c[0] = h[0]*h[0].
        c[N-1] = sum(h_i^2).
        So max(c) >= sum(h_i^2) >= 0.
        So J = 2*N*max(c)/S^2.
        
        We minimize J.
        
        Let u = t * c.
        C_soft = logsumexp(u) / t.
        dC_soft/dh_m = (1/t) * d/dh_m (logsumexp(t*c))
        d/dh_m (logsumexp(t*c)) = sum_j (t * c_j * exp(t*c_j) / sum_k exp(t*c_k)) * dc_j/dh_m
        Let p_j = exp(t*c_j - max(t*c)) / sum_k exp(t*c_k - max(t*c))
        Then dC_soft/dh_m = sum_j p_j * dc_j/dh_m
        
        c_j = sum_i h_i h_{i - (j-(N-1))} ?
        Let's stick to the fixed k gradient for c_j, but summed with weights p_j.
        c_j corresponds to lag L = j - (N-1).
        c_j = sum_{i} h_i h_{i+L} (with zero padding)
        dc_j/dh_m = h_{m+L} + h_{m-L} (with zero padding)
        
        So grad_C_soft[m] = sum_j p_j * (h_{m+L_j} + h_{m-L_j})
        
        This is a weighted sum of correlations.
        We can compute this efficiently?
        N=400. Number of lags is 2N-1 = 799.
        For each m, we sum over 799 terms.
        Total operations: 400 * 799 * 2 ~ 640k.
        This is fast enough in NumPy if vectorized, or even in a loop if we are careful.
        
        However, we can also just compute the gradient of the log-sum-exp directly.
        Let's implement it.
        """
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10, np.zeros_like(h)
        
        c = np.convolve(h, h)
        
        # Temperature for soft-max
        # We want to approximate max. 
        # If we use t=1, it's the mean. 
        # If t is large, it's the max.
        # Let's try t=10.
        t = 10.0
        
        c_max = np.max(c)
        u = t * (c - c_max)
        exp_u = np.exp(u)
        Z = np.sum(exp_u)
        p = exp_u / Z
        
        C_soft = c_max + np.log(Z) / t
        
        # Gradient of C_soft
        # dC_soft/dh_m = sum_j p_j * (h_{m+L_j} + h_{m-L_j})
        # L_j = j - (N-1)
        
        grad_C = np.zeros(N)
        
        # We can compute this by iterating over lags
        # For each lag L, weight is p[j] where j = L + N - 1
        # Contribution to grad_C is p[j] * (h shifted by L + h shifted by -L)
        
        for L in range(-(N-1), N):
            j = L + N - 1
            w = p[j]
            if w < 1e-10:
                continue
            
            # Term 1: h_{m+L}
            # m+L is valid index if 0 <= m+L < N
            # m in [-L, N-1-L]
            # Intersect with [0, N-1]
            
            # Term 2: h_{m-L}
            # m-L is valid if 0 <= m-L < N
            # m in [L, N-1+L]
            
            # Let's just use np.roll or slicing?
            # Slicing is faster.
            
            if L >= 0:
                # h_{m+L}: m in [0, N-1-L]
                # grad_C[0:N-L] += w * h[L:N]
                if N > L:
                    grad_C[:N-L] += w * h[L:]
                
                # h_{m-L}: m in [L, N-1]
                # grad_C[L:N] += w * h[:N-L]
                if L < N:
                    grad_C[L:] += w * h[:N-L]
            else:
                # L is negative. Let L' = -L > 0
                Lp = -L
                # h_{m-L} = h_{m+Lp}: m in [0, N-1-Lp]
                # grad_C[:N-Lp] += w * h[Lp:]
                if N > Lp:
                    grad_C[:N-Lp] += w * h[Lp:]
                
                # h_{m+L} = h_{m-Lp}: m in [Lp, N-1]
                # grad_C[Lp:] += w * h[:N-Lp]
                if Lp < N:
                    grad_C[Lp:] += w * h[:N-Lp]
        
        # Objective J = 2*N*C_soft / S^2
        # dJ/dh_m = 2*N * ( (dC_soft/dh_m) * S^2 - C_soft * 2*S ) / S^4
        #          = 2*N * ( grad_C[m] / S - 2*C_soft / S^2 )
        
        grad_J = (2.0 * N / s) * grad_C - (4.0 * N * C_soft / (s * s))
        
        val = 2.0 * N * C_soft / (s * s)
        
        return val, grad_J

    # Initial best: Linear Ramp
    h_best = np.arange(1, N + 1, dtype=np.float64)
    val_best = compute_value(h_best)
    
    # Strategy:
    # 1. Try different initializations.
    # 2. Use L-BFGS-B with the soft-max gradient.
    # 3. Try to push the max of the convolution to be flat?
    #    If c is constant, max(c) = mean(c).
    #    Sum(c) = sum_i sum_j h_i h_j = (sum h)^2 = S^2.
    #    So if c is constant, c_k = S^2 / (2N-1).
    #    Then J = 2*N * (S^2/(2N-1)) / S^2 = 2N / (2N-1) ~ 1.
    #    Seed is 2.0. So we want to get close to 1.0.
    #    Can we make c constant?
    #    c is the autoconvolution.
    #    If h is a delta function, c is a delta. Not constant.
    #    If h is uniform, c is a triangle.
    #    To make c flat, h needs to be "white noise" like?
    #    But we are minimizing max(c).
    #    If we make h random, c will have fluctuations.
    #    Maybe a specific pattern like a Barker code or Golay code?
    #    Or maybe just minimizing the variance of c?
    
    starts = [
        np.arange(1, N + 1, dtype=np.float64),
        np.ones(N, dtype=np.float64),
        np.sin(2 * np.pi * np.arange(N) / 10), # Low freq wave
        np.sin(2 * np.pi * np.arange(N) / 40), # Lower freq
        np.random.RandomState(42).randn(N), # Random
        np.random.RandomState(43).randn(N), # Random
    ]
    
    for h0 in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                objective_and_grad,
                h0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'maxls': 50, 'disp': False}
            )
            h_res = res.x
            val_res = compute_value(h_res)
            if val_res < val_best:
                val_best = val_res
                h_best = h_res.copy()
        except Exception:
            pass
            
    return h_best
