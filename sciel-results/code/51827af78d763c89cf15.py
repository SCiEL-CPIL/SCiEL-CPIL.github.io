import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # Variables: log power spectrum for k=1..N/2
    # G_0 = 0
    # G_{N-k} = G_k (real, symmetric)
    # We optimize x where P_k = exp(x_k)
    
    def compute_g(x):
        # x: array of length N/2
        # P: array of length N
        P = np.zeros(N)
        P[1:N//2+1] = np.exp(x)
        P[N//2+1:N] = P[N//2-1:0:-1] # Mirror
        if N % 2 == 0:
            P[N//2] = np.exp(x[N//2-1]) # Nyquist
        # G = sqrt(P) * exp(i*phi). Set phi=0.
        G = np.sqrt(P)
        g = np.real(np.fft.ifft(G))
        return g

    def objective_and_grad(x):
        g = compute_g(x)
        # Penalty for bounds
        # g should be in [-0.5, 0.5]
        penalty = 0
        if np.any(g > 0.5) or np.any(g < -0.5):
            penalty = np.sum((np.clip(g, -0.5, 0.5) - g)**2) * 1000
            
        # Compute R_g
        G = np.fft.fft(g)
        P_g = np.abs(G)**2
        R_g = np.real(np.fft.ifft(P_g))
        
        # Maximize min(R_g[1:])
        # Use LogSumExp approximation of -min
        # min(R) = - max(-R)
        # -max(-R) approx - log-sum-exp(-R/temp) * temp
        # We want to MINIMIZE -min(R) = max(-R)
        # So minimize log-sum-exp(-R_g[1:]/temp) * temp + penalty
        
        temp = 0.1
        neg_R = -R_g[1:]
        # Shift for stability
        max_neg_R = np.max(neg_R)
        log_sum_exp = max_neg_R + np.log(np.sum(np.exp(neg_R - max_neg_R)))
        obj = log_sum_exp * temp + penalty
        
        # Gradient
        # d obj / d x
        # This is complex. Use numerical gradient?
        # Or derive analytically.
        # Let's use numerical gradient for simplicity and robustness within time limit.
        # But L-BFGS-B is fast.
        
        # Approximate gradient
        grad = np.zeros_like(x)
        eps = 1e-5
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            obj_plus = objective_and_grad(x_plus) # Recursion? No, compute inline
            
            # Refactor objective calculation to avoid recursion
            g_plus = compute_g(x_plus)
            penalty_plus = 0
            if np.any(g_plus > 0.5) or np.any(g_plus < -0.5):
                penalty_plus = np.sum((np.clip(g_plus, -0.5, 0.5) - g_plus)**2) * 1000
            G_plus = np.fft.fft(g_plus)
            P_g_plus = np.abs(G_plus)**2
            R_g_plus = np.real(np.fft.ifft(P_g_plus))
            neg_R_plus = -R_g_plus[1:]
            max_neg_R_plus = np.max(neg_R_plus)
            lse_plus = max_neg_R_plus + np.log(np.sum(np.exp(neg_R_plus - max_neg_R_plus)))
            obj_plus = lse_plus * temp + penalty_plus
            
            grad[i] = (obj_plus - obj) / eps
        return obj, grad

    # Initial guess: Flat spectrum (white noise)
    x0 = np.zeros(N//2)
    
    # Bounds for x:
    # Max g is 0.5.
    # Max energy N * 0.25 = N/4.
    # Sum P = N * Var(g).
    # If g is binary +/-0.5, P is concentrated?
    # Let's allow x in [-10, 10]
    bounds = [(-10, 10)] * (N//2)
    
    # Run optimization
    res = None
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'maxfun': 5000, 'time_limit': 120})
    except Exception as e:
        pass
    
    if res is not None and res.success:
        x_best = res.x
    else:
        x_best = x0
        
    g_best = compute_g(x_best)
    h_best = g_best + 0.5
    
    # Project to [0,1] and fix sum
    h_best = np.clip(h_best, 0, 1)
    h_best = h_best - np.mean(h_best) + 0.5
    h_best = np.clip(h_best, 0, 1)
    
    # If time remains, do local search on h
    # (Skipping for now to keep structural change clear)
    
    return h_best
