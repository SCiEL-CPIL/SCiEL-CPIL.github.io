import numpy as np
import time

N = 100

def compute_S(h):
    # h is array of length N
    S = np.zeros(N-1)
    for k in range(1, N):
        # sum h[i] * (1-h[i+k]) for i=0 to N-k-1
        S[k-1] = np.sum(h[:N-k] * (1 - h[k:]))
    return S

def compute_grad_S(h):
    # Returns array of gradients for each k
    # grad[k] is array of length N
    # We only need the weighted sum for the objective gradient
    pass

def objective(h, beta):
    S = compute_S(h)
    # log-sum-exp
    S_max = np.max(S)
    log_sum_exp = S_max + np.log(np.sum(np.exp(beta * (S - S_max))))
    return log_sum_exp / beta

def gradient(h, beta):
    S = compute_S(h)
    S_max = np.max(S)
    exp_S = np.exp(beta * (S - S_max))
    sum_exp = np.sum(exp_S)
    
    grad_total = np.zeros(N)
    for k in range(1, N):
        # grad of S_k
        g = np.zeros(N)
        # term 1: i < N-k, deriv (1-h[i+k])
        g[:N-k] += (1 - h[k:])
        # term 2: i >= k, deriv -h[i-k]
        g[k:] -= h[:N-k]
        
        grad_total += (exp_S[k-1] / sum_exp) * g
        
    return grad_total

def project(h):
    # Project to sum=50, 0<=h<=1
    # Binary search on tau
    lo = -1.0
    hi = 2.0
    target = 50.0
    for _ in range(50):
        tau = (lo + hi) / 2
        h_proj = np.clip(h - tau, 0, 1)
        s = np.sum(h_proj)
        if s > target:
            lo = tau
        else:
            hi = tau
    return np.clip(h - (lo+hi)/2, 0, 1)

def construct():
    start_time = time.time()
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_val = objective(h, 100)
    
    # Gradient Descent
    betas = [1, 5, 20, 100, 500]
    for beta in betas:
        if time.time() - start_time > 140:
            break
        lr = 0.01
        # Restart GD for this beta
        h_curr = h.copy()
        for i in range(2000):
            if time.time() - start_time > 140:
                break
            g = gradient(h_curr, beta)
            h_next = h_curr - lr * g
            h_next = project(h_next)
            h_curr = h_next
            if i % 100 == 0:
                val = objective(h_curr, beta)
                if val < best_val:
                    best_val = val
                    best_h = h_curr.copy()
        h = h_curr
    
    return best_h
