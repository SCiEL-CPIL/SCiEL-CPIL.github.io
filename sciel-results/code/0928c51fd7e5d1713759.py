import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def shannon_entropy(p):
    """Calculate Shannon entropy in bits for a probability distribution p."""
    p = np.asarray(p)
    p = p[p > 1e-15]  # Ignore near-zero probabilities
    if len(p) == 0:
        return 0.0
    return -np.sum(p * np.log2(p))

def compute_score(P):
    """
    Compute the ratio D / M for a given 10x10 distribution P.
    P[y, x] corresponds to probability of (X=x, Y=y).
    """
    n = 10
    # Flatten P to get probabilities for each (x, y) pair
    # P is indexed P[y, x]
    # We need marginals/projections.
    
    # D = H(X - Y)
    # Values of X - Y range from -9 to 9.
    diff_vals = np.arange(-9, 10)
    p_diff = np.zeros(19)
    for y in range(n):
        for x in range(n):
            val = x - y
            idx = val + 9
            p_diff[idx] += P[y, x]
    H_diff = shannon_entropy(p_diff)
    
    # M = max(H(X), H(X+Y), H(X+2Y))
    
    # H(X): X ranges 0..9
    p_x = np.zeros(10)
    for y in range(n):
        for x in range(n):
            p_x[x] += P[y, x]
    H_x = shannon_entropy(p_x)
    
    # H(X+Y): X+Y ranges 0..18
    p_sum = np.zeros(19)
    for y in range(n):
        for x in range(n):
            val = x + y
            p_sum[val] += P[y, x]
    H_sum = shannon_entropy(p_sum)
    
    # H(X+2Y): X+2Y ranges 0..27
    p_2sum = np.zeros(28)
    for y in range(n):
        for x in range(n):
            val = x + 2 * y
            p_2sum[val] += P[y, x]
    H_2sum = shannon_entropy(p_2sum)
    
    M = max(H_x, H_sum, H_2sum)
    
    if M <= 1e-9:
        return 0.0 # Invalid or undefined, treat as bad
        
    return H_diff / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 550.0 # Safety margin
    
    # Initial best
    P_best = np.ones((n, n)) / (n * n)
    best_score = compute_score(P_best)
    
    # Helper to convert unconstrained vector u to probability matrix P
    def u_to_P(u):
        u = u.reshape(n, n)
        # Softmax to ensure positive and sum to 1
        # Subtract max for numerical stability
        u_max = np.max(u)
        exp_u = np.exp(u - u_max)
        P = exp_u / np.sum(exp_u)
        return P

    # Objective function for minimizers (we want to maximize score, so minimize -score)
    def objective(u):
        P = u_to_P(u)
        score = compute_score(P)
        return -score

    # Generate diverse starting points
    starts = []
    
    # 1. Uniform
    starts.append(np.zeros((n, n)))
    
    # 2. Random Dirichlet
    for _ in range(20):
        alpha = np.random.dirichlet(np.ones(n*n))
        # Convert to log-space for u_to_P? 
        # u_to_P uses softmax. If we want P ~ Dirichlet, we can just use log(alpha) as u?
        # No, softmax(u) != alpha. 
        # We want to start near specific P. 
        # We can invert softmax: u = log(P) + constant. 
        # Since softmax is invariant to shift, u = log(P) works.
        P_init = alpha.reshape(n, n)
        # Avoid log(0)
        P_init = np.maximum(P_init, 1e-10)
        P_init /= np.sum(P_init)
        u_init = np.log(P_init).flatten()
        starts.append(u_init)
        
    # 3. Structured patterns
    # Pattern: High on diagonal x=y
    P_diag = np.zeros((n, n))
    for i in range(n): P_diag[i, i] = 1.0
    P_diag /= np.sum(P_diag)
    starts.append(np.log(np.maximum(P_diag, 1e-10)).flatten())
    
    # Pattern: High on anti-diagonal x+y=9
    P_anti = np.zeros((n, n))
    for i in range(n): P_anti[i, 9-i] = 1.0
    P_anti /= np.sum(P_anti)
    starts.append(np.log(np.maximum(P_anti, 1e-10)).flatten())
    
    # Pattern: Cross (x=0 or y=0)
    P_cross = np.zeros((n, n))
    P_cross[0, :] = 1.0
    P_cross[:, 0] = 1.0
    P_cross[0, 0] = 1.0 # avoid double counting if we sum, but for init it's fine
    P_cross /= np.sum(P_cross)
    starts.append(np.log(np.maximum(P_cross, 1e-10)).flatten())
    
    # Pattern: Random "blob"
    for _ in range(10):
        cx, cy = np.random.rand(2) * (n-1)
        X, Y = np.meshgrid(np.arange(n), np.arange(n))
        dist_sq = (X - cx)**2 + (Y - cy)**2
        P_blob = np.exp(-dist_sq / 2.0)
        P_blob /= np.sum(P_blob)
        starts.append(np.log(np.maximum(P_blob, 1e-10)).flatten())

    # Run Nelder-Mead from each start
    for i, u_start in enumerate(starts):
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(objective, u_start, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success or res.fun < -best_score:
                P_curr = u_to_P(res.x)
                score_curr = -res.fun
                if score_curr > best_score:
                    best_score = score_curr
                    P_best = P_curr
        except Exception:
            pass

    # If we have time, try Differential Evolution on the best found so far or globally
    # DE is expensive but robust. Let's try a short run.
    if time.time() < deadline - 60:
        try:
            # Bounds for u: [-10, 10] is usually enough for softmax
            bounds = [(-10, 10)] * (n * n)
            # We can use the best P found as a hint? DE doesn't take hints directly, but we can seed the population?
            # scipy DE doesn't support custom seeding easily in older versions, but we can just run it.
            # To save time, we might restrict the search space or iterations.
            # Let's run DE for a limited number of generations.
            res_de = differential_evolution(objective, bounds, maxiter=20, popsize=15, 
                                            tol=1e-6, polish=True)
            if res_de.fun < -best_score:
                P_de = u_to_P(res_de.x)
                score_de = -res_de.fun
                if score_de > best_score:
                    best_score = score_de
                    P_best = P_de
        except Exception:
            pass

    # Final refinement of P_best using Powell or L-BFGS-B on the log-space
    if time.time() < deadline - 10 and best_score > 0:
        try:
            # Start from best found
            u_best = np.log(np.maximum(P_best, 1e-10)).flatten()
            # Use Powell for robustness
            res_powell = minimize(objective, u_best, method='Powell', 
                                  options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-10})
            if res_powell.fun < -best_score:
                P_powell = u_to_P(res_powell.x)
                score_powell = -res_powell.fun
                if score_powell > best_score:
                    best_score = score_powell
                    P_best = P_powell
        except Exception:
            pass

    # Ensure P_best is valid
    P_best = np.clip(P_best, 0, None)
    total = np.sum(P_best)
    if total <= 1e-9:
        P_best = np.ones((n, n)) / (n * n)
    else:
        P_best /= total
        
    return P_best
