import numpy as np
import time
import random

N = 10
# Pre-compute indices for fast evaluation
y_idx, x_idx = np.indices((N, N))
DIFF_IDX = (x_idx - y_idx) + 9  # 0..18
X_IDX = x_idx                   # 0..9
SUM_IDX = x_idx + y_idx         # 0..18
SUM2_IDX = x_idx + 2 * y_idx    # 0..27

def _compute_ratio(P):
    """
    Compute D/M ratio.
    P is a 10x10 array of non-negative weights.
    """
    total = np.sum(P)
    if total <= 1e-9:
        return 0.0
    
    # Normalize P to probabilities for entropy calculation
    # Note: Entropy is scale invariant, but we need probs for the formula
    # H = - sum (p_i) log2(p_i)
    
    # H_{-1} : distribution of x - y (19 bins)
    dist_diff = np.zeros(19)
    np.add.at(dist_diff, DIFF_IDX.ravel(), P.ravel())
    
    # H_0 : distribution of x (10 bins)
    dist_x = np.zeros(10)
    np.add.at(dist_x, X_IDX.ravel(), P.ravel())
    
    # H_1 : distribution of x + y (19 bins)
    dist_sum = np.zeros(19)
    np.add.at(dist_sum, SUM_IDX.ravel(), P.ravel())
    
    # H_2 : distribution of x + 2y (28 bins)
    dist_sum2 = np.zeros(28)
    np.add.at(dist_sum2, SUM2_IDX.ravel(), P.ravel())
    
    def entropy_from_counts(counts):
        c_total = counts.sum()
        if c_total <= 1e-15:
            return 0.0
        probs = counts / c_total
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
        
    H_diff = entropy_from_counts(dist_diff)
    H_x = entropy_from_counts(dist_x)
    H_sum = entropy_from_counts(dist_sum)
    H_sum2 = entropy_from_counts(dist_sum2)
    
    M = max(H_x, H_sum, H_sum2)
    
    if M <= 1e-9:
        return 0.0
        
    return H_diff / M

def _nelder_mead_minimize(func, x0, max_iter=500, tol=1e-10, deadline=None):
    """
    A simple implementation of Nelder-Mead optimization.
    func: function to minimize
    x0: initial point (numpy array)
    max_iter: maximum number of iterations
    tol: tolerance for convergence
    deadline: absolute time limit
    """
    n = len(x0)
    # Simplex initialization
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0.copy()
        simplex[i + 1, i] += 0.1 * (abs(x0[i]) + 1e-6)
    
    # Evaluate initial simplex
    fvals = np.array([func(v) for v in simplex])
    
    # Constants
    alpha = 1.0  # Reflection
    gamma = 2.0  # Expansion
    rho = 0.5    # Contraction
    sigma = 0.5  # Shrinkage
    
    best_x = simplex[np.argmin(fvals)]
    best_f = np.min(fvals)
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol:
            break
            
        # Centroid of all points except the worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = func(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            # Simple Reflection
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + 0.5 * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid - 0.5 * (simplex[-1] - centroid)
            fc = func(xc)
            
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrinkage
                for j in range(1, n + 1):
                    simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                    fvals[j] = func(simplex[j])
        
        # Update best
        current_min_idx = np.argmin(fvals)
        if fvals[current_min_idx] < best_f:
            best_f = fvals[current_min_idx]
            best_x = simplex[current_min_idx]
            
    return best_x, best_f

def _generate_initial_P(k):
    """
    Generate an initial P with k 'atoms' (Gaussian blobs).
    Returns log-probabilities for Nelder-Mead.
    """
    P = np.zeros((N, N))
    for _ in range(k):
        cx = random.uniform(0, N-1)
        cy = random.uniform(0, N-1)
        weight = random.uniform(0.5, 1.0)
        sigma = random.uniform(0.5, 2.0)
        
        r = int(np.ceil(3 * sigma))
        cx_min = max(0, int(cx) - r)
        cx_max = min(N, int(cx) + r + 1)
        cy_min = max(0, int(cy) - r)
        cy_max = min(N, int(cy) + r + 1)
        
        for y in range(cy_min, cy_max):
            for x in range(cx_min, cx_max):
                dist_sq = (x - cx)**2 + (y - cy)**2
                val = weight * np.exp(-dist_sq / (2 * sigma**2))
                P[y, x] += val
                
    P = np.maximum(P, 1e-12)
    P /= P.sum()
    
    # Convert to log-probabilities
    log_P = np.log(P)
    return log_P

def _log_to_P(log_P):
    """
    Convert log-probabilities to probabilities using softmax-like normalization
    to ensure sum is 1 and values are positive.
    """
    # Subtract max for numerical stability
    max_log = np.max(log_P)
    exp_log_P = np.exp(log_P - max_log)
    P = exp_log_P / np.sum(exp_log_P)
    return P

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600 - 10  # 10s safety margin
    
    best_P = None
    best_score = -np.inf
    
    # Strategy:
    # 1. Try a few specific structural heuristics
    # 2. Use Nelder-Mead on log-probabilities with different initializations
    
    # Heuristic 1: Uniform on main diagonal (x=y)
    P_diag = np.zeros((N, N))
    for i in range(N):
        P_diag[i, i] = 1.0
    P_diag /= N
    score_diag = _compute_ratio(P_diag)
    if score_diag > best_score:
        best_score = score_diag
        best_P = P_diag.copy()
        
    # Heuristic 2: Uniform on anti-diagonal (x+y=9)
    P_adiag = np.zeros((N, N))
    for i in range(N):
        P_adiag[i, N-1-i] = 1.0
        P_adiag[i, N-1-i] = 1.0
    P_adiag /= N
    score_adiag = _compute_ratio(P_adiag)
    if score_adiag > best_score:
        best_score = score_adiag
        best_P = P_adiag.copy()
        
    # Nelder-Mead Optimization
    # We will run multiple restarts with different numbers of atoms
    # and different random seeds
    
    num_restarts = 20
    atom_counts = [3, 5, 8]
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        k = atom_counts[restart % len(atom_counts)]
        
        # Generate initial log-probabilities
        log_P_init = _generate_initial_P(k)
        
        # Define the objective function for Nelder-Mead
        # We want to MAXIMIZE the ratio, so we minimize the negative ratio
        def objective(log_P_flat):
            log_P = log_P_flat.reshape((N, N))
            P = _log_to_P(log_P)
            score = _compute_ratio(P)
            return -score
        
        # Run Nelder-Mead
        # Limit iterations based on remaining time
        remaining = deadline - time.time()
        if remaining < 1:
            break
            
        # Estimate time per iteration is small, but Nelder-Mead can be slow in high dims
        # 100 dimensions is quite high for NM. 
        # However, the structure of the problem (sum of entropies) might be smooth enough.
        # To be safe, we limit iterations.
        max_iter = int(min(200, remaining * 10)) # Rough estimate: 10 iterations per second?
        if max_iter < 10:
            max_iter = 10
            
        try:
            log_P_opt, neg_score_opt = _nelder_mead_minimize(
                objective, log_P_init.ravel(), max_iter=max_iter, tol=1e-8, deadline=deadline
            )
            score_opt = -neg_score_opt
            P_opt = _log_to_P(log_P_opt.reshape((N, N)))
            
            if score_opt > best_score:
                best_score = score_opt
                best_P = P_opt.copy()
                
        except Exception as e:
            # If NM fails, try a simple random walk refinement from the initial point
            P_cand = _log_to_P(log_P_init)
            # Simple refinement
            for _ in range(50):
                if time.time() > deadline:
                    break
                mask = np.random.rand(N, N) < 0.1
                if not np.any(mask):
                    continue
                noise = np.random.normal(1.0, 0.1, size=(N, N))
                new_P = P_cand.copy()
                new_P[mask] *= noise[mask]
                new_P = np.maximum(new_P, 1e-12)
                new_P /= new_P.sum()
                score = _compute_ratio(new_P)
                if score > _compute_ratio(P_cand):
                    P_cand = new_P
            score_cand = _compute_ratio(P_cand)
            if score_cand > best_score:
                best_score = score_cand
                best_P = P_cand.copy()

    # If no valid P was found (shouldn't happen), return uniform
    if best_P is None:
        return np.ones((N, N)) / 100.0
        
    # Final validation and normalization
    best_P = np.clip(best_P, 0, None)
    total = np.sum(best_P)
    if total <= 1e-9:
        return np.ones((N, N)) / 100.0
    best_P /= total
    
    return best_P
