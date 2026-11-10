import numpy as np
import time

N = 10
y_idx, x_idx = np.indices((N, N))
DIFF_IDX = (x_idx - y_idx) + 9  # 0..18
X_IDX = x_idx                   # 0..9
SUM_IDX = x_idx + y_idx         # 0..18
SUM2_IDX = x_idx + 2 * y_idx    # 0..27

def _entropy_from_counts(counts):
    c_total = counts.sum()
    if c_total <= 1e-15:
        return 0.0
    probs = counts / c_total
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _compute_ratio(P):
    total = np.sum(P)
    if total <= 1e-9:
        return 0.0
    
    dist_diff = np.zeros(19)
    np.add.at(dist_diff, DIFF_IDX.ravel(), P.ravel())
    
    dist_x = np.zeros(10)
    np.add.at(dist_x, X_IDX.ravel(), P.ravel())
    
    dist_sum = np.zeros(19)
    np.add.at(dist_sum, SUM_IDX.ravel(), P.ravel())
    
    dist_sum2 = np.zeros(28)
    np.add.at(dist_sum2, SUM2_IDX.ravel(), P.ravel())
    
    H_diff = _entropy_from_counts(dist_diff)
    H_x = _entropy_from_counts(dist_x)
    H_sum = _entropy_from_counts(dist_sum)
    H_sum2 = _entropy_from_counts(dist_sum2)
    
    M = max(H_x, H_sum, H_sum2)
    
    if M <= 1e-9:
        return 0.0
        
    return H_diff / M

def _generate_structural_P(type_id):
    """Generate structured initial distributions for better starting points."""
    P = np.zeros((N, N))
    
    if type_id == 0:
        # Uniform on main diagonal
        for i in range(N):
            P[i, i] = 1.0
    elif type_id == 1:
        # Uniform on anti-diagonal
        for i in range(N):
            P[i, N-1-i] = 1.0
    elif type_id == 2:
        # Corner-heavy: weight corners more
        for y in range(N):
            for x in range(N):
                dist_from_center = (x - 4.5)**2 + (y - 4.5)**2
                P[y, x] = np.exp(-dist_from_center / 5.0)
    elif type_id == 3:
        # Edge-heavy: weight edges more
        for y in range(N):
            for x in range(N):
                edge_dist = min(x, N-1-x, y, N-1-y)
                P[y, x] = 1.0 / (1.0 + edge_dist)
    elif type_id == 4:
        # Sparse random points
        n_points = np.random.randint(3, 8)
        points = np.random.randint(0, N, size=(n_points, 2))
        for px, py in points:
            P[py, px] = 1.0
    elif type_id == 5:
        # Two opposing corners
        P[0, 0] = 1.0
        P[N-1, N-1] = 1.0
    elif type_id == 6:
        # Other two opposing corners
        P[0, N-1] = 1.0
        P[N-1, 0] = 1.0
    elif type_id == 7:
        # Middle cross
        mid = N // 2
        P[mid, :] = 1.0
        P[:, mid] = 1.0
    elif type_id == 8:
        # Concentric rings
        for y in range(N):
            for x in range(N):
                dist = np.sqrt((x - 4.5)**2 + (y - 4.5)**2)
                P[y, x] = np.cos(dist * 0.5) if dist < 4 else 0
                P[y, x] = max(0, P[y, x])
    else:
        # Fallback: slightly random
        P = np.random.dirichlet(np.ones(N*N)).reshape(N, N)
    
    total = np.sum(P)
    if total <= 1e-9:
        P = np.ones((N, N)) / 100.0
    else:
        P /= total
    
    return P

def _nelder_mead_minimize(func, x0, max_iter=1000, tol=1e-12, deadline=None):
    """
    Nelder-Mead optimization to minimize func.
    func: function to minimize
    x0: initial point (numpy array)
    max_iter: maximum number of iterations
    tol: tolerance for convergence
    deadline: absolute time limit
    """
    n = len(x0)
    
    # Initialize simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(1, n + 1):
        simplex[i] = x0.copy()
        simplex[i, i-1] += 0.1 if x0[i-1] != 0 else 0.1
    
    # Evaluate initial simplex
    fvals = np.array([func(simplex[i]) for i in range(n + 1)])
    
    best_idx = np.argmin(fvals)
    best_val = fvals[best_idx]
    best_x = simplex[best_idx].copy()
    
    alpha = 1.0  # Reflection
    gamma = 2.0  # Expansion
    rho = 0.5    # Contraction
    sigma = 0.5  # Shrink
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        
        # Check convergence
        if np.max(fvals) - np.min(fvals) < tol:
            break
        
        # Current worst point
        x_worst = simplex[n]
        f_worst = fvals[n]
        
        # Centroid of all points except worst
        x_centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_reflect = x_centroid + alpha * (x_centroid - x_worst)
        f_reflect = func(x_reflect)
        
        if f_reflect < fvals[0]:  # Better than best
            # Expansion
            x_expand = x_centroid + gamma * (x_reflect - x_centroid)
            f_expand = func(x_expand)
            
            if f_expand < f_reflect:
                simplex[n] = x_expand
                fvals[n] = f_expand
            else:
                simplex[n] = x_reflect
                fvals[n] = f_reflect
        elif f_reflect < fvals[n-1]:  # Better than second worst
            simplex[n] = x_reflect
            fvals[n] = f_reflect
        else:  # Worse than worst
            # Contraction
            x_contract = x_centroid + rho * (x_worst - x_centroid)
            f_contract = func(x_contract)
            
            if f_contract < f_worst:
                simplex[n] = x_contract
                fvals[n] = f_contract
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
        
        # Update best
        best_idx = np.argmin(fvals)
        if fvals[best_idx] < best_val:
            best_val = fvals[best_idx]
            best_x = simplex[best_idx].copy()
    
    return best_x, -best_val  # Return best x and negative of best f (since we're maximizing)

def _params_to_P(params):
    """Convert log-probability parameters to normalized P."""
    # params is a vector of 100 log-values
    P = np.exp(params - np.max(params))  # Stable softmax-like
    total = np.sum(P)
    if total <= 1e-15:
        return np.ones((N, N)) / 100.0
    return P / total

def _objective_from_params(params):
    """Compute negative D/M ratio from log-probability parameters."""
    P = _params_to_P(params)
    return -_compute_ratio(P)

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 630 - 30  # 30s safety margin
    
    best_P = None
    best_score = -np.inf
    
    # Try multiple structural initializations
    for type_id in range(9):
        if time.time() > deadline:
            break
        
        P_init = _generate_structural_P(type_id)
        score_init = _compute_ratio(P_init)
        
        if score_init > best_score:
            best_score = score_init
            best_P = P_init.copy()
        
        # Convert to log-params for Nelder-Mead
        P_clipped = np.clip(P_init, 1e-15, None)
        P_clipped /= np.sum(P_clipped)
        params_init = np.log(P_clipped).ravel()
        
        # Run Nelder-Mead from this starting point
        if time.time() < deadline - 10:
            remaining_time = deadline - time.time()
            max_iter = min(2000, int(remaining_time * 50))  # ~50 iterations per second estimate
            
            best_params, score = _nelder_mead_minimize(
                _objective_from_params, 
                params_init, 
                max_iter=max_iter, 
                tol=1e-10, 
                deadline=deadline
            )
            
            if score > best_score:
                best_score = score
                best_P = _params_to_P(best_params)
        
        # Also try a random restart from this structural point
        if time.time() < deadline - 5:
            P_random = P_init.copy()
            noise = np.random.normal(0, 0.1, size=(N, N))
            P_random += noise
            P_random = np.clip(P_random, 1e-10, None)
            P_random /= np.sum(P_random)
            
            score_random = _compute_ratio(P_random)
            if score_random > best_score:
                best_score = score_random
                best_P = P_random.copy()
    
    # Additional random restarts with different strategies
    n_restarts = 0
    while time.time() < deadline - 5:
        n_restarts += 1
        
        # Generate a random distribution
        if n_restarts % 3 == 0:
            # Sparse random
            P_cand = np.zeros((N, N))
            n_points = np.random.randint(2, 15)
            points = np.random.randint(0, N, size=(n_points, 2))
            for px, py in points:
                P_cand[py, px] = np.random.uniform(0.1, 1.0)
        elif n_restarts % 3 == 1:
            # Gaussian blobs
            P_cand = np.zeros((N, N))
            n_blobs = np.random.randint(2, 8)
            for _ in range(n_blobs):
                cx = np.random.uniform(0, N-1)
                cy = np.random.uniform(0, N-1)
                weight = np.random.uniform(0.1, 1.0)
                sigma = np.random.uniform(0.5, 2.0)
                for y in range(N):
                    for x in range(N):
                        dist_sq = (x - cx)**2 + (y - cy)**2
                        P_cand[y, x] += weight * np.exp(-dist_sq / (2 * sigma**2))
        else:
            # Dirichlet random
            P_cand = np.random.dirichlet(np.ones(N*N)).reshape(N, N)
        
        # Refine with Nelder-Mead
        P_clipped = np.clip(P_cand, 1e-15, None)
        P_clipped /= np.sum(P_clipped)
        params_init = np.log(P_clipped).ravel()
        
        remaining_time = deadline - time.time()
        max_iter = min(1000, int(remaining_time * 30))
        
        best_params, score = _nelder_mead_minimize(
            _objective_from_params, 
            params_init, 
            max_iter=max_iter, 
            tol=1e-10, 
            deadline=deadline
        )
        
        if score > best_score:
            best_score = score
            best_P = _params_to_P(best_params)
    
    # Final validation and normalization
    if best_P is None:
        return np.ones((N, N)) / 100.0
    
    best_P = np.clip(best_P, 0, None)
    total = np.sum(best_P)
    if total <= 1e-9:
        return np.ones((N, N)) / 100.0
    best_P /= total
    
    return best_P
