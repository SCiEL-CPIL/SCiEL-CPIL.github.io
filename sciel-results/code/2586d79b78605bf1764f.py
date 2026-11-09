import numpy as np
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _score(points):
    """Compute the score: largest common radius."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points inside unit circle
    norms = np.hypot(points[:, 0], points[:, 1])
    if np.any(norms > 1.0):
        return 0.0
    # Min distance from any point to boundary
    boundary_min = 1.0 - np.max(norms)
    # Min pairwise distance / 2
    if N > 1:
        # Efficient pairwise distance calculation
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
    else:
        min_pair = boundary_min
    return min(min_pair, boundary_min)

def _soft_min_value_and_grad(points, sharpness):
    """
    Compute the value and gradient of the soft-min of the constraints.
    Constraints:
    1. Pairwise distances d_ij >= 2r
    2. Boundary distances b_i = 1 - ||p_i|| >= r
    
    We maximize r. This is equivalent to minimizing the "violation" 
    or maximizing the soft-min of (d_ij/2, b_i).
    
    Let f_i be the set of constraint values (d_ij/2 and b_i).
    We want max min(f_i).
    Using soft-min: S = -1/k * log(sum(exp(-k * f_i))).
    Gradient of S w.r.t positions gives direction to increase the min.
    """
    pts = points
    n = len(pts)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(d, np.inf)
    
    # Boundary distances
    norms = np.hypot(pts[:, 0], pts[:, 1])
    b = 1.0 - norms
    
    # Collect all constraint values
    # To avoid building a huge array, we compute the log-sum-exp terms directly
    # Terms: exp(-k * d_ij / 2) for i<j and exp(-k * b_i)
    
    # We need the sum of exp(-k * f_i)
    # Let's compute the max f to ensure numerical stability in exp
    # f_max = max(min_pair/2, min_boundary)
    
    min_pair_half = np.min(d) / 2.0
    min_b = np.min(b)
    f_max = max(min_pair_half, min_b)
    
    # Compute log_sum = log( sum exp(-k * f_i) )
    # sum exp(-k * f_i) = exp(-k * f_max) * sum exp(-k * (f_i - f_max))
    
    log_sum = -sharpness * f_max
    
    # Add pairwise terms
    # exp(-k * (d_ij/2 - f_max))
    # d_ij/2 - f_max <= 0
    d_half = d / 2.0
    d_half_minus_fmax = d_half - f_max
    # Mask diagonal
    d_half_minus_fmax = np.where(np.eye(n, dtype=bool), -np.inf, d_half_minus_fmax)
    
    exp_terms_pair = np.exp(-sharpness * d_half_minus_fmax)
    log_sum += np.log(np.sum(exp_terms_pair))
    
    # Add boundary terms
    b_minus_fmax = b - f_max
    exp_terms_b = np.exp(-sharpness * b_minus_fmax)
    log_sum += np.log(np.sum(exp_terms_b))
    
    soft_min_val = -log_sum / sharpness
    
    # Gradient of soft_min_val w.r.t pts
    # d/dp soft_min = -1/sharpness * d/dp log_sum
    # d/dp log_sum = (1/sum_exp) * sum (d/dp exp_term)
    # d/dp exp(-k * f_i) = exp(-k * f_i) * (-k * df_i/dp)
    # So d/dp log_sum = -k * sum( exp_term_i * df_i/dp ) / sum(exp_term_i)
    # d/dp soft_min = (1/sum_exp) * sum( exp_term_i * df_i/dp )
    
    sum_exp = np.sum(exp_terms_pair) + np.sum(exp_terms_b)
    
    grad = np.zeros_like(pts)
    
    # Pairwise contribution
    # f_ij = d_ij / 2
    # df_ij/dp_i = (p_i - p_j) / d_ij
    # df_ij/dp_j = (p_j - p_i) / d_ij
    
    # We have exp_terms_pair[i,j] = exp(-k * (d_ij/2 - f_max))
    # We need to sum over j for each i: exp_term_ij * (p_i - p_j) / d_ij
    # Note: d_ij is in denominator.
    
    # To be efficient, we can compute the weighted sum.
    # Let w_ij = exp_terms_pair[i,j]
    # grad_i += sum_j w_ij * (p_i - p_j) / d_ij
    
    # d is symmetric. diff = p_i - p_j.
    # term = exp_terms_pair[:, :, np.newaxis] * diff / d[:, :, np.newaxis]
    # But d has 0 on diagonal. We masked exp to 0 on diagonal (exp(-inf)=0).
    # So we can just compute.
    
    safe_d = np.where(d > 1e-8, d, 1.0)
    term_pair = exp_terms_pair[:, :, np.newaxis] * diff / safe_d[:, :, np.newaxis]
    grad_pair = np.sum(term_pair, axis=1)
    
    # Boundary contribution
    # f_i = 1 - ||p_i||
    # df_i/dp_i = -p_i / ||p_i||
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    term_b = exp_terms_b[:, np.newaxis] * (-pts / safe_norms[:, np.newaxis])
    grad_b = term_b
    
    grad = (grad_pair + grad_b) / sum_exp
    
    return soft_min_val, grad

def _optimize_with_soft_min(points, time_limit, sharpness_schedule=None):
    """
    Optimize using gradient ascent on the soft-min objective.
    """
    current = points.copy()
    best = current.copy()
    best_score = _score(current)
    
    # Adaptive sharpness
    k = 50.0
    
    lr = 0.01
    
    rng = np.random.default_rng(12345)
    
    step = 0
    while time.time() < time_limit:
        val, grad = _soft_min_value_and_grad(current, k)
        
        # Gradient ascent
        current = current + lr * grad
        
        # Project to unit disk
        norms = np.hypot(current[:, 0], current[:, 1])
        outside = norms > 1.0
        if np.any(outside):
            current[outside] = current[outside] / norms[outside, np.newaxis] * 0.9999
            
        # Check score
        if step % 10 == 0:
            score = _score(current)
            if score > best_score:
                best_score = score
                best = current.copy()
                
        # Increase sharpness slowly to sharpen the min
        k *= 1.01
        if k > 200.0:
            k = 200.0
            
        # Decrease learning rate
        lr *= 0.999
        
        step += 1
        
    return best, best_score

def _simulated_annealing_search(points, time_limit):
    """
    Simulated Annealing strategy:
    1. Optimize current configuration.
    2. Perturb configuration (move a random circle).
    3. Accept if score improves, or with probability exp(-delta/T).
    4. Cool down.
    """
    current = points.copy()
    best = current.copy()
    best_score = _score(current)
    
    T = 0.05
    T_min = 1e-4
    cooling = 0.99
    
    rng = np.random.default_rng(99)
    
    step = 0
    while time.time() < time_limit:
        # Optimization step
        # We run a short optimization to converge
        opt_end = time.time() + 0.5 # 0.5s per optimization
        current, score = _optimize_with_soft_min(current, opt_end)
        
        if score > best_score:
            best_score = score
            best = current.copy()
            
        # Perturbation
        # Pick 1-3 random circles and move them
        num_move = rng.integers(1, 4)
        idx = rng.choice(N, num_move, replace=False)
        
        # Move them to random positions or slightly perturb
        for i in idx:
            # Random perturbation
            current[i] += rng.normal(0, 0.05, 2)
            # Project to disk
            norm = np.linalg.norm(current[i])
            if norm > 1.0:
                current[i] = current[i] / norm * 0.9999
                
        # Evaluate new score
        new_score = _score(current)
        
        # Acceptance
        if new_score > best_score:
            best_score = new_score
            best = current.copy()
        else:
            # Accept with probability
            delta = best_score - new_score
            if delta < 0:
                accept = True
            else:
                accept_prob = np.exp(-delta / T)
                accept = rng.random() < accept_prob
                
            if accept:
                # Keep current (worse) state to explore
                pass
            else:
                # Revert to best
                current = best.copy()
        
        # Cool
        T *= cooling
        if T < T_min:
            T = T_min
            
        step += 1
        
    return best, best_score

def _initial_hexagonal():
    """Generate a hexagonal lattice initial guess."""
    points = []
    # Try to fit 50 circles
    # Hexagonal packing density ~ 0.9069
    # Area of unit circle = pi. Area of 50 circles = 50 * pi * r^2.
    # r^2 ~ 0.9069 * pi / (50 * pi) = 0.018 -> r ~ 0.134
    # Spacing s = 2r ~ 0.27
    
    s = 0.27
    k = int(1.0 / s) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - s / 2.0) ** 2:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) < N:
        # Fallback to random
        rng = np.random.default_rng(0)
        points = rng.uniform(-0.8, 0.8, (N, 2))
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 0.9
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9
    else:
        # Select N closest to origin
        norms = np.linalg.norm(points, axis=1)
        idx = np.argsort(norms, kind='stable')[:N]
        points = points[idx]
        
    return points

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Initial guess
    init_pts = _initial_hexagonal()
    
    # Time budget
    # We have 145s.
    # Use Simulated Annealing to explore different local minima.
    
    best_points, best_score = _simulated_annealing_search(init_pts, START_TIME + TIME_LIMIT)
    
    # Final polish with higher sharpness
    if time.time() < START_TIME + TIME_LIMIT - 5:
        polished, score_p = _optimize_with_soft_min(best_points, START_TIME + TIME_LIMIT - 2)
        if score_p > best_score:
            best_score = score_p
            best_points = polished.copy()
            
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
