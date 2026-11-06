"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Place 100 equal non-overlapping spheres in the unit cube [-0.5, 0.5]^3
    to maximize the common radius.
    
    Strategy:
    1. Start with a grid-based initialization (5x5x4 = 100 points).
    2. Use a repulsion-based local search (gradient ascent on the minimum distance).
    3. Use soft-min smoothing to make the objective differentiable.
    4. Time-bounded: returns best result found before deadline.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # --- Initialization: 5x5x4 grid ---
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    # Create grid points
    points = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    # We have exactly 5*5*4 = 100 points
    assert len(points) == N, f"Expected {N} points, got {len(points)}"
    
    def compute_radius(pts):
        """Compute the maximum valid radius for given points."""
        # Check boundary constraint
        if not np.all(np.abs(pts) <= 0.5):
            return 0.0
        # Compute minimum pairwise distance
        n = len(pts)
        min_dist = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
        return min_dist / 2.0
    
    def soft_min_radius(pts, alpha=50.0):
        """
        Smooth surrogate for the minimum radius using log-sum-exp.
        r = min( min_{i<j} |c_i - c_j|/2, min_i min_k (0.5 - |c_ik|) )
        
        We use a soft version: for each constraint, we compute a "violated" amount
        and combine them smoothly.
        """
        n = len(pts)
        # Boundary constraints: 0.5 - |c_ik| for each point and axis
        boundary_terms = []
        for i in range(n):
            for k in range(3):
                boundary_terms.append(0.5 - abs(pts[i, k]))
        
        # Pairwise distance constraints: |c_i - c_j|/2
        pair_terms = []
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                pair_terms.append(d / 2.0)
        
        all_terms = np.array(boundary_terms + pair_terms)
        
        # Soft min using log-sum-exp: softmin(x) = -logsumexp(-alpha * x) / alpha
        # For large alpha, this approaches min(x)
        neg_alpha_x = -alpha * all_terms
        max_val = np.max(neg_alpha_x)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_alpha_x - max_val)))
        soft_min_val = -log_sum_exp / alpha
        
        return soft_min_val, all_terms, pair_terms, boundary_terms
    
    def gradient_step(pts, alpha=50.0, lr=0.001):
        """
        Compute gradient of soft_min_radius w.r.t. point positions.
        """
        n = len(pts)
        grad = np.zeros_like(pts)
        
        # Compute all terms
        all_terms = []
        term_indices = []  # (type, i, j, k) for gradient computation
        
        # Boundary terms
        for i in range(n):
            for k in range(3):
                val = 0.5 - abs(pts[i, k])
                all_terms.append(val)
                term_indices.append(('b', i, k))
        
        # Pairwise terms
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-10:
                    d = 1e-10
                val = d / 2.0
                all_terms.append(val)
                term_indices.append(('p', i, j))
        
        all_terms = np.array(all_terms)
        
        # Soft min weights: w_t = exp(-alpha * x_t) / sum(exp(-alpha * x_s))
        neg_alpha_x = -alpha * all_terms
        max_val = np.max(neg_alpha_x)
        exp_vals = np.exp(neg_alpha_x - max_val)
        weights = exp_vals / np.sum(exp_vals)
        
        # Gradient of soft_min w.r.t. each term is -weight_t * alpha / alpha = -weight_t
        # But we need gradient w.r.t. positions, so chain rule
        # d(soft_min)/d(pos) = sum_t weight_t * d(term_t)/d(pos) * (-1) ... 
        # Actually: soft_min = -logsumexp(-alpha * x) / alpha
        # d(soft_min)/d(x_t) = alpha * exp(-alpha * x_t) / sum(exp(-alpha * x_s)) * (-1) / alpha * (-1)
        # Let me redo: 
        # f = -1/alpha * log(sum_s exp(-alpha * x_s))
        # df/dx_t = -1/alpha * (1/sum) * (-alpha * exp(-alpha * x_t)) = exp(-alpha * x_t) / sum(exp(-alpha * x_s)) = w_t
        # So df/dx_t = w_t
        # Then df/d(pos) = sum_t w_t * d(x_t)/d(pos)
        
        for idx, term in enumerate(term_indices):
            w = weights[idx]
            if w < 1e-10:
                continue
            
            if term[0] == 'b':
                _, i, k = term
                # x_t = 0.5 - |pts[i,k]|
                # d(x_t)/d(pts[i,k]) = -sign(pts[i,k])
                if pts[i, k] > 0:
                    grad[i, k] += w * (-1.0)
                elif pts[i, k] < 0:
                    grad[i, k] += w * (1.0)
                else:
                    grad[i, k] += 0.0  # Subgradient, pick 0
            else:
                _, i, j = term
                # x_t = |pts[i] - pts[j]| / 2
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < 1e-10:
                    continue
                # d(x_t)/d(pts[i]) = diff / (2 * d)
                # d(x_t)/d(pts[j]) = -diff / (2 * d)
                grad[i] += w * diff / (2.0 * d)
                grad[j] -= w * diff / (2.0 * d)
        
        return grad
    
    # Main optimization loop
    current_pts = points.copy()
    best_radius = compute_radius(current_pts)
    best_pts = current_pts.copy()
    
    # Adaptive learning rate
    lr = 0.005
    alpha = 30.0
    
    while time.time() < deadline:
        # Compute gradient
        grad = gradient_step(current_pts, alpha=alpha, lr=lr)
        
        # Update positions
        new_pts = current_pts + lr * grad
        
        # Project back to cube [-0.5, 0.5]^3
        new_pts = np.clip(new_pts, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Compute new radius
        new_radius = compute_radius(new_pts)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            current_pts = new_pts
            # Decrease learning rate slowly
            lr *= 0.999
        else:
            # Rejected, reduce learning rate
            lr *= 0.95
            if lr < 1e-6:
                # Try to restart with different alpha
                lr = 0.005
                alpha = np.random.uniform(20.0, 80.0)
        
        # Occasionally increase alpha for sharper min
        if np.random.random() < 0.01:
            alpha = np.random.uniform(20.0, 100.0)
    
    # Final polish: try to improve with smaller steps
    if time.time() < deadline - 5.0:
        polish_start = time.time()
        lr_polish = 0.001
        for _ in range(1000):
            if time.time() - polish_start > 5.0:
                break
            grad = gradient_step(best_pts, alpha=50.0, lr=lr_polish)
            new_pts = best_pts + lr_polish * grad
            new_pts = np.clip(new_pts, -0.5 + 1e-6, 0.5 - 1e-6)
            new_radius = compute_radius(new_pts)
            if new_radius > best_radius:
                best_radius = new_radius
                best_pts = new_pts
    
    return best_pts
# EVOLVE-BLOCK-END
