# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 600

    # Fix scale: place one point at origin, constrain max distance to 1.
    # We optimize the positions of the other 15 points in complex plane (30 real dims).
    # Objective: maximize min(dist^2) subject to max(dist^2) <= 1.
    # We use a smooth surrogate: soft-min of log(dist^2) and soft-max of log(dist^2).
    # Actually, we want to maximize R = (d_min / d_max)^2.
    # Let f = log(d_min) - log(d_max). We want to maximize f.
    # d_min = min_i,j ||p_i - p_j||
    # d_max = max_i,j ||p_i - p_j||
    
    # Parameterization:
    # p_0 = 0
    # p_i = z_i for i=1..15, where z_i are complex numbers.
    # Constraint: max_{i,j} |p_i - p_j|^2 <= 1.
    # We can just optimize unconstrained and normalize?
    # If we scale points by s, d_min -> s*d_min, d_max -> s*d_max. Ratio is invariant.
    # So we can fix d_max = 1 (or any constant) and maximize d_min.
    # This is equivalent to: maximize min_{i,j} ||p_i - p_j|| subject to max_{i,j} ||p_i - p_j|| = 1.
    
    # Let's use the soft-min/soft-max approach.
    # Objective to maximize: S = soft_min(log(dists^2)) - soft_max(log(dists^2))
    # But we need to handle the constraint.
    # Alternatively, just maximize soft_min(log(dists^2)) and normalize the result at the end?
    # No, normalization changes the shape.
    
    # Better: Maximize min(dists) / max(dists).
    # Let's use the variable t = min(dists) / max(dists).
    # We can use the epigraph form:
    # Maximize t
    # Subject to: d_ij >= t * d_max for all i,j
    #             d_max = max_kl d_kl
    # This is non-smooth.
    
    # Let's stick to the smooth surrogate on the ratio directly.
    # R = (d_min / d_max)^2
    # log R = 2 * (log d_min - log d_max)
    # log d_min = soft_min(log d_ij)
    # log d_max = soft_max(log d_ij)
    # We want to maximize log R.
    # Gradient ascent on log R.
    
    # Let's define the function to maximize:
    # F(z) = soft_min(log(dists^2)) - soft_max(log(dists^2))
    # where dists are pairwise distances.
    # Note: dists^2 avoids sqrt.
    # log(dists^2) = 2 log(dists).
    # soft_min(x) = -logsumexp(-x) / beta ... wait.
    # soft_min(x) = -(1/beta) log(sum(exp(-beta * x)))
    # soft_max(x) = (1/beta) log(sum(exp(beta * x)))
    
    # We will use L-BFGS-B to maximize F(z).
    # We need to handle the constraint that points are distinct and within bounds?
    # The problem is scale invariant. The landscape has a valley at 0 (all points coincide).
    # We need a good initialization.
    
    # Initialization: regular polygon on a circle.
    # p_0 = 0.
    # p_i for i=1..15 on a circle of radius r?
    # If we put 15 points on a circle of radius 1, and one at 0,
    # d_max will be 2 (diameter). d_min will be chord length.
    # This is a reasonable start.
    
    def compute_dists(z):
        # z is shape (15, 2)
        # p_0 is (0,0)
        # Points: [0, z_0, ..., z_14]
        pts = np.vstack([np.array([[0.0, 0.0]]), z])
        # Compute pairwise squared distances
        # dists_sq[i,j] = ||p_i - p_j||^2
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Exclude diagonal
        mask = ~np.eye(n, dtype=bool)
        dists_sq = dists_sq[mask]
        return dists_sq

    def objective(z, beta):
        dists_sq = compute_dists(z)
        # Ensure positive
        dists_sq = np.maximum(dists_sq, 1e-10)
        log_dists_sq = np.log(dists_sq)
        
        # soft_min of log_dists_sq
        # soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
        # soft_max(x) = 1/beta * log(sum(exp(beta*x)))
        
        # Numerical stability
        # For soft_min:
        m = np.min(log_dists_sq)
        soft_min = -m - (1.0/beta) * np.log(np.sum(np.exp(-beta * (log_dists_sq - m))))
        
        # For soft_max:
        M = np.max(log_dists_sq)
        soft_max = M + (1.0/beta) * np.log(np.sum(np.exp(beta * (log_dists_sq - M))))
        
        # We want to maximize soft_min - soft_max
        # So objective for minimization is -(soft_min - soft_max)
        return -(soft_min - soft_max)

    def gradient(z, beta):
        dists_sq = compute_dists(z)
        dists_sq = np.maximum(dists_sq, 1e-10)
        log_dists_sq = np.log(dists_sq)
        
        # Gradient of soft_min
        # d/dx soft_min(x) = -sum(exp(-beta*x) * (-beta)) / sum(exp(-beta*x)) = beta * sum(exp(-beta*x)) / sum(exp(-beta*x)) ? No.
        # soft_min(x) = -1/beta log(sum exp(-beta x_i))
        # d/dx_k soft_min = -1/beta * 1/S * (-beta exp(-beta x_k)) = exp(-beta x_k) / S
        # where S = sum exp(-beta x_i)
        
        m = np.min(log_dists_sq)
        exp_neg = np.exp(-beta * (log_dists_sq - m))
        S_neg = np.sum(exp_neg)
        grad_soft_min = exp_neg / S_neg
        
        # Gradient of soft_max
        # soft_max(x) = 1/beta log(sum exp(beta x_i))
        # d/dx_k soft_max = 1/beta * 1/S * (beta exp(beta x_k)) = exp(beta x_k) / S
        # where S = sum exp(beta x_i)
        
        M = np.max(log_dists_sq)
        exp_pos = np.exp(beta * (log_dists_sq - M))
        S_pos = np.sum(exp_pos)
        grad_soft_max = exp_pos / S_pos
        
        # Gradient of objective = -(soft_min - soft_max) = soft_max - soft_min
        # grad_obj = grad_soft_max - grad_soft_min
        # This is a vector of length 120 (number of pairs)
        grad_log_dists = grad_soft_max - grad_soft_min
        
        # Now chain rule to z
        # log_dists_sq = log(||p_i - p_j||^2)
        # d log_dists_sq / d p_k = 2 * (p_k - p_i) / ||p_i - p_j||^2 if k is i or j, 0 otherwise.
        # Wait, d/dp_k (||p_i - p_j||^2) = 2(p_i - p_j) if k=i, -2(p_i - p_j) if k=j, 0 otherwise.
        # So d/dp_k log_dists_sq = 2(p_i - p_j)/dists_sq if k=i, -2(p_i - p_j)/dists_sq if k=j.
        
        # We need to map the 120 pairs back to the 16 points.
        # Points: 0 is fixed. 1..15 are variables z_0..z_14.
        # Index mapping:
        # pair (i, j) with i < j.
        # i, j in 0..15.
        # If i=0 or j=0, the variable is the other point.
        # Else both are variables.
        
        grad_z = np.zeros((15, 2))
        
        # Iterate over pairs
        # This is slow in Python. Let's vectorize if possible or just loop.
        # 120 pairs is small.
        
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                w = grad_log_dists[idx]
                # Vector v = p_i - p_j
                if i == 0:
                    p_i = np.array([0.0, 0.0])
                else:
                    p_i = z[i-1]
                if j == 0:
                    p_j = np.array([0.0, 0.0])
                else:
                    p_j = z[j-1]
                
                v = p_i - p_j
                d2 = np.dot(v, v)
                if d2 < 1e-10:
                    d2 = 1e-10
                
                coeff = 2.0 * w / d2
                
                # d/dp_i = coeff * v
                # d/dp_j = -coeff * v
                
                if i != 0:
                    grad_z[i-1] += coeff * v
                if j != 0:
                    grad_z[j-1] -= coeff * v
                
                idx += 1
                
        return grad_z

    # Optimization loop with annealing beta
    # Start with low beta (smooth), increase beta (sharper).
    
    best_score = -np.inf
    best_z = None
    
    # Initial guess: 15 points on a circle of radius 1, centered at 0?
    # But p_0 is at 0. So center of circle should be 0.
    angles = np.linspace(0, 2*np.pi, 15, endpoint=False)
    z0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Let's try a few different initializations and betas.
    
    betas = [10, 20, 50, 100, 200]
    
    current_z = z0.copy()
    
    for beta in betas:
        if time.time() > deadline - 60:
            break
            
        # Run L-BFGS-B
        res = minimize(
            lambda z: objective(z, beta),
            current_z,
            jac=lambda z: gradient(z, beta),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success:
            current_z = res.x
            
        # Check time
        if time.time() > deadline - 30:
            break
            
    # Final polish with high beta
    beta_final = 500
    if time.time() < deadline - 10:
        res = minimize(
            lambda z: objective(z, beta_final),
            current_z,
            jac=lambda z: gradient(z, beta_final),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if res.success:
            current_z = res.x

    # Construct final points
    points = np.vstack([np.array([[0.0, 0.0]]), current_z])
    
    # Normalize? The ratio is scale invariant, so normalization doesn't change the score.
    # But we should ensure d_max is not 0.
    # The problem statement says R = (d_min/d_max)^2.
    # Our optimization maximized log(d_min) - log(d_max), which is equivalent.
    
    return points

# EVOLVE-BLOCK-END
