"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _generate_probes(n_probes):
    """
    Generate a dense set of probe points on the sphere using a Fibonacci spiral.
    These points serve as a fixed grid to evaluate the covering radius.
    The density is chosen such that the grid spacing is much smaller than the 
    expected covering radius (~11 degrees), ensuring the max over probes 
    closely approximates the true max over the continuous sphere.
    """
    # 11 degrees is ~0.19 rad. A spacing of ~2-3 degrees is sufficient.
    # Fibonacci spiral has roughly uniform area distribution.
    # Area per point ~ 4*pi / n_probes.
    # Angular spacing ~ sqrt(4*pi / n_probes).
    # For spacing ~ 0.05 rad (3 deg): 0.05^2 ~ 4*pi / n => n ~ 5000.
    # Let's use 5000 probes for good accuracy and speed.
    i = np.arange(n_probes) + 0.5
    z = 1.0 - 2.0 * i / n_probes
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    probes = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return probes

def _covering_radius_and_grad(pts, probes, beta):
    """
    Compute the soft-max covering radius and its gradient.
    
    For each probe point y, let d(y) be the angular distance to the nearest configuration point.
    We want to minimize R = max_y d(y).
    We approximate max_y d(y) with soft-max: R_soft = (1/beta) * log(sum(exp(beta * d_i))).
    
    d_i = arccos(max_j (y . p_j)).
    To make it differentiable and stable, we work with cos(theta) = c_i.
    d_i = arccos(c_i).
    grad d_i / grad c_i = -1 / sqrt(1 - c_i^2).
    grad c_i / grad p_j: 
    c_i = max_j (y . p_j). Let j* be the argmax.
    If we use a hard max, grad is 0 for all j except j*, where it is y.
    To make it smooth, we can use a soft-max for the "nearest neighbor" selection as well, 
    or simply use the hard max for the nearest neighbor index but smooth the outer max.
    Given the high beta for the outer max, the "active" probes are those with large d_i.
    For a specific probe y, the nearest point p_j* is the one maximizing y . p_j.
    The gradient of d_i w.r.t p_j* is -y / sin(d_i).
    The gradient w.r.t other p_j is 0 (in the hard max case).
    
    However, to be robust, we can compute the gradient of the soft-max over probes.
    Let w_i = exp(beta * d_i) / sum(exp(beta * d_k)).
    R_soft = sum(w_i * d_i).
    grad R_soft = sum(w_i * grad d_i).
    
    grad d_i = -1/sqrt(1-c_i^2) * grad c_i.
    If we use hard max for c_i (nearest neighbor), grad c_i is y for the nearest neighbor j*.
    So grad d_i = -y / sqrt(1-c_i^2) applied to p_j*.
    
    This is effectively a weighted sum of "pulls" towards the probe points, weighted by 
    how "worst" that probe is (via w_i) and the geometric factor.
    """
    pts = _normalize(pts)
    
    # Compute dot products: (n_probes, N)
    # probes: (P, 3), pts: (N, 3)
    dots = probes @ pts.T
    
    # Find nearest neighbor for each probe
    # c_i = max_j dots[i, j]
    max_dots = np.max(dots, axis=1)
    # Clip for arccos stability
    max_dots = np.clip(max_dots, -1.0 + 1e-12, 1.0 - 1e-12)
    
    # Angular distances d_i
    d_i = np.arccos(max_dots)
    
    # Soft-max weights
    # Shift for stability
    max_d = np.max(d_i)
    exp_d = np.exp(beta * (d_i - max_d))
    sum_exp = np.sum(exp_d)
    w_i = exp_d / sum_exp
    
    # Soft-max radius
    R_soft = max_d + np.log(sum_exp) / beta
    
    # Gradient
    # grad R_soft = sum_i w_i * grad d_i
    # grad d_i = -1/sqrt(1 - c_i^2) * grad c_i
    # grad c_i w.r.t p_j*: y_i (the probe vector)
    # grad c_i w.r.t p_j (j != j*): 0
    
    # We need to accumulate the gradient for each point p_j.
    # For each probe i, the nearest point is j* = argmax_j dots[i, j].
    # The contribution to grad p_j* is:
    # w_i * (-1/sqrt(1 - max_dots[i]^2)) * probes[i]
    
    # Compute the factor for each probe
    # sin(d_i) = sqrt(1 - cos^2(d_i)) = sqrt(1 - max_dots^2)
    sin_d_i = np.sqrt(np.clip(1.0 - max_dots**2, 1e-12, 1.0))
    factors = -w_i / sin_d_i
    
    # Find nearest neighbor indices
    nearest_idx = np.argmax(dots, axis=1)
    
    # Accumulate gradient
    grad = np.zeros_like(pts)
    # Use np.add.at for efficient accumulation
    # grad[nearest_idx] += factors[:, None] * probes
    np.add.at(grad, nearest_idx, factors[:, None] * probes)
    
    return R_soft, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Generate probes
    n_probes = 5000
    probes = _generate_probes(n_probes)
    
    best_pts = None
    best_score = float('inf')
    
    # 1. Generate initial candidates
    candidates = []
    
    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    candidates.append(fib_pts)
    
    # Random starts
    for seed in range(15):
        rng = np.random.RandomState(seed)
        pts = rng.randn(N, 3)
        pts = _normalize(pts)
        candidates.append(pts)
        
    # Evaluate candidates
    for pts in candidates:
        if time.time() > deadline - 20.0:
            break
        try:
            # Use a moderate beta for initial ranking
            score, _ = _covering_radius_and_grad(pts, probes, beta=20.0)
            if score < best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
            
    if best_pts is None:
        best_pts = fib_pts
        
    # 2. Gradient Descent with dynamic temperature (beta) schedule
    current_pts = best_pts.copy()
    current_score = best_score
    
    # Start with a smoother approximation (lower beta) to escape local minima,
    # then increase beta to sharpen the max approximation.
    beta = 10.0
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(current_pts)
    
    no_improve = 0
    max_no_improve = 100
    
    # Main optimization loop
    while time.time() < deadline - 10.0 and no_improve < max_no_improve:
        score, grad = _covering_radius_and_grad(current_pts, probes, beta=beta)
        
        if not np.isfinite(score):
            break
            
        # Update velocity
        v = momentum * v - lr * grad
        
        # Update points
        new_pts = current_pts + v
        new_pts = _normalize(new_pts)
        
        # Check for improvement
        new_score, _ = _covering_radius_and_grad(new_pts, probes, beta=beta)
        
        if new_score < current_score - 1e-7:
            current_pts = new_pts
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_pts = current_pts.copy()
                
            # Increase beta slightly as we converge to sharpen the objective
            # This helps us focus on the true worst-case points
            if beta < 100.0:
                beta *= 1.05
                # Reduce learning rate as beta increases (gradient becomes steeper)
                lr *= 0.98
        else:
            no_improve += 1
            # If stuck, reduce learning rate and momentum to settle
            lr *= 0.9
            v *= 0.5
            # Occasionally, try to escape by increasing beta more aggressively?
            # Or restart from a nearby point?
            # For now, just dampen.
            
            # If we are very stuck, try a random perturbation?
            if no_improve > 50:
                # Perturb
                rng = np.random.RandomState(int(time.time() * 1000) % 10000)
                perturb = rng.randn(N, 3) * 0.01
                current_pts = _normalize(current_pts + perturb)
                v = np.zeros_like(current_pts)
                no_improve = 0
                current_score, _ = _covering_radius_and_grad(current_pts, probes, beta=beta)

    # 3. Final polishing with high beta
    if time.time() < deadline - 5.0:
        current_pts = best_pts.copy()
        current_score = best_score
        beta = 100.0
        lr = 0.001
        v = np.zeros_like(current_pts)
        no_improve = 0
        
        while time.time() < deadline - 5.0 and no_improve < 50:
            score, grad = _covering_radius_and_grad(current_pts, probes, beta=beta)
            if not np.isfinite(score):
                break
            v = 0.9 * v - lr * grad
            new_pts = _normalize(current_pts + v)
            new_score, _ = _covering_radius_and_grad(new_pts, probes, beta=beta)
            
            if new_score < current_score - 1e-8:
                current_pts = new_pts
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            else:
                no_improve += 1
                lr *= 0.9
                v *= 0.5

    # Final validation and normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
