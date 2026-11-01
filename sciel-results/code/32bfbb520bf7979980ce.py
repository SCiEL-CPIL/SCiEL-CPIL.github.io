"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(centers):
    """Calculate the score (radius) for a given set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Min pairwise distance
    min_pair_dist = np.min(dists[np.triu_indices(N, k=1)])
    
    # Min distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min(values, beta=100.0):
    """
    Soft minimum of an array of values using log-sum-exp trick.
    As beta -> infinity, this approaches the true minimum.
    """
    values = np.asarray(values)
    min_val = np.min(values)
    # Avoid overflow in exp
    shifted = values - min_val
    # Use a stable computation: min_val - log(sum(exp(-beta * (values - min_val)))) / beta
    # Actually: soft_min = -1/beta * log(sum(exp(-beta * values)))
    # = -1/beta * log(exp(-beta*min_val) * sum(exp(-beta*(values-min_val))))
    # = min_val - 1/beta * log(sum(exp(-beta*(values-min_val))))
    # Let u = -beta * (values - min_val) >= 0
    # sum(exp(u))
    # To avoid overflow, we can just compute directly if beta isn't too large
    # Or use the fact that values - min_val >= 0, so -beta*(values-min_val) <= 0
    # exp(negative) is safe.
    u = -beta * (values - min_val)
    sum_exp = np.sum(np.exp(u))
    return min_val - np.log(sum_exp) / beta

def _smooth_objective(x, beta=100.0):
    """
    Negative of the soft-min of all constraints (pair distances and wall distances).
    We want to maximize the minimum, so we minimize the negative.
    The objective is -soft_min(pair_dists/2, wall_dists)
    Actually, we want to maximize r = min(min_pair_dist/2, min_wall_dist).
    So we minimize -r.
    Using soft_min: r_soft = soft_min(all_constraints)
    where all_constraints = [d_ij/2 for i<j] + [0.5 - |c_ik| for i,k]
    We minimize -r_soft.
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    pair_constraints = dists[iu] / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    wall_constraints = np.min(wall_dists, axis=1)
    
    # Combine all constraints
    all_constraints = np.concatenate([pair_constraints, wall_constraints])
    
    r_soft = _soft_min(all_constraints, beta)
    
    return -r_soft

def _smooth_gradient(x, beta=100.0):
    """
    Gradient of the smooth objective.
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_constraints = pair_dists / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    wall_constraints = np.min(wall_dists, axis=1)
    
    all_constraints = np.concatenate([pair_constraints, wall_constraints])
    
    # Weights for soft min: w_k = exp(-beta * (c_k - min_c)) / sum(exp(-beta * (c_j - min_c)))
    min_c = np.min(all_constraints)
    u = -beta * (all_constraints - min_c)
    exp_u = np.exp(u)
    weights = exp_u / np.sum(exp_u)
    
    grad = np.zeros_like(x)
    
    # Pairwise part
    # d/dx_i (d_ij / 2) = (c_i - c_j) / d_ij
    # d/dx_j (d_ij / 2) = (c_j - c_i) / d_ij
    n_pairs = len(pair_constraints)
    for idx in range(n_pairs):
        w = weights[idx]
        i, j = iu[0][idx], iu[1][idx]
        d = pair_dists[idx]
        if d > 1e-10:
            vec = (centers[i] - centers[j]) / d
            # Objective is -soft_min, so gradient is -sum(w_k * grad_k)
            # grad of constraint is vec for i, -vec for j
            # So grad of -soft_min is -sum(w * grad_constraint)
            # Wait, soft_min = min_c - log(sum(exp(u)))/beta
            # d(soft_min)/dx = -1/beta * sum(exp(u_k) * d(-beta * c_k)/dx) / sum(exp(u_k))
            # = -1/beta * sum(exp(u_k) * (-beta * d(c_k)/dx)) / sum(exp(u_k))
            # = sum(exp(u_k) * d(c_k)/dx) / sum(exp(u_k))
            # = sum(w_k * d(c_k)/dx)
            # So d(soft_min)/dx = sum(w_k * grad(c_k)/dx)
            # We want to minimize -soft_min, so grad_obj = -sum(w_k * grad(c_k)/dx)
            
            grad[3*i:3*i+3] -= w * vec
            grad[3*j:3*j+3] += w * vec
    
    # Wall part
    # c_ik = 0.5 - |c_ik|
    # d(c_ik)/dx = -sign(c_ik)
    # For each sphere i, the wall constraint is min_k (0.5 - |c_ik|)
    # This is a bit tricky because of the min.
    # Actually, I defined wall_constraints as min_k (0.5 - |c_ik|) for each i.
    # So the constraint for sphere i is w_i = min_k (0.5 - |c_ik|).
    # The gradient of w_i with respect to c_i is -sign(c_ik*) where k* is the argmin.
    # If multiple k are tied, we take a weighted average, but usually one dominates.
    # Let's just use the argmin for simplicity, or the soft-min of the wall distances per sphere.
    # To be consistent, let's use soft-min for the wall constraints per sphere too.
    
    # Let's re-evaluate. The current implementation uses min for wall constraints per sphere.
    # This makes the gradient discontinuous.
    # Let's change wall_constraints to be the individual wall distances 0.5 - |c_ik| for all i,k.
    # Then we have N*3 wall constraints.
    
    # Let's redo the objective and gradient with all individual constraints.
    
    # Pair constraints: N*(N-1)/2
    # Wall constraints: N*3
    # Total: 1225 + 150 = 1375
    
    # Let's recompute weights with all constraints.
    
    # Actually, the current code uses min(wall_dists, axis=1) which is N constraints.
    # Let's stick with that for now, but handle the gradient carefully.
    
    # For sphere i, w_i = min_k (0.5 - |c_ik|).
    # Let k_i be the argmin.
    # grad(w_i) = -sign(c_{i,k_i}) e_{k_i}
    # grad_obj = -sum_i w_i * grad(w_i) ... wait, the weights are global.
    # The soft_min is over all constraints (pairs + walls).
    # So grad_obj = -sum_all_constraints w_k * grad(c_k)
    
    # For wall constraint i (which is min_k ...), the gradient is only non-zero in the direction of the argmin.
    # This is an approximation.
    
    for i in range(N):
        w = weights[n_pairs + i]
        # Find argmin wall distance for sphere i
        k_min = np.argmin(wall_dists[i])
        # grad of (0.5 - |c_ik|) wrt c_ik is -sign(c_ik)
        # So grad of constraint is -sign(c_ik)
        # grad_obj = -w * (-sign(c_ik)) = w * sign(c_ik)
        # Wait, grad_obj = -sum(w_k * grad(c_k))
        # grad(c_k) = -sign(c_ik)
        # So term is -w * (-sign(c_ik)) = w * sign(c_ik)
        # This pushes the sphere away from the wall?
        # If c_ik > 0, sign is 1, grad is 1. We minimize obj, so we decrease x.
        # Wait, gradient descent: x_new = x - lr * grad.
        # If grad is 1, x decreases.
        # If c_ik is positive (near +0.5 wall), we want to decrease x to move away.
        # So this is correct.
        
        grad[3*i + k_min] += w * np.sign(centers[i, k_min])
        
    return grad

def _refine_with_smooth(x0, time_limit=10.0, beta_start=10.0, beta_end=1000.0):
    """
    Refine the packing using smooth objective with increasing beta.
    """
    start_time = time.time()
    x = x0.copy()
    n_iter = 0
    
    # Increase beta gradually
    n_steps = 5
    betas = np.linspace(beta_start, beta_end, n_steps)
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        # Use L-BFGS-B with bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Limit iterations per beta step
        max_iter = max(1, int((time_limit - (time.time() - start_time)) / (n_steps - np.where(betas == beta)[0][0]) / 0.05))
        
        res = minimize(
            _smooth_objective,
            x,
            jac=_smooth_gradient,
            method='L-BFGS-B',
            bounds=bounds,
            args=(beta,),
            options={'maxiter': min(100, max_iter), 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        x = res.x
        n_iter += 1
        
    return x

def _initial_grid():
    """
    Generate an initial grid-based packing.
    """
    # 4x4x4 grid = 64 points, pick 50
    # Or 5x5x2 = 50 points?
    # Let's try a 5x5x2 grid
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.4, 0.4, 2)
    
    grid = np.array([(a, b, c) for c in z for b in y for a in x])
    return grid[:N]

def _initial_fcc_like():
    """
    Generate an FCC-like initial packing.
    """
    # FCC lattice in a cube
    # This is complex, let's just use a perturbed grid for now
    return _initial_grid() + np.random.normal(0, 0.01, (N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_centers = None
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different initializations
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Choose initialization
        if restart == 0:
            centers = _initial_grid()
        elif restart == 1:
            centers = _initial_fcc_like()
        elif restart < 5:
            # Perturbed grid
            centers = _initial_grid() + rng.normal(0, 0.02, (N, 3))
        else:
            # Random
            centers = rng.uniform(-0.45, 0.45, (N, 3))
        
        centers = np.clip(centers, -0.49, 0.49)
        
        # Refine using smooth objective
        time_left = deadline - time.time()
        refine_time = min(10.0, time_left / (n_restarts - restart))
        
        refined = _refine_with_smooth(centers, time_limit=refine_time)
        
        score = _score(refined)
        
        if score > best_score:
            best_score = score
            best_centers = refined.copy()
    
    # Final polish on the best candidate
    if best_centers is not None and (deadline - time.time()) > 10:
        final_time = deadline - time.time() - 5
        refined = _refine_with_smooth(best_centers, time_limit=final_time, beta_start=100.0, beta_end=1000.0)
        score = _score(refined)
        if score > best_score:
            best_score = score
            best_centers = refined.copy()
    
    if best_centers is None:
        # Fallback
        best_centers = _initial_grid()
    
    return best_centers
# EVOLVE-BLOCK-END
