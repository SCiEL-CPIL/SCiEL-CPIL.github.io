import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    dx = centers[:, 0:1] - centers[:, 0:1].T
    dy = centers[:, 1:2] - centers[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _concentric_packing():
    """
    Generate an initial packing using concentric rings.
    Ring counts: 1, 6, 12, 18, 13 = 50
    Radii tuned to be roughly optimal for r ~ 0.125
    """
    ring_counts = [1, 6, 12, 18, 13]
    radii = [0.0, 0.25, 0.50, 0.75, 0.90]
    
    centers = []
    for k, count in enumerate(ring_counts):
        r_k = radii[k]
        if count == 1:
            centers.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            # Interleave phases
            if k % 2 == 1:
                angles += np.pi / count
            for a in angles:
                x = r_k * np.cos(a)
                y = r_k * np.sin(a)
                centers.append([x, y])
                
    return np.array(centers)

def _soft_min_value_and_grad(x, beta):
    """
    Compute the negative log-sum-exp approximation of the minimum distance.
    We want to maximize the minimum of:
    1. Pairwise distances / 2
    2. Boundary distances (1 - |c_i|)
    
    Let d_ij = |c_i - c_j| / 2
    Let b_i = 1 - |c_i|
    
    We want to maximize min(d_ij, b_i).
    This is equivalent to minimizing -min(d_ij, b_i).
    
    Using log-sum-exp:
    min(x) ≈ -(1/beta) * log(sum(exp(-beta * x)))
    
    So -min(x) ≈ (1/beta) * log(sum(exp(-beta * x)))
    
    We construct a vector of all candidate radii:
    R = [d_01, d_02, ..., d_49, b_0, b_1, ..., b_49]
    
    Objective to minimize: f(x) = (1/beta) * log(sum(exp(-beta * R)))
    Gradient: 
    Let S = sum(exp(-beta * R))
    Let p_k = exp(-beta * R_k) / S  (softmax probabilities)
    
    df/dx = (1/beta) * (1/S) * sum_k (exp(-beta * R_k) * (-beta * dR_k/dx))
           = - sum_k p_k * (dR_k/dx)
    
    This is the negative weighted average of the gradients of the individual radii.
    """
    x = x.reshape(N, 2)
    
    # Pairwise distances
    dx = x[:, 0:1] - x[:, 0:1].T
    dy = x[:, 1:2] - x[:, 1:2].T
    dists_sq = dx**2 + dy**2
    # Avoid division by zero
    dists_sq[dist_sq < 1e-12] = 1e-12
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    # Half distances
    d = dists / 2.0
    
    # Boundary distances
    r_vec = np.hypot(x[:, 0], x[:, 1])
    b = 1.0 - r_vec
    
    # Construct the full set of radii
    # We need to be careful with memory and speed.
    # N=50, so ~1225 pairs + 50 boundary = 1275 values.
    
    # Get upper triangle indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_pairs = d[i_idx, j_idx]
    
    # Combine
    R = np.concatenate([d_pairs, b])
    
    # Log-sum-exp stability
    # log(sum(exp(-beta * R)))
    # = -beta * max(R) + log(sum(exp(-beta * (R - max(R)))))
    R_max = np.max(R)
    # If R_max is very small or negative, the packing is invalid, but we handle that via constraints or large penalty.
    # For gradient, we need exp(-beta * (R - R_max))
    
    # If beta is large, underflow can happen.
    # Let's use the standard stable formula.
    # Actually, we want to minimize (1/beta) * log(sum(exp(-beta * R)))
    # Let's compute the probabilities p_k = exp(-beta * (R_k - R_max)) / sum(...)
    # Note: exp(-beta * (R_k - R_max)) = exp(beta * (R_max - R_k))
    
    # To avoid overflow if R_max - R_k is large (which means R_k is small), 
    # note that we are looking at the MIN. 
    # The term with the smallest R will have the largest exponent in the "exp(-beta*R)" sum?
    # No. exp(-beta * R_small) is LARGE.
    # So the sum is dominated by the smallest R.
    # This is correct.
    
    # Let's compute z_k = -beta * (R_k - R_max) = beta * (R_max - R_k)
    # Since R_max is the max, R_k - R_max <= 0, so z_k >= 0.
    # Wait.
    # We want min(R).
    # sum(exp(-beta * R)).
    # If R_min is small, exp(-beta * R_min) is large.
    # So the sum is dominated by R_min.
    # log(sum) ≈ -beta * R_min.
    # (1/beta) * log(sum) ≈ -R_min.
    # We want to MINIMIZE this? No, we want to MAXIMIZE R_min.
    # So we want to MINIMIZE -R_min.
    # So objective J = (1/beta) * log(sum(exp(-beta * R))) approximates -min(R).
    # Minimizing J is maximizing min(R). Correct.
    
    # Gradient of J:
    # J = (1/beta) * log(S), S = sum(exp(-beta * R_k))
    # dJ/dx = (1/beta) * (1/S) * sum(-beta * exp(-beta * R_k) * dR_k/dx)
    #       = - sum( (exp(-beta * R_k) / S) * dR_k/dx )
    #       = - sum( p_k * dR_k/dx )
    # where p_k = exp(-beta * R_k) / S.
    
    # Let's compute p_k stably.
    # p_k = exp(-beta * R_k) / S
    # Multiply num and den by exp(beta * R_max):
    # p_k = exp(-beta * (R_k - R_max)) / sum(exp(-beta * (R_j - R_max)))
    # Let u_k = R_max - R_k. u_k >= 0.
    # p_k = exp(-beta * u_k) / sum(exp(-beta * u_j))
    
    u = R_max - R
    # If beta is large, exp(-beta * u) underflows for u > 0.
    # This is good, it means only the minimums contribute.
    # But we need to handle the case where multiple R are close to min.
    
    # To avoid underflow to 0 for all, we can shift by min(R) instead?
    # No, the standard trick is:
    # log(sum(exp(-beta * R))) = -beta * R_max + log(sum(exp(-beta * (R - R_max))))
    # Let v_k = R_k - R_max. v_k <= 0.
    # exp(-beta * v_k) = exp(beta * (R_max - R_k)) = exp(beta * u_k).
    # This overflows if u_k is large!
    
    # Correct stable form for min:
    # We are summing exp(-beta * R).
    # The largest term is for the smallest R.
    # Let R_min = min(R).
    # log(sum(exp(-beta * R))) = -beta * R_min + log(sum(exp(-beta * (R - R_min))))
    # Let w_k = R_k - R_min. w_k >= 0.
    # Term is exp(-beta * w_k).
    # This underflows for large w_k, which is fine.
    
    R_min = np.min(R)
    w = R - R_min
    # w >= 0
    exp_terms = np.exp(-beta * w)
    S = np.sum(exp_terms)
    
    if S == 0:
        # All underflowed? Should not happen for R_min term which is exp(0)=1.
        return 0.0, np.zeros_like(x)
    
    p = exp_terms / S
    
    # Objective value
    # J = (1/beta) * (-beta * R_min + log(S)) = -R_min + (1/beta) * log(S)
    J = -R_min + (1.0 / beta) * np.log(S)
    
    # Gradient
    # dJ/dx = - sum(p_k * dR_k/dx)
    
    grad = np.zeros_like(x)
    
    # Part 1: Pairwise distances
    # p_pairs = p[:len(d_pairs)]
    p_pairs = p[:N*(N-1)//2]
    
    # We need dR_k/dx for each pair.
    # R_k = 0.5 * |x_i - x_j|
    # dR_k/dx_i = 0.5 * (x_i - x_j) / |x_i - x_j|
    # dR_k/dx_j = 0.5 * (x_j - x_i) / |x_i - x_j|
    
    # d_pairs is the distance (not halved? No, d was halved).
    # In _soft_min_value_and_grad, d was dists/2.
    # So R_k = d[i,j].
    # dR_k/dx_i = (x_i - x_j) / (2 * |x_i - x_j|) ?
    # Let D = |x_i - x_j|. R = D/2.
    # dR/dx_i = 0.5 * (x_i - x_j)/D.
    
    # We have dists matrix.
    # dists[i,j] = D.
    # So dR/dx_i = 0.5 * (x_i - x_j) / dists[i,j]
    
    # Let's accumulate gradients for pairs.
    # This loop is O(N^2). N=50, 1225 pairs. Fast.
    for k in range(len(d_pairs)):
        if p_pairs[k] < 1e-10:
            continue
        i = i_idx[k]
        j = j_idx[k]
        D = dists[i, j]
        if D < 1e-10:
            continue
        # Vector from j to i
        vec = x[i] - x[j]
        # Gradient contribution for i
        # dR/dx_i = 0.5 * vec / D
        grad_i = 0.5 * vec / D
        # Gradient contribution for j
        # dR/dx_j = -0.5 * vec / D
        grad_j = -0.5 * vec / D
        
        # J = - sum p dR
        # dJ/dx_i += - p_k * (dR/dx_i)
        grad[i] -= p_pairs[k] * grad_i
        grad[j] -= p_pairs[k] * grad_j
        
    # Part 2: Boundary distances
    # p_bound = p[len(d_pairs):]
    p_bound = p[N*(N-1)//2:]
    
    # R_k = 1 - |x_k|
    # dR_k/dx_k = - x_k / |x_k|
    
    for k in range(N):
        if p_bound[k] < 1e-10:
            continue
        r_k = np.hypot(x[k, 0], x[k, 1])
        if r_k < 1e-10:
            # Gradient of |x| at 0 is undefined, but for packing, we don't want center at 0 usually.
            # If we are at 0, boundary dist is 1.
            # d(1 - 0)/dx = 0.
            continue
        grad_vec = -x[k] / r_k
        # dJ/dx_k += - p_k * (dR/dx_k)
        grad[k] -= p_bound[k] * grad_vec
        
    return J, grad

def construct():
    deadline = time.time() + 145.0
    
    # Initial packing
    centers = _concentric_packing()
    x0 = centers.flatten()
    
    # We use L-BFGS-B to minimize the soft-min objective.
    # We need to choose beta.
    # Small beta -> smooth, far from min.
    # Large beta -> sharp, close to min, but non-differentiable limit.
    # Start with moderate beta, increase over time?
    # Or just use a fixed high beta if L-BFGS-B is robust enough.
    # Let's try annealing beta.
    
    best_x = x0.copy()
    best_val = _compute_radius(centers)
    
    # List of betas to try
    betas = [10, 50, 100, 200, 500, 1000]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        # Define objective and gradient for L-BFGS-B
        def obj_grad(x):
            val, grad = _soft_min_value_and_grad(x, beta)
            return val, grad
            
        # Use a callback to track time and best solution
        # L-BFGS-B doesn't have a built-in time limit, so we rely on maxiter
        # or a callback that raises an exception?
        # Better: use maxiter and check time between major steps?
        # L-BFGS-B is fast. Let's give it a budget of iterations.
        
        # Approximate iterations needed.
        # With N=50, 100 variables.
        # Let's allow 200 iterations per beta stage.
        
        try:
            res = minimize(
                obj_grad, 
                current_x, 
                method='L-BFGS-B', 
                jac=True,
                options={
                    'maxiter': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            current_x = res.x
            curr_centers = current_x.reshape(N, 2)
            curr_r = _compute_radius(curr_centers)
            
            if curr_r > best_val:
                best_val = curr_r
                best_x = current_x.copy()
                
        except Exception as e:
            # If optimization fails, keep current_x
            pass
            
    # Final polish with very high beta and more iterations if time permits
    if time.time() < deadline - 5:
        beta = 2000
        def obj_grad_high(x):
            val, grad = _soft_min_value_and_grad(x, beta)
            return val, grad
            
        try:
            res = minimize(
                obj_grad_high, 
                best_x, 
                method='L-BFGS-B', 
                jac=True,
                options={
                    'maxiter': 100,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 50
                }
            )
            final_x = res.x
            final_centers = final_x.reshape(N, 2)
            final_r = _compute_radius(final_centers)
            
            if final_r > best_val:
                best_val = final_r
                best_x = final_x.copy()
        except Exception:
            pass
            
    return best_x.reshape(N, 2)

# Dummy functions to maintain interface compatibility if strictly required, 
# though construct is the main entry.
def _hexagonal_fallback():
    return _concentric_packing()

def _build_grid(centers, cell_size):
    return {}

def _get_neighbors(grid, x, y, cell_size, search_radius):
    return []

def _calc_forces(centers, r_target):
    return np.zeros_like(centers)

def _anneal(centers, deadline):
    return centers, _compute_radius(centers)
