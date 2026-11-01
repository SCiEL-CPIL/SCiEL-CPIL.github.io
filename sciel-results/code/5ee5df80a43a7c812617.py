"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import time


def _hexagonal_init(n, scale=1.0, offset=(0.0, 0.0)):
    """Generate a hexagonal packing initialization for n circles."""
    r = 0.5 * scale
    dx = 2 * r
    dy = r * np.sqrt(3)
    centers = []
    row = 0
    while len(centers) < n:
        x_start = r + (row % 2) * r
        for col in range(50):
            x = x_start + col * dx
            y = r + row * dy
            if 0 <= x <= 1 and 0 <= y <= 1:
                centers.append((x + offset[0], y + offset[1]))
            if len(centers) >= n:
                break
        row += 1
    # Shuffle to break symmetry
    np.random.shuffle(centers)
    return np.array(centers[:n])


def _compute_objective(x, n):
    """
    x: [2n + 1] vector: [x_1..x_n, y_1..y_n, t]
    Objective: -t (we maximize t)
    """
    centers = x[:2*n].reshape(n, 2)
    t = x[2*n]
    return -t


def _compute_gradients(x, n, beta=50.0):
    """
    Compute gradient of the log-sum-exp surrogate for the minimum radius constraint.
    We want to maximize the minimum of (min pairwise dist/2, min boundary dist).
    """
    centers = x[:2*n].reshape(n, 2)
    t = x[2*n]
    
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    boundary = np.minimum(
        np.minimum(centers[:, 0], 1 - centers[:, 0]),
        np.minimum(centers[:, 1], 1 - centers[:, 1])
    )
    
    # For each circle i, the constraint is:
    # r_i <= min_j (dist_ij / 2) and r_i <= boundary_i
    # We use r_i = t (all equal radii for simplicity in this formulation)
    # Actually, let's use the formulation where we maximize t such that
    # all pairwise dists >= 2t and all boundary dists >= t
    
    # Log-sum-exp for pairwise constraints
    # For each pair (i,j), constraint: dist_ij >= 2t
    # Surrogate: sum exp(-beta * (dist_ij - 2t)) should be small
    
    # Gradient w.r.t. centers
    grad = np.zeros(2*n + 1)
    
    # Pairwise gradient contribution
    # For pair (i,j): contribution to grad of x_i is -beta * exp(-beta*(d_ij - 2t)) * (x_i - x_j) / d_ij
    # Actually, we want to maximize min, so we minimize sum exp(-beta*(d_ij - 2t))
    # d/dx_i exp(-beta*(d_ij - 2t)) = exp(-beta*(d_ij - 2t)) * (-beta) * d/dx_i d_ij
    # d/dx_i d_ij = (x_i - x_j) / d_ij
    
    for i in range(n):
        for j in range(i+1, n):
            d = dists[i, j]
            if d < 1e-10:
                continue
            val = np.exp(-beta * (d - 2*t))
            # Gradient for x_i
            diff = centers[i] - centers[j]
            grad[2*i] += -beta * val * diff[0] / d
            grad[2*i+1] += -beta * val * diff[1] / d
            # Gradient for x_j
            grad[2*j] += beta * val * diff[0] / d
            grad[2*j+1] += beta * val * diff[1] / d
    
    # Boundary gradient contribution
    # For circle i: constraint boundary_i >= t
    # Surrogate: exp(-beta * (boundary_i - t))
    for i in range(n):
        # x-boundary
        if centers[i, 0] < 1 - centers[i, 0]:
            # boundary is x_i
            val_x = np.exp(-beta * (centers[i, 0] - t))
            grad[2*i] += -beta * val_x
            grad[2*n] += beta * val_x
        else:
            # boundary is 1 - x_i
            val_x = np.exp(-beta * (1 - centers[i, 0] - t))
            grad[2*i] += beta * val_x
            grad[2*n] += beta * val_x
        
        # y-boundary
        if centers[i, 1] < 1 - centers[i, 1]:
            val_y = np.exp(-beta * (centers[i, 1] - t))
            grad[2*i+1] += -beta * val_y
            grad[2*n] += beta * val_y
        else:
            val_y = np.exp(-beta * (1 - centers[i, 1] - t))
            grad[2*i+1] += beta * val_y
            grad[2*n] += beta * val_y
    
    # Gradient w.r.t. t: -1 (since objective is -t)
    grad[2*n] += -1
    
    return grad


def _true_objective(x, n):
    """Compute the true minimum radius (the actual objective value)."""
    centers = x[:2*n].reshape(n, 2)
    
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2
    
    # Boundary distances
    boundary = np.minimum(
        np.minimum(centers[:, 0], 1 - centers[:, 0]),
        np.minimum(centers[:, 1], 1 - centers[:, 1])
    )
    min_boundary = np.min(boundary)
    
    return min(min_pairwise, min_boundary)


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    np.random.seed(42)
    
    best_result = None
    best_score = -np.inf
    
    # Multiple restarts with different initializations
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Different initialization strategies
        if restart % 3 == 0:
            # Hexagonal
            centers = _hexagonal_init(N, scale=0.95, offset=(0.02, 0.02))
        elif restart % 3 == 1:
            # Grid with jitter
            side = int(np.ceil(np.sqrt(N)))
            centers = []
            for i in range(side):
                for j in range(side):
                    centers.append([(i + 0.5) / side + np.random.randn() * 0.01,
                                    (j + 0.5) / side + np.random.randn() * 0.01])
                    if len(centers) >= N:
                        break
                if len(centers) >= N:
                    break
            centers = np.array(centers[:N])
        else:
            # Random
            centers = np.random.rand(N, 2) * 0.9 + 0.05
        
        # Clip to [0.05, 0.95] to ensure feasible start
        centers = np.clip(centers, 0.05, 0.95)
        
        # Initialize t
        t0 = 0.01
        
        x0 = np.concatenate([centers.flatten(), [t0]])
        
        # Bounds: centers in [0.01, 0.99], t in [0.001, 0.5]
        bounds = Bounds(
            lb=np.concatenate([np.full(2*N, 0.01), [0.001]]),
            ub=np.concatenate([np.full(2*N, 0.99), [0.5]])
        )
        
        # L-BFGS-B optimization with smooth surrogate
        try:
            result = minimize(
                _compute_objective,
                x0,
                args=(N,),
                method='L-BFGS-B',
                jac=_compute_gradients,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        if time.time() > deadline - 5.0:
            break
        
        # Polish with Powell on the true objective
        def true_neg_objective(x, n=N):
            return -_true_objective(x, n)
        
        try:
            result2 = minimize(
                true_neg_objective,
                result.x,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-12}
            )
        except Exception:
            result2 = result
        
        score = -result2.fun  # This is the minimum radius
        if score > best_score:
            best_score = score
            best_result = result2.x
    
    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    centers = best_result[:2*N].reshape(N, 2)
    t = best_result[2*N]
    
    # Ensure feasibility with a small epsilon
    # Check and adjust if needed
    eps = 1e-7
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2*t + eps:
                # Push apart
                mid = (centers[i] + centers[j]) / 2
                direction = (centers[i] - centers[j]) / (d + 1e-15)
                centers[i] = mid + direction * (t + eps/2)
                centers[j] = mid - direction * (t + eps/2)
        
        # Boundary
        for dim in range(2):
            if centers[i, dim] < t + eps:
                centers[i, dim] = t + eps
            if centers[i, dim] > 1 - t - eps:
                centers[i, dim] = 1 - t - eps
    
    # Final radius is the true minimum
    final_r = _true_objective(np.concatenate([centers.flatten(), [t]]), N)
    final_r = max(final_r, 0.001)
    
    return np.column_stack([centers[:, 0], centers[:, 1], np.full(N, final_r)])
# EVOLVE-BLOCK-END
