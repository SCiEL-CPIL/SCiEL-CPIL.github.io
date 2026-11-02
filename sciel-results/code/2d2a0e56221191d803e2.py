"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Pack 32 circles in the unit square maximizing sum of radii.
    
    Uses a continuous differentiable optimization with a soft contact graph
    derived from differentiable nearest-neighbor search. We optimize positions
    and radii simultaneously, using smooth penalty functions that are only
    active for pairs predicted to be in contact.
    """
    best_result = None
    best_score = -np.inf
    
    def smooth_max(x, k=50):
        """Smooth approximation of max(0, x)."""
        return x / 2 + np.sqrt(x**2 / 4 + 1e-8)
    
    def objective(x, n_circles, sigma=0.03, w_overlap=800.0, w_boundary=300.0):
        """
        Differentiable objective with soft contact graph.
        
        x: array of shape (3*n,) containing [x1, y1, r1, x2, y2, r2, ...]
        """
        positions = x[0::3].reshape(-1, 2)
        radii = x[2::3]
        
        # Objective: maximize sum of radii (minimize negative)
        obj = -np.sum(radii)
        
        # Compute all pairwise distances
        diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=2)
        dists = np.sqrt(dists_sq + 1e-10)
        
        # Sum of radii pairs
        r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap amount: positive when circles overlap
        overlap = r_sums - dists
        
        # Soft contact graph: weight each pair by proximity to contact
        # Contact when overlap ≈ 0; use Gaussian kernel on overlap
        contact_weights = np.exp(-overlap**2 / (2 * sigma**2))
        np.fill_diagonal(contact_weights, 0)
        
        # Smooth overlap penalty: only penalize actual overlaps
        smooth_ov = smooth_max(overlap, k=20)
        
        overlap_penalty = w_overlap * np.sum(contact_weights * smooth_ov**2)
        
        # Boundary violations
        boundary_viol = np.zeros(n_circles)
        for i in range(n_circles):
            # Left: r_i - x_i > 0 means violation
            bv = radii[i] - positions[i, 0]
            boundary_viol[i] += smooth_max(bv, k=50)**2
            # Right: r_i - (1 - x_i) > 0 means violation
            bv = radii[i] - (1 - positions[i, 0])
            boundary_viol[i] += smooth_max(bv, k=50)**2
            # Bottom: r_i - y_i > 0 means violation
            bv = radii[i] - positions[i, 1]
            boundary_viol[i] += smooth_max(bv, k=50)**2
            # Top: r_i - (1 - y_i) > 0 means violation
            bv = radii[i] - (1 - positions[i, 1])
            boundary_viol[i] += smooth_max(bv, k=50)**2
        
        boundary_penalty = w_boundary * np.sum(boundary_viol)
        
        return obj + overlap_penalty + boundary_penalty
    
    def gradient(x, n_circles, sigma=0.03, w_overlap=800.0, w_boundary=300.0):
        """
        Analytical gradient for the objective function.
        """
        positions = x[0::3].reshape(-1, 2)
        radii = x[2::3]
        
        grad = np.zeros_like(x)
        
        # Gradient of -sum(radii)
        grad[2::3] += -1.0
        
        # Compute pairwise quantities
        diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=2)
        dists = np.sqrt(dists_sq + 1e-10)
        
        r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = r_sums - dists
        
        # Contact weights
        contact_weights = np.exp(-overlap**2 / (2 * sigma**2))
        np.fill_diagonal(contact_weights, 0)
        
        # Smooth max derivative: d/dx [x/2 + sqrt(x^2/4 + eps)] = 0.5 + x/(2*sqrt(x^2/4 + eps))
        def d_smooth_max(x, k=20):
            return 0.5 + x / (2 * np.sqrt(x**2 / 4 + 1e-8))
        
        dsmo = d_smooth_max(overlap, k=20)
        smo = smooth_max(overlap, k=20)
        
        # Gradient of overlap penalty
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                w = contact_weights[i, j]
                s = smo[i, j]
                ds = dsmo[i, j]
                
                # d(overlap)/d(positions[i]) = -(positions[i] - positions[j]) / dist
                diff_ij = positions[i] - positions[j]
                dist_ij = dists[i, j]
                d_ov_d_pos_i = -diff_ij / (dist_ij + 1e-10)
                
                # Gradient contribution
                # penalty = w_overlap * w * s^2
                # d(penalty)/d(pos_i) = w_overlap * (dw/dpos_i * s^2 + w * 2*s * ds/dov * dov/dpos_i)
                # Simplified: focus on dominant term w * 2*s * ds/dov * dov/dpos_i
                grad_pos_i = w_overlap * w * 2 * s * ds * d_ov_d_pos_i
                grad_pos_j = -grad_pos_i
                
                grad[3*i:3*i+2] += grad_pos_i
                grad[3*j:3*j+2] += grad_pos_j
                
                # Gradient w.r.t. radii: d(overlap)/d(r_i) = 1
                grad_r = w_overlap * w * 2 * s * ds
                grad[3*i+2] += grad_r
                grad[3*j+2] += grad_r
        
        # Gradient of boundary penalty
        def d_smooth_max_boundary(bv, k=50):
            return 0.5 + bv / (2 * np.sqrt(bv**2 / 4 + 1e-8))
        
        for i in range(n_circles):
            # Left: bv = r_i - x_i
            bv = radii[i] - positions[i, 0]
            ds = d_smooth_max_boundary(bv)
            s = smooth_max(bv, k=50)
            # d(penalty)/d(x_i) = w_boundary * 2*s*ds*(-1)
            grad[3*i] += w_boundary * 2 * s * ds * (-1)
            # d(penalty)/d(r_i) = w_boundary * 2*s*ds*(1)
            grad[3*i+2] += w_boundary * 2 * s * ds
        
            # Right: bv = r_i - (1 - x_i) = r_i - 1 + x_i
            bv = radii[i] - (1 - positions[i, 0])
            ds = d_smooth_max_boundary(bv)
            s = smooth_max(bv, k=50)
            grad[3*i] += w_boundary * 2 * s * ds * (1)
            grad[3*i+2] += w_boundary * 2 * s * ds
        
            # Bottom: bv = r_i - y_i
            bv = radii[i] - positions[i, 1]
            ds = d_smooth_max_boundary(bv)
            s = smooth_max(bv, k=50)
            grad[3*i+1] += w_boundary * 2 * s * ds * (-1)
            grad[3*i+2] += w_boundary * 2 * s * ds
        
            # Top: bv = r_i - (1 - y_i) = r_i - 1 + y_i
            bv = radii[i] - (1 - positions[i, 1])
            ds = d_smooth_max_boundary(bv)
            s = smooth_max(bv, k=50)
            grad[3*i+1] += w_boundary * 2 * s * ds * (1)
            grad[3*i+2] += w_boundary * 2 * s * ds
        
        return grad
    
    def is_valid(x, n_circles, tol=1e-5):
        """Check if solution satisfies all constraints."""
        positions = x[0::3].reshape(-1, 2)
        radii = x[2::3]
        
        # Check boundary
        for i in range(n_circles):
            if positions[i, 0] < radii[i] - tol or positions[i, 0] > 1 - radii[i] + tol:
                return False
            if positions[i, 1] < radii[i] - tol or positions[i, 1] > 1 - radii[i] + tol:
                return False
        
        # Check overlaps
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                dist = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
                if dist < radii[i] + radii[j] - tol:
                    return False
        
        return True
    
    def solve_from_init(x0, maxiter=800):
        """Solve optimization from initial point."""
        n_circles = len(x0) // 3
        bounds = []
        for i in range(n_circles):
            bounds.append((0.01, 0.99))  # x
            bounds.append((0.01, 0.99))  # y
            bounds.append((0.01, 0.25))  # radius
        
        result = minimize(
            objective,
            x0,
            jac=gradient,
            args=(n_circles,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
        )
        return result
    
    # Generate diverse initial configurations
    inits = []
    
    # Grid initialization (6x6)
    side = 6
    x0 = np.zeros(3 * N)
    for idx in range(N):
        i = idx // side
        j = idx % side
        x0[3*idx] = (j + 0.5) / side
        x0[3*idx+1] = (i + 0.5) / side
        x0[3*idx+2] = 0.5 / side * 0.85
    inits.append(x0)
    
    # Grid initialization (5x7)
    x0 = np.zeros(3 * N)
    for idx in range(N):
        i = idx // 7
        j = idx % 7
        x0[3*idx] = (j + 0.5) / 7
        x0[3*idx+1] = (i + 0.5) / 5
        x0[3*idx+2] = min(0.5/7, 0.5/5) * 0.85
    inits.append(x0)
    
    # Random initializations with different seeds
    for seed in [42, 123, 456, 789, 1000, 2024, 3141, 2718, 1618, 1414]:
        rng = np.random.RandomState(seed)
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = rng.uniform(0.1, 0.9)
            x0[3*i+1] = rng.uniform(0.1, 0.9)
            x0[3*i+2] = rng.uniform(0.03, 0.07)
        inits.append(x0)
    
    # Hexagonal packing initialization
    r_hex = 0.06
    dx = 2 * r_hex
    dy = r_hex * np.sqrt(3)
    x0 = np.zeros(3 * N)
    count = 0
    for row in range(10):
        for col in range(10):
            if count >= N:
                break
            x = dx * (col + 0.5 * (row % 2))
            y = dy * row
            x_centered = 0.5 + (x - 4.5 * dx) * 0.09
            y_centered = 0.5 + (y - 4.5 * dy) * 0.09
            if 0.05 < x_centered < 0.95 and 0.05 < y_centered < 0.95:
                x0[3*count] = x_centered
                x0[3*count+1] = y_centered
                x0[3*count+2] = r_hex * 0.85
                count += 1
    while count < N:
        x0[3*count] = 0.5 + np.random.uniform(-0.3, 0.3)
        x0[3*count+1] = 0.5 + np.random.uniform(-0.3, 0.3)
        x0[3*count+2] = 0.04
        count += 1
    inits.append(x0)
    
    # Solve from each initialization
    for x0 in inits:
        try:
            result = solve_from_init(x0, maxiter=600)
            x_opt = result.x
            
            if is_valid(x_opt, N):
                radii = x_opt[2::3]
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([x_opt[0::3].reshape(-1, 2), radii])
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(j + 0.5) / side, (i + 0.5) / side, 0.5 / side - 1e-3])
        return np.array(cells)
    
    return best_result
# EVOLVE-BLOCK-END
