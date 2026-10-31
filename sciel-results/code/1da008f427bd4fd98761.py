"""Combined program: Smooth surrogate optimization with annealed sharpness and diverse geometric inits."""
import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    # Smooth objective and gradient with tunable sharpness k
    def make_objective_and_grad(k):
        def smooth_objective(x):
            pts = x.reshape(N, 2)
            d_sq = get_sq_dists(pts)
            d = np.sqrt(d_sq)
            
            d_max_val = np.max(d)
            d_min_val = np.min(d)
            
            # Soft max
            exp_term_max = np.exp(k * (d - d_max_val))
            soft_max = d_max_val + (1.0 / k) * np.log(np.sum(exp_term_max))
            
            # Soft min
            exp_term_min = np.exp(-k * (d - d_min_val))
            soft_min = d_min_val - (1.0 / k) * np.log(np.sum(exp_term_min))
            
            if soft_min < 1e-12:
                return 1e10
            
            ratio = soft_max / soft_min
            return ratio ** 2

        def smooth_gradient(x):
            pts = x.reshape(N, 2)
            d_sq = get_sq_dists(pts)
            d = np.sqrt(d_sq)
            
            d_max_val = np.max(d)
            d_min_val = np.min(d)
            
            exp_term_max = np.exp(k * (d - d_max_val))
            sum_exp_max = np.sum(exp_term_max)
            soft_max = d_max_val + (1.0 / k) * np.log(sum_exp_max)
            
            exp_term_min = np.exp(-k * (d - d_min_val))
            sum_exp_min = np.sum(exp_term_min)
            soft_min = d_min_val - (1.0 / k) * np.log(sum_exp_min)
            
            if soft_min < 1e-12:
                return np.zeros_like(x)
            
            ratio = soft_max / soft_min
            
            grad_soft_max_d = exp_term_max / sum_exp_max
            grad_soft_min_d = exp_term_min / sum_exp_min
            
            grad_obj_d = 2.0 * ratio / soft_min * (grad_soft_max_d - ratio * grad_soft_min_d)
            
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            d_full = np.sqrt(np.sum(diff**2, axis=2))
            d_full_safe = np.where(d_full > 1e-12, d_full, 1e-12)
            
            grad_obj_d_full = np.zeros((N, N))
            iu = np.triu_indices(N, 1)
            grad_obj_d_full[iu] = grad_obj_d
            grad_obj_d_full[iu[1], iu[0]] = grad_obj_d
            
            coeff = grad_obj_d_full / d_full_safe
            np.fill_diagonal(coeff, 0.0)
            
            row_sums = np.sum(coeff, axis=1)
            grad_x = row_sums[:, np.newaxis] * pts - (coeff @ pts)
            
            return grad_x.ravel()

        return smooth_objective, smooth_gradient

    # Initializations
    inits = []
    
    # 1. Random
    for _ in range(3):
        pts = np.random.uniform(-1, 1, (N, 2))
        inits.append(normalize(pts).ravel())
        
    # 2. Grid (4x4)
    x = np.arange(4) - 1.5
    y = np.arange(4) - 1.5
    xx, yy = np.meshgrid(x, y)
    base_grid = np.column_stack([xx.ravel(), yy.ravel()])
    inits.append(normalize(base_grid).ravel())
    
    # 3. Single Circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(normalize(circle_pts).ravel())
    
    # 4. Two Concentric Rings (Interleaved)
    # 8 inner, 8 outer
    r_inner = 0.5
    r_outer = 1.0
    angles_in = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    inner_pts = np.column_stack([r_inner * np.cos(angles_in), r_inner * np.sin(angles_in)])
    outer_pts = np.column_stack([r_outer * np.cos(angles_out), r_outer * np.sin(angles_out)])
    ring_pts = np.vstack([inner_pts, outer_pts])
    inits.append(normalize(ring_pts).ravel())
    
    # 5. Two Concentric Rings (Different radii)
    r_inner2 = 0.4
    r_outer2 = 0.9
    inner_pts2 = np.column_stack([r_inner2 * np.cos(angles_in), r_inner2 * np.sin(angles_in)])
    outer_pts2 = np.column_stack([r_outer2 * np.cos(angles_out), r_outer2 * np.sin(angles_out)])
    ring_pts2 = np.vstack([inner_pts2, outer_pts2])
    inits.append(normalize(ring_pts2).ravel())
    
    # 6. Hexagonal-like (Perturbed grid)
    for _ in range(2):
        noise = np.random.normal(0, 0.05, (N, 2))
        perturbed = base_grid + noise
        inits.append(normalize(perturbed).ravel())

    best_x = None
    best_score = 1e10
    
    # Optimization Strategy:
    # Phase 1: Smooth (k=5) to find basin
    # Phase 2: Sharp (k=50) to refine
    k_basin = 5.0
    k_refine = 50.0
    
    obj_basin, grad_basin = make_objective_and_grad(k_basin)
    obj_refine, grad_refine = make_objective_and_grad(k_refine)
    
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        # Phase 1
        try:
            res1 = minimize(
                obj_basin,
                x0,
                jac=grad_basin,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_curr = res1.x
        except Exception:
            continue
            
        # Phase 2
        try:
            res2 = minimize(
                obj_refine,
                x_curr,
                jac=grad_refine,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_curr = res2.x
        except Exception:
            pass # Keep res1 result if refine fails

        final_pts = x_curr.reshape(N, 2)
        raw_score = score_raw(final_pts)
        
        if raw_score < best_score:
            best_score = raw_score
            best_x = x_curr.copy()

    if best_x is None:
        # Fallback
        best_x = inits[0]
        best_score = score_raw(best_x.reshape(N, 2))
        
    # Final check for distinctness
    pts = best_x.reshape(N, 2)
    d_sq = get_sq_dists(pts)
    if np.min(d_sq) < 1e-10:
        # Add tiny perturbation
        pts = pts + np.random.normal(0, 1e-6, (N, 2))
        best_x = pts.ravel()

    return best_x.reshape(N, 2)

# EVOLVE-BLOCK-END
