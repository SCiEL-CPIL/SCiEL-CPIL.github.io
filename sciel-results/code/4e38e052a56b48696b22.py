"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def construct():
    # Precompute all triangle indices
    tri_indices = list(itertools.combinations(range(N), 3))
    n_tris = len(tri_indices)
    
    def bary_to_cartesian(bary):
        """Convert barycentric coords (l1, l2) to Cartesian in the unit equilateral triangle."""
        l1 = bary[:, 0]
        l2 = bary[:, 1]
        x = 0.5 - 0.5 * l1 + 0.5 * l2
        y = (1 - l1 - l2) * np.sqrt(3) / 2
        return np.column_stack([x, y])
    
    def get_valid_bary(params):
        """Reshape and clip parameters to valid barycentric coordinates."""
        bary = params.reshape(N, 2)
        bary = np.clip(bary, 0.001, 0.999)
        s = bary[:, 0] + bary[:, 1]
        mask = s > 0.998
        if np.any(mask):
            bary[mask, 0] *= 0.998 / s[mask]
            bary[mask, 1] *= 0.998 / s[mask]
        return bary
    
    def compute_min_area(bary):
        """Compute the actual minimum triangle area."""
        l3 = 1.0 - bary[:, 0] - bary[:, 1]
        bary_full = np.column_stack([bary[:, 0], bary[:, 1], l3])
        min_area = np.inf
        for i, j, k in tri_indices:
            M = bary_full[[i, j, k], :]
            det_val = abs(np.linalg.det(M))
            if det_val < min_area:
                min_area = det_val
        return min_area
    
    def make_softmin_objective(temperature):
        """Create objective and gradient functions with given temperature."""
        def compute_areas_and_grads(params):
            bary = get_valid_bary(params)
            l3 = 1.0 - bary[:, 0] - bary[:, 1]
            bary_full = np.column_stack([bary[:, 0], bary[:, 1], l3])
            
            grad_acc = np.zeros((N, 2))
            t = temperature
            
            # Compute areas and weights
            exp_terms = np.zeros(n_tris)
            max_exp = -np.inf
            for idx, (i, j, k) in enumerate(tri_indices):
                M = bary_full[[i, j, k], :]
                det_val = np.linalg.det(M)
                area_val = max(abs(det_val), 1e-10)
                exp_term = -area_val / t
                exp_terms[idx] = exp_term
                max_exp = max(max_exp, exp_term)
            
            exp_terms_shifted = np.exp(exp_terms - max_exp)
            sum_exp = np.sum(exp_terms_shifted)
            weights = exp_terms_shifted / sum_exp
            
            soft_min = -t * (max_exp + np.log(sum_exp))
            obj_val = -soft_min
            
            # Compute gradients
            for idx, (i, j, k) in enumerate(tri_indices):
                w = weights[idx]
                if w < 1e-10:
                    continue
                
                M = bary_full[[i, j, k], :]
                det_val = np.linalg.det(M)
                sign = 1.0 if det_val > 0 else -1.0
                
                m = M
                C = np.zeros((3, 3))
                C[0, 0] = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
                C[0, 1] = -(m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
                C[0, 2] = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]
                
                C[1, 0] = -(m[0, 1] * m[2, 2] - m[0, 2] * m[2, 1])
                C[1, 1] = m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]
                C[1, 2] = -(m[0, 0] * m[2, 1] - m[0, 1] * m[2, 0])
                
                C[2, 0] = m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]
                C[2, 1] = -(m[0, 0] * m[1, 2] - m[0, 2] * m[1, 0])
                C[2, 2] = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
                
                dA_d_l1 = np.array([C[0, 0] - C[0, 2], C[1, 0] - C[1, 2], C[2, 0] - C[2, 2]])
                dA_d_l2 = np.array([C[0, 1] - C[0, 2], C[1, 1] - C[1, 2], C[2, 1] - C[2, 2]])
                
                dA_d_l1 *= sign
                dA_d_l2 *= sign
                
                grad_acc[i, 0] += w * dA_d_l1[0]
                grad_acc[i, 1] += w * dA_d_l2[0]
                grad_acc[j, 0] += w * dA_d_l1[1]
                grad_acc[j, 1] += w * dA_d_l2[1]
                grad_acc[k, 0] += w * dA_d_l1[2]
                grad_acc[k, 1] += w * dA_d_l2[2]
            
            grad_obj = -grad_acc / t
            return obj_val, grad_obj.flatten()
        
        def objective(params):
            val, _ = compute_areas_and_grads(params)
            return val
        
        def objective_grad(params):
            _, grad = compute_areas_and_grads(params)
            return grad
        
        return objective, objective_grad
    
    def symmetric_init():
        """Create a symmetric initial configuration for 11 points."""
        # Use a known good pattern: 1 center, 3 near midpoints of sides, 6 more
        # Barycentric coordinates (l1, l2) where l3 = 1 - l1 - l2
        bary = np.zeros((N, 2))
        
        # Center
        bary[0] = [1/3, 1/3]
        
        # Three points near the midpoints of each side
        # Side 1: l1=0 (base), midpoint at l2=0.5
        bary[1] = [0.05, 0.5]
        # Side 2: l2=0 (right side), midpoint at l1=0.5
        bary[2] = [0.5, 0.05]
        # Side 3: l3=0 (left side), l1+l2=1, midpoint at l1=0.5, l2=0.5
        bary[3] = [0.5, 0.5]
        
        # Six more points in a hexagonal pattern around center
        r = 0.15
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for i in range(6):
            # Convert from polar to barycentric-like
            # In barycentric, we can place them symmetrically
            angle = angles[i]
            # Place in a ring around center
            bary[4+i, 0] = 1/3 + r * np.cos(angle)
            bary[4+i, 1] = 1/3 + r * np.sin(angle)
        
        # Ensure validity
        bary = np.clip(bary, 0.01, 0.98)
        s = bary[:, 0] + bary[:, 1]
        mask = s > 0.98
        if np.any(mask):
            bary[mask, 0] *= 0.98 / s[mask]
            bary[mask, 1] *= 0.98 / s[mask]
        
        return bary
    
    def grid_init():
        """Create a grid-based initial configuration."""
        bary = np.zeros((N, 2))
        # 3x3 grid with one removed, adjusted
        positions = [
            [0.2, 0.2], [0.5, 0.2], [0.8, 0.2],
            [0.2, 0.5], [0.5, 0.5], [0.8, 0.5],
            [0.2, 0.8], [0.5, 0.8], [0.8, 0.8],
            [0.35, 0.35], [0.65, 0.65]
        ]
        bary = np.array(positions)
        
        bary = np.clip(bary, 0.01, 0.98)
        s = bary[:, 0] + bary[:, 1]
        mask = s > 0.98
        if np.any(mask):
            bary[mask, 0] *= 0.98 / s[mask]
            bary[mask, 1] *= 0.98 / s[mask]
        
        return bary
    
    def random_init(seed):
        """Create a random initial configuration."""
        np.random.seed(seed)
        r = np.random.exponential(1.0, (N, 3))
        r /= r.sum(axis=1, keepdims=True)
        bary = r[:, :2]
        bary += np.random.normal(0, 0.01, (N, 2))
        
        bary = np.clip(bary, 0.01, 0.98)
        s = bary[:, 0] + bary[:, 1]
        mask = s > 0.98
        if np.any(mask):
            bary[mask, 0] *= 0.98 / s[mask]
            bary[mask, 1] *= 0.98 / s[mask]
        
        return bary
    
    best_result = None
    best_score = -np.inf
    
    # Try different initializations with adaptive temperature
    initializations = [
        symmetric_init(),
        grid_init(),
        random_init(42),
        random_init(123),
        random_init(456),
        random_init(789),
        random_init(1024),
        random_init(2048),
    ]
    
    for init_bary in initializations:
        # Start with higher temperature, decrease it
        for temp in [0.05, 0.02, 0.01, 0.005]:
            objective, objective_grad = make_softmin_objective(temp)
            
            try:
                result = minimize(
                    objective,
                    init_bary.flatten(),
                    method='L-BFGS-B',
                    jac=objective_grad,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                final_bary = get_valid_bary(result.x)
                min_area = compute_min_area(final_bary)
                
                if min_area > best_score:
                    best_score = min_area
                    best_result = final_bary.copy()
                    # Use this as new starting point for next temperature
                    init_bary = final_bary.copy()
                    
            except Exception:
                continue
    
    if best_result is None:
        best_result = symmetric_init()
    
    final_bary = best_result
    cart = bary_to_cartesian(final_bary)
    
    # Ensure points are strictly inside
    eps = 1e-6
    cart = np.clip(cart, eps, 1 - eps)
    cart[:, 1] = np.clip(cart[:, 1], eps, np.sqrt(3)/2 - eps)
    
    return cart
# EVOLVE-BLOCK-END
