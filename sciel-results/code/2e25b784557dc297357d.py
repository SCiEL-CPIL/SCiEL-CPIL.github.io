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

    def compute_areas_and_grads(params):
        """
        Compute soft-min objective and its analytic gradient.
        Returns: (objective_value, gradient_array)
        """
        bary = get_valid_bary(params)
        
        # Build full barycentric matrix (N, 3)
        l3 = 1.0 - bary[:, 0] - bary[:, 1]
        bary_full = np.column_stack([bary[:, 0], bary[:, 1], l3])
        
        areas = np.zeros(n_tris)
        # We will accumulate gradients for each point's l1 and l2
        # grad_acc[i, 0] is dObj/dl1_i, grad_acc[i, 1] is dObj/dl2_i
        grad_acc = np.zeros((N, 2))
        
        t = 0.01
        # First pass: compute areas and weights
        exp_terms = np.zeros(n_tris)
        max_exp = -np.inf
        for idx, (i, j, k) in enumerate(tri_indices):
            M = bary_full[[i, j, k], :]
            det_val = np.linalg.det(M)
            area_val = abs(det_val)
            # Avoid log(0) issues
            area_val = max(area_val, 1e-10)
            areas[idx] = area_val
            exp_term = -area_val / t
            exp_terms[idx] = exp_term
            max_exp = max(max_exp, exp_term)
            
        # Normalize exp terms for stability
        exp_terms_shifted = np.exp(exp_terms - max_exp)
        sum_exp = np.sum(exp_terms_shifted)
        weights = exp_terms_shifted / sum_exp
        
        # Soft min value
        soft_min = -t * (max_exp + np.log(sum_exp))
        obj_val = -soft_min
        
        # Second pass: compute gradients
        # d(soft_min)/dx = (1/t) * sum_k w_k * d(A_k)/dx
        # d(obj_val)/dx = -d(soft_min)/dx = -(1/t) * sum_k w_k * d(A_k)/dx
        # Note: A_k = |det(M_k)|. Assuming det(M_k) > 0 for simplicity in gradient direction?
        # Actually, we should handle sign. But for max-min, we care about magnitude.
        # If det is negative, d|det|/dx = -d(det)/dx.
        # Let's use the signed determinant for gradient calculation and multiply by sign.
        
        for idx, (i, j, k) in enumerate(tri_indices):
            w = weights[idx]
            if w < 1e-8: continue # Skip negligible contributions
            
            M = bary_full[[i, j, k], :]
            det_val = np.linalg.det(M)
            sign = 1.0 if det_val > 0 else -1.0
            
            # Cofactor matrix / Adjugate
            # adj(M) = det(M) * inv(M)^T
            # But if singular, inv fails.
            # Use np.linalg.det and manual cofactor calculation?
            # Or use the formula: d(det)/dx_ij = C_ij (cofactor)
            # C_ij = (-1)^(i+j) * det(M_minor_ij)
            
            # Let's compute adjugate explicitly for 3x3
            # adj(M)_{ij} = C_{ji}
            # We need gradients wrt l1, l2 of points i, j, k.
            # For point p in {i,j,k}:
            # dA/dl1_p = adj(M)_{p,0} - adj(M)_{p,2}
            # dA/dl2_p = adj(M)_{p,1} - adj(M)_{p,2}
            
            # Calculate adjugate of M
            # M = [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]]
            # C00 = m11*m22 - m12*m21
            # C01 = -(m10*m22 - m12*m20)
            # C02 = m10*m21 - m11*m20
            # ...
            
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
            
            # adj(M) is transpose of C
            # adj(M)_{p, q} = C_{q, p}
            
            # Gradient for point i (row 0 of M)
            # dA/dl1_i = C_{0, 0} - C_{2, 0}  (Wait, adj(M)_{0,0} = C_{0,0}, adj(M)_{0,2} = C_{2,0})
            # Formula: dA/dl1_p = adj(M)_{p,0} - adj(M)_{p,2} = C_{0,p} - C_{2,p}
            # Let's re-verify:
            # d(det)/dl1_p = C_{p,0} - C_{p,2} ?
            # det = sum_j m_{0,j} C_{0,j} ...
            # If we change l1 of point p (row p), m_{p,0} changes by +1, m_{p,2} by -1.
            # d(det) = C_{p,0} * (+1) + C_{p,2} * (-1) = C_{p,0} - C_{p,2}.
            # Yes.
            
            dA_d_l1 = np.array([C[0, 0] - C[0, 2], C[1, 0] - C[1, 2], C[2, 0] - C[2, 2]])
            dA_d_l2 = np.array([C[0, 1] - C[0, 2], C[1, 1] - C[1, 2], C[2, 1] - C[2, 2]])
            
            # Multiply by sign to get d|det|/dx
            dA_d_l1 *= sign
            dA_d_l2 *= sign
            
            # Accumulate to grad_acc
            # Points i, j, k correspond to rows 0, 1, 2 of M
            grad_acc[i, 0] += w * dA_d_l1[0]
            grad_acc[i, 1] += w * dA_d_l2[0]
            grad_acc[j, 0] += w * dA_d_l1[1]
            grad_acc[j, 1] += w * dA_d_l2[1]
            grad_acc[k, 0] += w * dA_d_l1[2]
            grad_acc[k, 1] += w * dA_d_l2[2]
            
        # Total gradient for objective (which is -soft_min)
        # grad_obj = -(1/t) * grad_acc
        grad_obj = -grad_acc / t
        grad_obj_flat = grad_obj.flatten()
        
        return obj_val, grad_obj_flat

    def objective(params):
        val, _ = compute_areas_and_grads(params)
        return val

    def objective_grad(params):
        _, grad = compute_areas_and_grads(params)
        return grad

    best_result = None
    best_score = -np.inf
    
    # Multi-start strategy
    np.random.seed(42)
    for restart in range(20):
        # Random initialization inside the triangle
        # Generate random points in barycentric coords
        # u1, u2, u3 ~ Exp(1), normalize
        r = np.random.exponential(1.0, (N, 3))
        r /= r.sum(axis=1, keepdims=True)
        initial_bary = r[:, :2]
        
        # Add a bit of jitter to avoid exact duplicates or symmetry
        initial_bary += np.random.normal(0, 0.01, (N, 2))
        
        # Ensure validity
        initial_bary = np.clip(initial_bary, 0.01, 0.98)
        s = initial_bary[:, 0] + initial_bary[:, 1]
        mask = s > 0.98
        if np.any(mask):
            initial_bary[mask, 0] *= 0.98 / s[mask]
            initial_bary[mask, 1] *= 0.98 / s[mask]
        
        try:
            result = minimize(
                objective,
                initial_bary.flatten(),
                method='L-BFGS-B',
                jac=objective_grad,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Evaluate final score
            final_params = result.x
            final_bary = get_valid_bary(final_params)
            
            # Calculate actual min area for scoring
            l3 = 1.0 - final_bary[:, 0] - final_bary[:, 1]
            bary_full = np.column_stack([final_bary[:, 0], final_bary[:, 1], l3])
            min_area = np.inf
            for i, j, k in tri_indices:
                M = bary_full[[i, j, k], :]
                det_val = abs(np.linalg.det(M))
                if det_val < min_area:
                    min_area = det_val
            
            # The evaluator measures "raw value" which is the min area ratio?
            # Task: "maximize the smallest area... The evaluator measures the raw value"
            # The current program returns cart coords. The evaluator likely computes the min area from cart coords.
            # So we want to maximize min_area.
            # Our objective was minimizing -soft_min, which approximates maximizing min_area.
            
            if min_area > best_score:
                best_score = min_area
                best_result = final_bary.copy()
                
        except Exception as e:
            continue

    if best_result is None:
        # Fallback to simple grid
        best_result = np.array([
            [0.5, 0.5],
            [0.33, 0.33],
            [0.1, 0.1],
            [0.8, 0.1],
            [0.1, 0.8],
            [0.5, 0.2],
            [0.2, 0.5],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.5, 0.7],
            [0.7, 0.5]
        ])
        
    final_bary = best_result
    cart = bary_to_cartesian(final_bary)
    
    # Ensure points are strictly inside
    eps = 1e-6
    cart = np.clip(cart, eps, 1 - eps)
    cart[:, 1] = np.clip(cart[:, 1], eps, np.sqrt(3)/2 - eps)
    
    return cart
# EVOLVE-BLOCK-END
