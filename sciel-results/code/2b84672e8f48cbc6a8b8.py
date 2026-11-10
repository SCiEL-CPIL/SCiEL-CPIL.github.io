import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import lpmv, factorial

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c for degree-10 spherical harmonics minimizing ||f||_4 / ||f||_2.
    Strategy: 
    1. Precompute Spherical Harmonic matrix Y at quadrature points.
    2. Since the objective J(c) = (∫ f^4) / (∫ f^2)^2 is scale invariant, 
       we constrain c to the unit sphere ||c||_2 = 1.
    3. On this sphere, minimizing J(c) is equivalent to minimizing ∫ f^4 dΩ.
    4. We use Riemannian Optimization on the sphere. 
       Specifically, we use scipy's 'trust-constr' method with a custom 
       Riemannian step or simply project L-BFGS-B steps onto the sphere 
       with a retraction. However, a more robust approach for this specific 
       "flattening" problem is to use the fact that the objective is a 
       quartic form. 
       
       Actually, a very effective structural change is to use **Subspace Iteration** 
       or **Power Method** on the associated tensor, but that finds maxima (peaks). 
       We want minima (flatness). 
       
       Let's stick to a robust Riemannian approach:
       We define the objective on the sphere.
       We use `scipy.optimize.minimize` with method='L-BFGS-B' but we 
       explicitly handle the constraint by parameterizing the sphere? No, 
       L-BFGS-B doesn't do that.
       
       Let's use **trust-constr** with the constraint ||c||^2=1. 
       This is a standard way to handle non-linear equality constraints.
       
       To improve upon the seed (which used L-BFGS-B on the ratio), 
       I will:
       1. Use the exact constraint ||c||^2=1.
       2. Provide the exact gradient and Hessian-vector products if possible, 
          or at least a very good gradient.
       3. Use multiple random initializations on the sphere.
       4. Use a "cooling" schedule for the trust region or simply rely on 
          trust-constr's robustness.
          
       Wait, the seed already does L-BFGS-B on the ratio. 
       The "flattest" harmonic is related to the minimum eigenvalue of the 
       operator that maps c to the "f^4" interaction.
       
       A different strategy: **Genetic Algorithm** or **Simulated Annealing**? 
       No, those are stochastic and might be slow.
       
       Let's try **Riemannian Gradient Descent with Adaptive Step Size** 
       implemented manually. This is often faster and more stable for 
       sphere-constrained problems than generic constraint solvers.
       
       Algorithm:
       1. c = random unit vector.
       2. Loop:
          a. Compute f = Y @ c.
          b. Compute grad = 4 * Y^T (W * f^3) - 4 * lambda * c, 
             where lambda = c^T (Y^T (W * f^3)) / ||c||^2 (Lagrange multiplier for ||c||=1).
             Actually, if we normalize c after each step, we just need the 
             Euclidean gradient of the objective on the sphere.
             The Riemannian gradient is the projection of the Euclidean gradient 
             onto the tangent space: grad_r = grad_e - (grad_e . c) * c.
          c. Update c = c - alpha * grad_r.
          d. Retract: c = c / ||c||.
          e. Adaptive alpha: line search or backtracking.
       
       This is structurally different from L-BFGS-B (which is a quasi-Newton 
       method in Euclidean space, often struggling with the sphere constraint 
       if not handled perfectly).
       
       I will implement a robust Riemannian Gradient Descent with backtracking 
       line search, combined with multiple random restarts.
    """
    k = 10
    n_coeffs = 2 * k + 1  # 21
    
    start_time = time.time()
    deadline = start_time + 590
    
    # --- 1. Precompute Spherical Harmonics ---
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    x_grid, phi_grid = np.meshgrid(x_gl, phi_vals, indexing='ij')
    x_flat = x_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    # Effective weights
    W_eff = np.repeat(w_gl, n_phi) * (2.0 * np.pi / n_phi)
    
    # Normalization constants
    N = np.zeros(n_coeffs)
    for m in range(0, k + 1):
        norm_factor = (2 * k + 1) / (4.0 * np.pi) * factorial(k - m) / factorial(k + m)
        N[m] = np.sqrt(norm_factor)
        if m > 0:
            N[-m] = N[m]
            
    # Build Y matrix
    n_points = len(x_flat)
    Y_matrix = np.zeros((n_points, n_coeffs))
    
    for m_idx, m in enumerate(range(-k, k + 1)):
        if m == 0:
            P_val = lpmv(0, k, x_flat)
            Y_matrix[:, m_idx] = N[0] * P_val
        elif m > 0:
            P_val = lpmv(m, k, x_flat)
            Y_matrix[:, m_idx] = np.sqrt(2) * N[m] * P_val * np.cos(m * phi_flat)
        else:
            abs_m = -m
            P_val = lpmv(abs_m, k, x_flat)
            Y_matrix[:, m_idx] = np.sqrt(2) * N[abs_m] * P_val * np.sin(abs_m * phi_flat)
            
    # --- 2. Riemannian Optimization on the Sphere ---
    
    def objective_sphere(c):
        """Objective on the sphere: ∫ f^4 dΩ. Assumes ||c||=1."""
        f = Y_matrix @ c
        return np.dot(W_eff, f**4)
        
    def grad_sphere(c):
        """Euclidean gradient of ∫ f^4. The Riemannian gradient is the projection."""
        f = Y_matrix @ c
        # d/dc ∫ (Yc)^4 = 4 Y^T (W * (Yc)^3)
        return 4.0 * (Y_matrix.T @ (W_eff * f**3))
        
    def riemannian_gradient(c):
        """Project Euclidean gradient to tangent space at c."""
        grad_e = grad_sphere(c)
        # Projection: grad_r = grad_e - (grad_e . c) * c
        proj = np.dot(grad_e, c) * c
        return grad_e - proj
        
    def backtrack_line_search(c, grad_r, obj_val):
        """Backtracking line search for Riemannian gradient descent."""
        alpha = 1.0
        c_prev = c
        obj_prev = obj_val
        
        for _ in range(30):
            c_new = c - alpha * grad_r
            # Retraction: normalize
            norm = np.linalg.norm(c_new)
            if norm < 1e-10:
                return None, None
            c_new = c_new / norm
            
            obj_new = objective_sphere(c_new)
            
            # Armijo condition: f(x - alpha g) <= f(x) - c1 * alpha * ||g||^2
            # c1 is typically 1e-4
            if obj_new <= obj_prev - 1e-4 * alpha * np.dot(grad_r, grad_r):
                return c_new, obj_new
            
            alpha *= 0.5
            
        return None, None

    def optimize_from_start(c0, max_iter=500):
        """Run Riemannian Gradient Descent from a start point."""
        c = c0 / np.linalg.norm(c0)
        obj_val = objective_sphere(c)
        grad_r = riemannian_gradient(c)
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
                
            # If gradient is small, stop
            if np.linalg.norm(grad_r) < 1e-8:
                break
                
            c_new, obj_new = backtrack_line_search(c, grad_r, obj_val)
            
            if c_new is None:
                # If line search fails, try a smaller step or break
                break
                
            c = c_new
            obj_val = obj_new
            grad_r = riemannian_gradient(c)
            
        return c, obj_val

    # --- 3. Multiple Restarts ---
    best_score = np.inf
    best_coeffs = np.ones(n_coeffs) / np.sqrt(n_coeffs)
    
    # Generate diverse starts
    # 1. Random starts
    starts = []
    for _ in range(100):
        v = np.random.randn(n_coeffs)
        v /= np.linalg.norm(v)
        starts.append(v)
        
    # 2. Structured starts: Low frequency dominance
    for m in range(-k, k + 1):
        v = np.zeros(n_coeffs)
        v[m + k] = 1.0
        # Add small noise to break symmetry
        v += 0.01 * np.random.randn(n_coeffs)
        v /= np.linalg.norm(v)
        starts.append(v)
        
    # 3. Pairs
    for m in range(1, k + 1):
        v = np.zeros(n_coeffs)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        v += 0.01 * np.random.randn(n_coeffs)
        v /= np.linalg.norm(v)
        starts.append(v)
        
        v = np.zeros(n_coeffs)
        v[m + k] = 1.0
        v[-m + k] = -1.0
        v += 0.01 * np.random.randn(n_coeffs)
        v /= np.linalg.norm(v)
        starts.append(v)

    np.random.shuffle(starts)
    
    for c0 in starts:
        if time.time() > deadline:
            break
            
        c_res, score = optimize_from_start(c0)
        
        # Evaluate final score (L4/L2 ratio squared, since ||c||=1 => ||f||_2^2 = 1)
        # Wait, the task asks for R = ||f||_4 / ||f||_2.
        # If ||c||=1, then ||f||_2^2 = 1, so ||f||_2 = 1.
        # Then R = ||f||_4.
        # The objective we minimized is ∫ f^4 dΩ = ||f||_4^4.
        # So minimizing the objective is equivalent to minimizing R^4, which is minimizing R.
        
        if score < best_score:
            best_score = score
            best_coeffs = c_res.copy()
            
    # --- 4. Final Validation and Scaling ---
    # The problem allows any scale. We return the coefficients.
    # The evaluator computes R = ||f||_4 / ||f||_2. This is scale invariant.
    # However, the constraints on c are: 1e-3 <= max|c| <= 1e5.
    # Our c is unit norm. max|c| for unit norm in 21D is between 1/sqrt(21) and 1.
    # 1/sqrt(21) ~ 0.218 > 1e-3. So we are safe.
    
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(n_coeffs)
        
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3:
        best_coeffs *= (1.0 / max_c)
    elif max_c > 1e5:
        best_coeffs /= max_c
        
    return best_coeffs
