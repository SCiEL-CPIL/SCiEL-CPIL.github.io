import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import lpmv, factorial
from scipy.linalg import eigh

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c for degree-10 spherical harmonics minimizing ||f||_4 / ||f||_2.
    
    Strategy:
    1. Precompute Spherical Harmonic matrix Y at quadrature points.
    2. The objective R(c) = ||f||_4 / ||f||_2 is scale invariant. We fix ||c||_2 = 1.
       Then R(c) = (∫ f^4 dΩ)^(1/4) because ∫ f^2 dΩ = 1.
       Minimizing R is equivalent to minimizing J(c) = ∫ f^4 dΩ on the unit sphere.
    3. Instead of simple gradient descent, we use a Riemannian Trust-Region method 
       (implemented via scipy's trust-constr or a manual Newton step with projection).
       However, scipy's trust-constr can be slow. 
       
       A faster structural change: Use the **Riemannian Newton Method**.
       The Riemannian Hessian of J(c) = (Yc)^T W (Yc)^3 is complex, but we can use
       a quasi-Newton approach (L-BFGS) on the sphere, or better, exploit the 
       fact that J(c) is a quartic form.
       
       Actually, the most robust and fast method for this specific "flatness" problem 
       (minimizing L4 norm) is to recognize that we are looking for the eigenvector 
       corresponding to the **smallest** generalized eigenvalue of a certain operator? 
       No, J(c) = c^T T c where T is a 4th order tensor. This is not a quadratic form.
       
       Let's use **Riemannian Gradient Descent with Momentum** (Polyak's heavy ball).
       This is structurally different from plain GD and often much faster for 
       non-convex quartics. We will also use a **multi-start** strategy with 
       **coarse-to-fine** grid search in the low-dimensional subspace of dominant modes 
       if needed, but random restarts with momentum should be sufficient.
       
       Crucial improvement: Use **L-BFGS-B** on a reparameterization of the sphere 
       (e.g., using the stereographic projection or exponential map) to avoid 
       constraint handling issues, OR use **trust-constr** which handles 
       equality constraints well.
       
       Given the time limit, I will implement a **Riemannian L-BFGS** style 
       optimization using `scipy.optimize.minimize` with method='L-BFGS-B' 
       on the **stereographic projection** of the sphere. This removes the 
       equality constraint and allows L-BFGS to work in Euclidean space, 
       which is often faster and more robust than manual retraction.
       
       Stereographic projection from the north pole:
       z = (c_2, ..., c_n) / (1 - c_1)  (mapping S^{n-1} \\ {N} -> R^{n-1})
       Inverse: c_1 = (1 - ||z||^2) / (1 + ||z||^2)
                c_i = 2 z_i / (1 + ||z||^2) for i > 1
       
       This is a diffeomorphism from R^{n-1} to the sphere minus one point.
       We optimize in z-space.
    """
    k = 10
    n_coeffs = 2 * k + 1  # 21
    
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
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
            
    # --- 2. Optimization via Stereographic Projection ---
    
    def z_to_c(z):
        """Map z in R^{n-1} to c in S^{n-1} via stereographic projection."""
        z_norm_sq = np.dot(z, z)
        denom = 1.0 + z_norm_sq
        c1 = (1.0 - z_norm_sq) / denom
        c_rest = 2.0 * z / denom
        c = np.zeros(n_coeffs)
        c[0] = c1
        c[1:] = c_rest
        return c

    def c_to_z(c):
        """Map c in S^{n-1} to z in R^{n-1}."""
        # Handle case where c[0] is close to 1 (north pole)
        if c[0] > 0.9999:
            return np.zeros(n_coeffs - 1)
        z = c[1:] / (1.0 - c[0])
        return z

    def objective_z(z):
        """Objective in z-space: ∫ f^4 dΩ."""
        c = z_to_c(z)
        f = Y_matrix @ c
        return np.dot(W_eff, f**4)

    def gradient_z(z):
        """Gradient of objective in z-space."""
        c = z_to_c(z)
        f = Y_matrix @ c
        # Euclidean gradient of J(c)
        grad_c = 4.0 * (Y_matrix.T @ (W_eff * f**3))
        
        # Jacobian of z_to_c:
        # c1 = (1 - ||z||^2) / (1 + ||z||^2)
        # ci = 2 zi / (1 + ||z||^2)
        # 
        # dc1/dzj = -4 zj / (1 + ||z||^2)^2
        # dci/dzj = 2 (delta_ij (1 + ||z||^2) - 2 zi zj) / (1 + ||z||^2)^2
        #
        # grad_z = (dc/dz)^T grad_c
        # grad_z[j] = sum_i grad_c[i] * (dc_i/dz_j)
        
        z_norm_sq = np.dot(z, z)
        denom_sq = (1.0 + z_norm_sq)**2
        
        # Term for c1
        term_c1 = grad_c[0] * (-4.0 * z / denom_sq)
        
        # Terms for c_i (i>0)
        # grad_z[j] += sum_{i>0} grad_c[i] * 2 * (delta_ij (1 + z_norm_sq) - 2 z_i z_j) / denom_sq
        # = 2 / denom_sq * [ (1 + z_norm_sq) grad_c[j+1] - 2 z_j (z^T grad_c_rest) ]
        
        grad_c_rest = grad_c[1:]
        z_dot_grad_rest = np.dot(z, grad_c_rest)
        
        term_rest = 2.0 / denom_sq * ((1.0 + z_norm_sq) * grad_c_rest - 2.0 * z * z_dot_grad_rest)
        
        return term_c1 + term_rest

    # --- 3. Multiple Restarts with L-BFGS-B ---
    
    best_score = np.inf
    best_coeffs = np.ones(n_coeffs) / np.sqrt(n_coeffs)
    
    # Generate diverse starts
    starts_z = []
    
    # 1. Random starts in z-space
    np.random.seed(42)
    for _ in range(50):
        z0 = np.random.randn(n_coeffs - 1) * 0.5
        starts_z.append(z0)
        
    # 2. Structured starts: Map known good c vectors to z
    # Single modes
    for m in range(-k, k + 1):
        v = np.zeros(n_coeffs)
        v[m + k] = 1.0
        # Avoid north pole if m==0 (index 0 is m=-10? No, index 0 is m=-10)
        # Our c[0] corresponds to m=-10.
        # If we want m=0 (index 10), c[10]=1. c[0]=0. z is well-defined.
        # If we want m=-10 (index 0), c[0]=1. This is the north pole! z -> inf.
        # So we must be careful. We'll add noise to avoid exact poles.
        v += 0.01 * np.random.randn(n_coeffs)
        v /= np.linalg.norm(v)
        if v[0] < 0.9999:
            z0 = c_to_z(v)
            starts_z.append(z0)
            
    # 3. Pairs
    for m in range(1, k + 1):
        v = np.zeros(n_coeffs)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        v += 0.01 * np.random.randn(n_coeffs)
        v /= np.linalg.norm(v)
        if v[0] < 0.9999:
            z0 = c_to_z(v)
            starts_z.append(z0)
            
        v = np.zeros(n_coeffs)
        v[m + k] = 1.0
        v[-m + k] = -1.0
        v += 0.01 * np.random.randn(n_coeffs)
        v /= np.linalg.norm(v)
        if v[0] < 0.9999:
            z0 = c_to_z(v)
            starts_z.append(z0)

    np.random.shuffle(starts_z)
    
    for z0 in starts_z:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective_z, z0, jac=gradient_z, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < best_score:
                c_res = z_to_c(res.x)
                # Normalize to be safe
                c_res /= np.linalg.norm(c_res)
                score = objective_z(c_to_z(c_res))
                
                if score < best_score:
                    best_score = score
                    best_coeffs = c_res.copy()
                    
        except Exception:
            continue

    # --- 4. Final Validation and Scaling ---
    
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(n_coeffs)
        
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3:
        best_coeffs *= (1.0 / max_c)
    elif max_c > 1e5:
        best_coeffs /= max_c
        
    return best_coeffs
