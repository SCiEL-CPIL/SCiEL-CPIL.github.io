"""Spectral energy minimization for covering the sphere with 130 points."""
import numpy as np
from scipy.optimize import minimize

N = 130
L_MAX = 20

# EVOLVE-BLOCK-START
def _legendre_and_derivatives(dots, l_max):
    """
    Compute Legendre polynomials P_l(dots) and their derivatives P'_l(dots)
    for l=0 to l_max.
    dots: array of shape (N, N) containing x_i . x_j
    Returns:
        P: array of shape (l_max+1, N, N)
        dP: array of shape (l_max+1, N, N)
    """
    N_pts = dots.shape[0]
    P = np.zeros((l_max + 1, N_pts, N_pts))
    dP = np.zeros((l_max + 1, N_pts, N_pts))
    
    # P_0(x) = 1, P'_0(x) = 0
    P[0] = 1.0
    dP[0] = 0.0
    
    if l_max >= 1:
        # P_1(x) = x, P'_1(x) = 1
        P[1] = dots
        dP[1] = 1.0
        
        for l in range(1, l_max):
            # Recurrence: (l+1) P_{l+1}(x) = (2l+1) x P_l(x) - l P_{l-1}(x)
            # P_{l+1} = ((2l+1) x P_l - l P_{l-1}) / (l+1)
            P[l+1] = ((2*l + 1) * dots * P[l] - l * P[l-1]) / (l + 1)
            
            # Derivative recurrence:
            # P'_l(x) = l (P_{l-1}(x) - x P_l(x)) / (1 - x^2)  (for x != +/- 1)
            # Or use: P'_l(x) = (l+1) P_{l+1}(x) - (2l+1) P_l(x) ... no that's not right.
            # Standard derivative recurrence:
            # (1-x^2) P'_l(x) = l P_{l-1}(x) - l x P_l(x)
            # So P'_l(x) = l (P_{l-1}(x) - x P_l(x)) / (1 - x^2)
            
            # We need to handle x close to +/- 1 carefully.
            # For x=1, P'_l(1) = l(l+1)/2
            # For x=-1, P'_l(-1) = (-1)^{l+1} l(l+1)/2
            
            denom = 1.0 - dots * dots
            # Avoid division by zero
            mask = np.abs(denom) > 1e-12
            dP[l+1] = 0.0 # Initialize
            
            # Calculate for valid indices
            if np.any(mask):
                dP[l+1][mask] = (l + 1) * (P[l][mask] - dots[mask] * P[l+1][mask]) / denom[mask]
                # Wait, the formula is (l+1) P_{l+1} - (2l+1) P_l = (1-x^2) P'_l ? No.
                # Identity: (1-x^2) P'_l(x) = -l x P_l(x) + l P_{l-1}(x)
                # So P'_l(x) = l (P_{l-1}(x) - x P_l(x)) / (1-x^2)
                # This calculates P'_l. We are in loop for l, calculating P[l+1].
                # We want dP[l+1] which is P'_{l+1}.
                # So we need P'_{l+1} = (l+1) (P_l - x P_{l+1}) / (1-x^2)
                
                dP[l+1][mask] = (l + 1) * (P[l][mask] - dots[mask] * P[l+1][mask]) / denom[mask]
            
            # Handle boundaries where denom is 0
            # x = 1
            mask_1 = (dots > 1 - 1e-12) & (dots <= 1 + 1e-12)
            if np.any(mask_1):
                dP[l+1][mask_1] = (l + 1) * (l + 2) / 2.0
            
            # x = -1
            mask_m1 = (dots < -1 + 1e-12) & (dots >= -1 - 1e-12)
            if np.any(mask_m1):
                dP[l+1][mask_m1] = (-1)**(l + 2) * (l + 1) * (l + 2) / 2.0
                
    return P, dP

def _spectral_energy_and_grad(x):
    """
    Compute spectral energy and gradient with respect to x (shape 3N).
    x: flattened array of point coordinates.
    """
    pts = x.reshape(N, 3)
    # Normalize points to ensure they are on the sphere (for numerical stability during optimization)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pts = pts / norms
    
    dots = pts @ pts.T
    np.clip(dots, -1.0, 1.0, out=dots)
    
    P, dP = _legendre_and_derivatives(dots, L_MAX)
    
    # Compute S_l = sum_{i,j} P_l(x_i . x_j)
    # S_l is a scalar for each l
    S = np.sum(P, axis=(1, 2))
    
    # Energy E = sum_{l=1}^{L_MAX} w_l * (S_l / N)^2
    # We use weights w_l = 1/l to emphasize low frequencies (global uniformity)
    weights = 1.0 / np.arange(1, L_MAX + 1)
    
    energy = 0.0
    grad = np.zeros_like(x)
    
    for l in range(1, L_MAX + 1):
        w = weights[l-1]
        Sl = S[l]
        # dE/dx_k = 2 * w * (Sl / N^2) * d(Sl)/dx_k
        # d(Sl)/dx_k = sum_j dP_l(x_k . x_j)/dx_k + sum_i dP_l(x_i . x_k)/dx_k
        # As derived: 2 * sum_{j} dP_l(x_k . x_j) x_j  (including j=k term which is 0 for derivative of dot product? No)
        # Let's re-verify: S_l = sum_i sum_j P_l(x_i . x_j)
        # d S_l / d x_k = sum_j P'_l(x_k . x_j) x_j + sum_i P'_l(x_i . x_k) x_i
        # By symmetry, sum_i P'_l(x_i . x_k) x_i = sum_j P'_l(x_k . x_j) x_j (renaming i to j)
        # So d S_l / d x_k = 2 * sum_j P'_l(x_k . x_j) x_j
        
        # We need to compute this for all k.
        # grad_l[k] = 2 * sum_j dP_l[k, j] * pts[j]
        
        # dP_l is shape (N, N). pts is shape (N, 3).
        # grad_l is shape (N, 3).
        # grad_l = 2 * dP_l @ pts
        
        grad_l = 2.0 * (dP[l] @ pts)
        
        energy += w * (Sl / N)**2
        
        # dE/dx_k = 2 * w * (Sl / N^2) * grad_l[k]
        # Note: S_l is sum of N^2 terms. The definition of C_lm usually has 1/N.
        # Let's stick to E = sum w_l (S_l)^2. The scaling factor 1/N^2 is constant and doesn't change the minimizer.
        # However, for gradient magnitude stability, we might want to scale.
        # Let's use E = sum w_l (S_l / N)^2.
        # dE/dx = 2 * sum w_l (S_l/N) * (1/N) * dS_l/dx = 2/N * sum w_l S_l dS_l/dx
        
        grad += (2.0 * w * Sl / N) * grad_l
        
    # Project gradient to tangent space? 
    # Since we are optimizing in R^3N without constraints, the optimizer will move points off the sphere.
    # It is better to use a constrained optimizer or re-normalize in the objective.
    # If we re-normalize in the objective, the gradient calculated here is with respect to the *normalized* points.
    # The chain rule requires projecting the gradient back to the ambient space R^3N.
    # grad_ambient = grad_tangent - (grad_tangent . x_k) x_k
    # Actually, if we define f(x) = E(x/|x|), then grad f(x) = (I - xx^T) grad E(x/|x|).
    
    grad_ambient = np.zeros_like(x).reshape(N, 3)
    for k in range(N):
        # Project grad[k] to tangent space of sphere at pts[k]
        proj = grad[k] - np.dot(grad[k], pts[k]) * pts[k]
        grad_ambient[k] = proj
        
    return energy, grad_ambient.reshape(-1)

def _refine_spectral(x0):
    """
    Refine points using L-BFGS-B on spectral energy.
    """
    x0 = np.array(x0, dtype=float)
    
    # Ensure initial points are on sphere
    norms = np.linalg.norm(x0.reshape(N, 3), axis=1, keepdims=True)
    x0 = x0 / norms
    
    # L-BFGS-B doesn't handle constraints easily, so we rely on the fact that 
    # the spectral energy is rotationally invariant and the gradient projection 
    # keeps points near the sphere if we start there. 
    # However, L-BFGS-B might drift. 
    # A safer approach: use 'Powell' or 'Nelder-Mead' on (theta, phi) if gradient is too complex.
    # But gradient is computed. Let's try L-BFGS-B with a line search that projects?
    # scipy.optimize.minimize doesn't support custom line search projection easily.
    
    # Alternative: Use 'CG' or 'BFGS' with a wrapper that normalizes? No, that breaks the gradient.
    
    # Let's use 'L-BFGS-B' but monitor the norm. If points drift, we fail.
    # Better: Parameterize by (theta, phi) and use the Jacobian of the spectral energy w.r.t (theta, phi).
    # This is safer.
    
    # Convert to theta, phi
    pts = x0.reshape(N, 3)
    theta = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    
    def obj_theta_phi(x):
        t = x[:N]
        p = x[N:]
        pts = np.column_stack([
            np.sin(t) * np.cos(p),
            np.sin(t) * np.sin(p),
            np.cos(t)
        ])
        e, _ = _spectral_energy_and_grad(pts.flatten())
        return e

    def jac_theta_phi(x):
        t = x[:N]
        p = x[N:]
        pts = np.column_stack([
            np.sin(t) * np.cos(p),
            np.sin(t) * np.sin(p),
            np.cos(t)
        ])
        _, grad_ambient = _spectral_energy_and_grad(pts.flatten())
        grad_ambient = grad_ambient.reshape(N, 3)
        
        # Compute Jacobian of transformation from (theta, phi) to Cartesian
        # x = sin(t) cos(p), y = sin(t) sin(p), z = cos(t)
        # dx/dt = cos(t) cos(p), dy/dt = cos(t) sin(p), dz/dt = -sin(t)
        # dx/dp = -sin(t) sin(p), dy/dp = sin(t) cos(p), dz/dp = 0
        
        jac = np.zeros_like(x)
        
        for i in range(N):
            g = grad_ambient[i]
            # Gradient w.r.t theta_i
            jac[i] = g[0] * np.cos(t[i]) * np.cos(p[i]) + g[1] * np.cos(t[i]) * np.sin(p[i]) - g[2] * np.sin(t[i])
            # Gradient w.r.t phi_i
            jac[N + i] = g[0] * (-np.sin(t[i]) * np.sin(p[i])) + g[1] * (np.sin(t[i]) * np.cos(p[i]))
            
        return jac

    # Optimize
    result = minimize(obj_theta_phi, x0=np.concatenate([theta, phi]), 
                      jac=jac_theta_phi, method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
    
    t = result.x[:N]
    p = result.x[N:]
    pts = np.column_stack([
        np.sin(t) * np.cos(p),
        np.sin(t) * np.sin(p),
        np.cos(t)
    ])
    return pts, result.fun

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    start_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Refine using spectral gradient descent
    refined_pts, energy = _refine_spectral(start_pts)
    
    # Normalize
    norms = np.linalg.norm(refined_pts, axis=1, keepdims=True)
    refined_pts = refined_pts / norms
    
    return refined_pts
# EVOLVE-BLOCK-END
