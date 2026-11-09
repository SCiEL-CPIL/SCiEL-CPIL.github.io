# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize

def _build_quadrature():
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    # Gauss-Legendre nodes and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(70)
    # Equally spaced phi in [0, 2*pi)
    nphi = 130
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    dphi = 2 * np.pi / nphi
    return x, w, phi, dphi

def _compute_Y(x, phi, m):
    """
    Compute real orthonormal degree-10 spherical harmonics Y_m.
    x = cos(theta), phi = azimuthal angle.
    Y_0 = N_0 * P_10^0(x)
    Y_m = sqrt(2) * N_m * P_10^m(x) * cos(m*phi)  for m > 0
    Y_m = sqrt(2) * N_|m| * P_10^{|m|}(x) * sin(|m|*phi)  for m < 0
    N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    """
    k = 10
    am = abs(m)
    # Condon-Shortley phase is included in lpmv
    P = lpmv(am, k, x)
    # Normalization
    # N_m = sqrt(21/(4*pi) * (10-am)!/(10+am)!)
    # Use log to avoid overflow/underflow
    log_num = np.log(21.0) - np.log(4.0 * np.pi)
    # (10-am)! / (10+am)! = 1 / product_{j=11-am}^{10+am} j
    # For am=0: 1
    # For am>0: product from (10-am+1) to (10+am)
    if am == 0:
        ratio = 1.0
    else:
        # (10-am)!/(10+am)! = 1 / ((10-am+1)*(10-am+2)*...*(10+am))
        log_ratio = 0.0
        for j in range(11 - am, 11 + am):
            log_ratio += np.log(j)
        ratio = np.exp(-log_ratio)
    N = np.sqrt(log_num + np.log(ratio))
    # Actually N = sqrt(21/(4*pi) * ratio)
    N = np.sqrt(21.0 / (4.0 * np.pi) * ratio)
    
    if m == 0:
        return N * P
    elif m > 0:
        return np.sqrt(2.0) * N * P * np.cos(m * phi)
    else:
        return np.sqrt(2.0) * N * P * np.sin(am * phi)

def _precompute_basis(x, phi):
    """
    Precompute all 21 basis functions Y_m evaluated at all quadrature points.
    Returns array of shape (n_points, 21) where n_points = 70*130.
    """
    nx = len(x)
    nphi = len(phi)
    npts = nx * nphi
    
    # Build full grid
    X, PHI = np.meshgrid(x, phi, indexing='ij')
    X_flat = X.flatten()
    PHI_flat = PHI.flatten()
    
    basis = np.zeros((npts, 21))
    for i, m in enumerate(range(-10, 11)):
        basis[:, i] = _compute_Y(X_flat, PHI_flat, m)
    
    return basis, X_flat, PHI_flat

def _objective(coeffs, basis, weights):
    """
    Compute ||f||_4 / ||f||_2 where f = sum c_m Y_m.
    weights are the quadrature weights (including dphi).
    """
    f = basis @ coeffs  # shape (npts,)
    # ||f||_2^2 = sum w_i * f_i^2
    # ||f||_4^4 = sum w_i * f_i^4
    f2 = f ** 2
    f4 = f2 ** 2
    norm2_sq = np.sum(weights * f2)
    norm4_4 = np.sum(weights * f4)
    
    if norm2_sq <= 0:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    return norm4 / norm2

def _objective_grad(coeffs, basis, weights):
    """
    Gradient of ||f||_4 / ||f||_2 with respect to coeffs.
    """
    f = basis @ coeffs
    f2 = f ** 2
    f4 = f2 ** 2
    
    norm2_sq = np.sum(weights * f2)
    norm4_4 = np.sum(weights * f4)
    
    if norm2_sq <= 0:
        return np.zeros_like(coeffs)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    # d(norm4)/dc = (1/(4*norm4^3)) * sum w_i * 4*f_i^3 * Y_i = (1/norm4^3) * sum w_i * f_i^3 * Y_i
    # d(norm2)/dc = (1/norm2) * sum w_i * f_i * Y_i
    # d(norm4/norm2)/dc = (1/norm2) * d(norm4)/dc - (norm4/norm2^2) * d(norm2)/dc
    
    # sum w_i * f_i^3 * Y_i for each m
    grad_norm4 = np.zeros(21)
    grad_norm2 = np.zeros(21)
    
    wf3 = weights * f ** 3
    wf = weights * f
    
    for m in range(21):
        grad_norm4[m] = np.sum(wf3 * basis[:, m])
        grad_norm2[m] = np.sum(wf * basis[:, m])
    
    # d(norm4)/dc = grad_norm4 / norm4^3
    # d(norm2)/dc = grad_norm2 / norm2
    dnorm4 = grad_norm4 / (norm4 ** 3)
    dnorm2 = grad_norm2 / norm2
    
    grad = dnorm4 / norm2 - norm4 * dnorm2 / (norm2 ** 2)
    grad = grad / norm2  # Wait, let me redo this
    
    # R = norm4 / norm2
    # dR/dc = (dnorm4/dc * norm2 - norm4 * dnorm2/dc) / norm2^2
    grad = (dnorm4 * norm2 - norm4 * dnorm2) / (norm2 ** 2)
    
    return grad

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c_m for m=-10..10 that minimize ||f||_4/||f||_2
    for f = sum c_m Y_m^10 (real orthonormal degree-10 spherical harmonics).
    """
    t0 = time.time()
    deadline = t0 + 620  # safety margin
    
    # Build quadrature and basis
    x, w, phi, dphi = _build_quadrature()
    basis, X_flat, PHI_flat = _precompute_basis(x, phi)
    
    # Quadrature weights: w_i * dphi for each (theta, phi) point
    # x has 70 points, phi has 130 points
    # The weight for point (i, j) is w[i] * dphi
    w_expanded = np.tile(w, 130)  # each x-weight repeated for all phi
    weights = w_expanded * dphi
    
    # Initial guess: try several starting points
    best_coeffs = None
    best_val = np.inf
    
    # Strategy: use symmetry. The optimal solution likely has certain symmetries.
    # Try: all coefficients zero except one (to get a starting point)
    # Or use a random start and optimize
    
    # First, let's try the "seed" all-ones and see
    c0 = np.ones(21, dtype=np.float64)
    val0 = _objective(c0, basis, weights)
    best_coeffs = c0.copy()
    best_val = val0
    
    # Try several random initializations and gradient descent
    np.random.seed(42)
    
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization with some structure
        if restart == 0:
            c = c0.copy()
        elif restart == 1:
            # Only even m
            c = np.zeros(21)
            c[5] = 1.0  # m=0
        elif restart == 2:
            # Only m=0
            c = np.zeros(21)
            c[10] = 1.0  # m=0 is index 10
        elif restart == 3:
            # Only |m|=10
            c = np.zeros(21)
            c[0] = 1.0   # m=-10
            c[20] = 1.0  # m=10
        elif restart == 4:
            # Only |m|=8
            c = np.zeros(21)
            c[2] = 1.0   # m=-8
            c[18] = 1.0  # m=8
        elif restart == 5:
            # Only |m|=6
            c = np.zeros(21)
            c[4] = 1.0   # m=-6
            c[16] = 1.0  # m=6
        elif restart == 6:
            # Only |m|=4
            c = np.zeros(21)
            c[6] = 1.0   # m=-4
            c[14] = 1.0  # m=4
        elif restart == 7:
            # Only |m|=2
            c = np.zeros(21)
            c[8] = 1.0   # m=-2
            c[12] = 1.0  # m=2
        elif restart == 8:
            # Random
            c = np.random.randn(21)
        else:
            # Random with sign symmetry: c[-m] = c[m] for cosine part
            c = np.zeros(21)
            for m in range(1, 11):
                val = np.random.randn()
                c[10 - m] = val  # m = -m, sin part
                c[10 + m] = val  # m = +m, cos part
            c[10] = np.random.randn()
        
        # Normalize
        norm = np.linalg.norm(c)
        if norm > 0:
            c = c / norm
        
        # Optimize
        try:
            result = minimize(
                _objective,
                c,
                args=(basis, weights),
                jac=_objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if result.success or result.fun < best_val:
                c_opt = result.x
                val_opt = result.fun
                if val_opt < best_val:
                    best_val = val_opt
                    best_coeffs = c_opt.copy()
        except Exception:
            pass
    
    # Additional refinement: try more restarts if time permits
    n_extra = 20
    for restart in range(n_extra):
        if time.time() > deadline:
            break
        
        c = np.random.randn(21)
        # Add some structured components
        for m in range(1, 11):
            if np.random.rand() < 0.3:
                c[10 - m] = 0
                c[10 + m] = 0
        
        norm = np.linalg.norm(c)
        if norm > 0:
            c = c / norm
        
        try:
            result = minimize(
                _objective,
                c,
                args=(basis, weights),
                jac=_objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 50}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Final check: ensure validity
    if best_coeffs is None:
        best_coeffs = np.ones(21, dtype=np.float64)
    
    # Ensure no NaN or inf
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21, dtype=np.float64)
    
    # Ensure max|c| is in valid range
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3:
        best_coeffs = best_coeffs / max_c * 1.0
    elif max_c > 1e5:
        best_coeffs = best_coeffs / max_c * 1.0
    
    return best_coeffs.astype(np.float64)

# EVOLVE-BLOCK-END
