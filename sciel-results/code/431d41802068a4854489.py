# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_basis(k=10):
    """Build real orthonormal spherical harmonic basis for degree k."""
    # Gauss-Legendre quadrature (70 points in cos theta)
    x, w = np.polynomial.legendre.leggauss(70)
    # 130 equally spaced phi points
    phi = np.linspace(0, 2 * np.pi, 130, endpoint=False)
    
    # Weights: w_i * dphi (uniform phi spacing)
    dphi = 2 * np.pi / 130
    weights = np.outer(w, np.ones(130)) * dphi
    # Total weight should be 4*pi
    total = weights.sum()
    weights = weights * (4 * np.pi / total)
    
    # Build association matrices for cos(m*phi) and sin(m*phi)
    cos_m = np.zeros((k + 1, 70, 130))
    sin_m = np.zeros((k + 1, 70, 130))
    for m in range(1, k + 1):
        cos_m[m] = np.cos(m * phi)[np.newaxis, :]
        sin_m[m] = np.sin(m * phi)[np.newaxis, :]
    
    # Compute associated Legendre polynomials P_k^m(x) with Condon-Shortley phase
    # Using recurrence: P_l^m(x)
    # P_0^0 = 1, P_1^0 = x
    # (l-m+1)P_l^m = (2l-1)x P_{l-1}^m - (l+m-1)P_{l-2}^m
    # P_l^1 = -(1-x^2)^{1/2} P_{l-1}'
    # P_l^{m+1} = (1-x^2)^{1/2} P_l' - m x P_l^m
    
    # Compute P_k^m for m = 0, 1, ..., k
    P = np.zeros((k + 1, k + 1, 70))  # P[l, m, x_idx]
    P[0, 0, :] = 1.0
    if k >= 1:
        P[1, 0, :] = x
    
    for l in range(2, k + 1):
        for m in range(0, l + 1):
            if m == 0:
                P[l, 0, :] = ((2 * l - 1) * x * P[l - 1, 0, :] - (l - 1) * P[l - 2, 0, :]) / l
            else:
                # Use recurrence for m > 0
                if m == l:
                    # P_l^l = (-1)^l (2l-1)!! (1-x^2)^{l/2}
                    # Compute via derivative formula
                    # P_l^l = (-1)^l (2l-1)!! (1-x^2)^{l/2}
                    # Simpler: use the relation P_l^l = -(1-x^2)^{1/2} P_{l-1}'
                    # But let's use the standard recurrence
                    # P_l^m = (1/(l-m)) [ (2l-1) x P_{l-1}^m - (l+m-1) P_{l-2}^m ] for m < l
                    # For m = l, we need P_l^l
                    # P_l^l = (-1)^l (2l-1)!! (1-x^2)^{l/2}
                    # Let's compute using: P_l^l = -(1-x^2)^{1/2} * d/dx P_{l-1}
                    # Actually, let's use the forward recurrence properly
                    # P_l^m for m=l: use P_l^l = (-1)^l (2l-1)!! (1-x^2)^{l/2}
                    # (2l-1)!! = (2l)! / (2^l l!)
                    double_fact = 1.0
                    for j in range(1, 2 * l, 2):
                        double_fact *= j
                    sign = (-1) ** l
                    P[l, l, :] = sign * double_fact * (1 - x**2) ** (l / 2.0)
                else:
                    P[l, m, :] = ((2 * l - 1) * x * P[l - 1, m, :] - (l + m - 1) * P[l - 2, m, :]) / (l - m)
    
    # Normalization: N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    # For degree k=10, (2k+1)/(4*pi) = 21/(4*pi)
    N = np.zeros(k + 1)
    for m in range(k + 1):
        # (10-m)!/(10+m)!
        ratio = 1.0
        for j in range(10 - m + 1, 10 + m + 1):
            ratio /= j
        N[m] = np.sqrt(21.0 / (4 * np.pi) * ratio)
    
    # Build basis functions Y_m for m = -10..10
    # Y_0 = N_0 P_10^0
    # Y_m = sqrt(2) N_m P_10^m cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m| sin(|m| phi) for m < 0
    basis = np.zeros((21, 70, 130))
    # m = 0 (index 10)
    basis[10, :, :] = N[0] * P[10, 0, :][:, np.newaxis]
    
    for m in range(1, k + 1):
        # m > 0: index = 10 + m
        basis[10 + m, :, :] = np.sqrt(2) * N[m] * P[10, m, :][:, np.newaxis] * cos_m[m]
        # m < 0: index = 10 - m
        basis[10 - m, :, :] = np.sqrt(2) * N[m] * P[10, m, :][:, np.newaxis] * sin_m[m]
    
    return basis, weights


def _compute_R(coeffs, basis, weights):
    """Compute R = ||f||_4 / ||f||_2."""
    f = np.tensordot(coeffs, basis, axes=([0], [0]))
    f2 = (f * f).sum(axis=(1, 2)) * weights.sum()  # No, need to do weighted sum
    # Actually: ||f||_2^2 = sum_i,j f(i,j)^2 * w(i,j)
    f2 = np.sum(f * f * weights)
    f4 = np.sum(f * f * f * f * weights)
    if f2 <= 0:
        return np.inf, f
    R = np.sqrt(f4 / f2)
    return R, f


def _compute_grad(coeffs, basis, weights):
    """Compute gradient of R with respect to coeffs."""
    f = np.tensordot(coeffs, basis, axes=([0], [0]))
    f2 = np.sum(f * f * weights)
    f4 = np.sum(f * f * f * f * weights)
    if f2 <= 0:
        return np.zeros(21)
    
    # R = sqrt(f4/f2)
    # dR/dc_i = (1/(2*R)) * (d(f4/f2)/dc_i)
    # f4/f2 = f4 * f2^{-1}
    # d(f4/f2)/dc_i = (df4/dc_i * f2 - f4 * df2/dc_i) / f2^2
    # df2/dc_i = 2 * sum(f * Y_i * w)
    # df4/dc_i = 4 * sum(f^3 * Y_i * w)
    
    # Compute sum(f * Y_i * w) for all i: this is basis^T (f * w)
    fw = f * weights
    sum_fY = np.tensordot(basis, fw, axes=([1, 2], [1, 2]))  # shape (21,)
    
    # Compute sum(f^3 * Y_i * w)
    f3w = f * f * f * weights
    sum_f3Y = np.tensordot(basis, f3w, axes=([1, 2], [1, 2]))
    
    df2 = 2.0 * sum_fY
    df4 = 4.0 * sum_f3Y
    
    R = np.sqrt(f4 / f2)
    grad = (1.0 / (2.0 * R)) * (df4 * f2 - f4 * df2) / (f2 * f2)
    
    return grad


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients for degree-10 spherical harmonic minimizing
    R = ||f||_4 / ||f||_2.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 580s to leave margin
    
    basis, weights = _build_basis(10)
    
    # Initial guess: all ones
    coeffs = np.ones(21, dtype=np.float64)
    best_coeffs = coeffs.copy()
    best_R, _ = _compute_R(coeffs, basis, weights)
    
    # Gradient descent with adaptive step size
    lr = 0.01
    momentum = 0.9
    vel = np.zeros(21)
    
    # Try multiple restarts with different initial conditions
    n_restarts = 5
    max_iter = 5000
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            coeffs = np.ones(21, dtype=np.float64)
        elif restart == 1:
            # Only m=0
            coeffs = np.zeros(21, dtype=np.float64)
            coeffs[10] = 1.0
        elif restart == 2:
            # Only even m
            coeffs = np.zeros(21, dtype=np.float64)
            for i in range(0, 21, 2):
                coeffs[i] = 1.0
        elif restart == 3:
            # Random
            rng = np.random.default_rng(restart)
            coeffs = rng.standard_normal(21)
        else:
            # Alternating signs
            coeffs = np.ones(21, dtype=np.float64)
            coeffs[::2] = -1.0
        
        vel = np.zeros(21)
        lr = 0.01
        best_R_restart = np.inf
        best_coeffs_restart = coeffs.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 30:
                break
            
            R, f = _compute_R(coeffs, basis, weights)
            
            if R < best_R_restart:
                best_R_restart = R
                best_coeffs_restart = coeffs.copy()
            
            # Compute gradient
            grad = _compute_grad(coeffs, basis, weights)
            
            # Normalize gradient
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-12:
                break
            
            grad = grad / gnorm
            
            # Momentum update
            vel = momentum * vel - lr * grad
            coeffs = coeffs + vel
            
            # Adaptive learning rate: if R increased, reduce lr
            R_new, _ = _compute_R(coeffs, basis, weights)
            if R_new > R * 1.01:
                lr *= 0.8
                vel *= 0.5
                coeffs = best_coeffs_restart.copy()
            
            # Check convergence
            if gnorm < 1e-8:
                break
        
        if best_R_restart < best_R:
            best_R = best_R_restart
            best_coeffs = best_coeffs_restart.copy()
    
    # Final refinement with smaller steps
    if time.time() < deadline - 10:
        coeffs = best_coeffs.copy()
        vel = np.zeros(21)
        lr = 0.001
        for it in range(2000):
            if time.time() > deadline - 5:
                break
            R, f = _compute_R(coeffs, basis, weights)
            grad = _compute_grad(coeffs, basis, weights)
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-12:
                break
            grad = grad / gnorm
            vel = 0.9 * vel - lr * grad
            coeffs = coeffs + vel
            R_new, _ = _compute_R(coeffs, basis, weights)
            if R_new < R:
                best_R = R_new
                best_coeffs = coeffs.copy()
            else:
                lr *= 0.9
                vel *= 0.5
                coeffs = best_coeffs.copy()
    
    # Validate
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs *= (1.0 / max_abs) if max_abs > 0 else 1.0
    if max_abs > 1e5:
        best_coeffs *= (1e5 / max_abs)
    
    return best_coeffs


# EVOLVE-BLOCK-END
