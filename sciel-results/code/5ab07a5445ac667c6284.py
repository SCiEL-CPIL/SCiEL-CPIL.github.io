"""Fourier-invariant Hermite-Gaussian test function tuned via full moment matrices with cross terms."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial


def _hermite_cross_moments(degrees, power=2):
    """
    Compute the matrix M_ij = ∫ x^power H_i(x) H_j(x) e^{-x^2} dx
    for the given list of Hermite degrees.
    
    Uses the recurrence: x H_n = (1/2) H_{n+1} + n H_{n-1}
    and orthogonality: ∫ H_m H_n e^{-x^2} dx = sqrt(pi) 2^n n! δ_mn
    
    For power=2: x^2 H_n = (1/4) H_{n+2} + n H_n + (n(n-1)/4) H_{n-2} ... 
    Actually: x^2 H_n = (1/4)(H_{n+2} + 2n H_n + n(n-1) H_{n-2})
    More carefully: x H_n = 0.5 H_{n+1} + n H_{n-1}
    x^2 H_n = x(0.5 H_{n+1} + n H_{n-1}) = 0.5 x H_{n+1} + n x H_{n-1}
             = 0.5(0.5 H_{n+2} + (n+1) H_n) + n(0.5 H_n + (n-1) H_{n-2})
             = 0.25 H_{n+2} + 0.5(n+1) H_n + 0.5n H_n + n(n-1) H_{n-2}
             = 0.25 H_{n+2} + (n + 0.5) H_n + n(n-1) H_{n-2}
    
    So ∫ x^2 H_i H_j e^{-x^2} dx = sqrt(pi) * [
        0.25 * 2^j j! * δ_{i, j+2} +
        (j + 0.5) * 2^j j! * δ_{i,j} +
        j(j-1) * 2^j j! * δ_{i, j-2}
    ]
    Wait, I need to be more careful. Let me use:
    ∫ x^2 H_i H_j e^{-x^2} dx.
    
    Using x^2 H_j = 0.25 H_{j+2} + (j+0.5) H_j + j(j-1) H_{j-2}
    (with H_{-1} = H_{-2} = 0)
    
    Then ∫ x^2 H_i H_j e^{-x^2} dx = 0.25 ∫ H_i H_{j+2} e^{-x^2} dx + (j+0.5) ∫ H_i H_j e^{-x^2} dx + j(j-1) ∫ H_i H_{j-2} e^{-x^2} dx
    = sqrt(pi) [ 0.25 * 2^j j! * δ_{i,j+2} + (j+0.5) * 2^j j! * δ_{i,j} + j(j-1) * 2^{j-2} (j-2)! * δ_{i,j-2} ]
    
    Hmm, the last term: ∫ H_i H_{j-2} e^{-x^2} dx = sqrt(pi) 2^{j-2} (j-2)! δ_{i,j-2}
    So the last term is j(j-1) * sqrt(pi) * 2^{j-2} (j-2)! * δ_{i,j-2}
    = sqrt(pi) * j(j-1) * 2^{j-2} * (j-2)! * δ_{i,j-2}
    = sqrt(pi) * 2^j j! / 4 * δ_{i,j-2}  [since j(j-1)(j-2)! = j! and 2^{j-2} = 2^j/4]
    = sqrt(pi) * 0.25 * 2^j j! * δ_{i,j-2}
    
    So: ∫ x^2 H_i H_j e^{-x^2} dx = sqrt(pi) * 2^j j! * [ 0.25 δ_{i,j+2} + (j+0.5) δ_{i,j} + 0.25 δ_{i,j-2} ]
    
    For power=0: ∫ H_i H_j e^{-x^2} dx = sqrt(pi) 2^j j! δ_{i,j}
    """
    n = len(degrees)
    deg_set = set(degrees)
    M = np.zeros((n, n))
    for i_idx, di in enumerate(degrees):
        for j_idx, dj in enumerate(degrees):
            # ∫ x^power H_di H_dj e^{-x^2} dx
            if power == 0:
                if di == dj:
                    M[i_idx, j_idx] = 2.0**dj * factorial(dj, exact=True)
                else:
                    M[i_idx, j_idx] = 0.0
            elif power == 2:
                val = 0.0
                w_base = 2.0**dj * factorial(dj, exact=True)
                if di == dj + 2:
                    val += 0.25 * w_base
                elif di == dj:
                    val += (dj + 0.5) * w_base
                elif di == dj - 2:
                    val += 0.25 * w_base
                M[i_idx, j_idx] = val
            else:
                raise ValueError(f"Unsupported power {power}")
    return M


def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on A(f)*A(F) for g(x) = (a0 H0 + a4 H4 + a8 H8 + a12 H12) e^{-x^2/2}
    where a12 is chosen so g(0) = 0.
    
    The bound involves the second moments of |g|^2 and |F(g)|^2.
    By Fourier self-similarity of Hermite-Gaussians, these are equal.
    The uncertainty product bound scales as sqrt(<x^2><xi^2>) = <x^2>.
    But the actual bound C4 may involve the norm as well.
    
    Let's compute: bound = <x^2> = (v^T M2 v) / (v^T M0 v)
    where v = [a0, a4, a8, a12] and M0, M2 are the moment matrices.
    """
    H0_0, H4_0, H8_0, H12_0 = 1.0, -12.0, 1680.0, -47880.0
    a12 = (a0 + a4*H4_0 + a8*H8_0) / (-H12_0)
    # g(0) = a0 + a4*H4_0 + a8*H8_0 + a12*H12_0 = 0 => a12 = -(a0 + a4*H4_0 + a8*H8_0)/H12_0
    a12 = -(a0 + a4*H4_0 + a8*H8_0) / H12_0
    
    v = np.array([a0, a4, a8, a12])
    degrees = [0, 4, 8, 12]
    
    M0 = _hermite_cross_moments(degrees, power=0)
    M2 = _hermite_cross_moments(degrees, power=2)
    
    norm_sq = v @ M0 @ v
    x2_moment = v @ M2 @ v
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return np.inf
    
    x2 = x2_moment / norm_sq
    if not np.isfinite(x2) or x2 < 0:
        return np.inf
    
    # The bound on the uncertainty product is proportional to x2
    # (since <xi^2> = <x^2> by Fourier self-similarity)
    return x2


def construct():
    start = time.time()
    deadline = start + 175.0  # safety margin before 210s kill
    
    # Precompute moment matrices (they don't depend on a)
    degrees = [0, 4, 8, 12]
    M0 = _hermite_cross_moments(degrees, power=0)
    M2 = _hermite_cross_moments(degrees, power=2)
    
    H0_0, H4_0, H8_0, H12_0 = 1.0, -12.0, 1680.0, -47880.0
    
    def _objective(a):
        a0, a4, a8 = a
        a12 = -(a0 + a4*H4_0 + a8*H8_0) / H12_0
        v = np.array([a0, a4, a8, a12])
        norm_sq = v @ M0 @ v
        x2_moment = v @ M2 @ v
        if norm_sq <= 0 or not np.isfinite(norm_sq):
            return 1e10
        x2 = x2_moment / norm_sq
        if not np.isfinite(x2) or x2 < 0:
            return 1e10
        return x2
    
    # Try multiple initial points
    best_val = np.inf
    best_a = None
    
    initial_points = [
        [-113.0/100.0, 1.0/25.0, 1.0/3240.0],  # seed
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [0.5, -0.3, 0.1],
        [-0.5, 0.3, -0.1],
        [2.0, -1.0, 0.5],
        [-2.0, 1.0, -0.5],
        [1.0, -2.0, 0.3],
        [-1.0, 2.0, -0.3],
    ]
    
    # Add some random initial points
    rng = np.random.default_rng(42)
    for _ in range(20):
        initial_points.append(list(rng.normal(size=3) * 5.0))
    
    for x0 in initial_points:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.success or res.fun < best_val:
                if np.isfinite(res.fun) and res.fun < best_val:
                    best_val = res.fun
                    best_a = res.x
        except Exception:
            pass
    
    if best_a is not None and np.all(np.isfinite(best_a)):
        # Normalize for stability
        scale = np.max(np.abs(best_a))
        if scale > 0:
            best_a = best_a / scale
        return best_a
    else:
        return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
# EVOLVE-BLOCK-END
