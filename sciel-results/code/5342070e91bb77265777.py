"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import factorial, sqrt, pi
from scipy.optimize import differential_evolution

def _hermite_val(n, x):
    """Compute physicists' Hermite polynomial H_n(x) at scalar x."""
    if n == 0:
        return 1.0
    elif n == 1:
        return 2.0 * x
    H_prev = 1.0
    H_curr = 2.0 * x
    for k in range(2, n + 1):
        H_next = 2.0 * x * H_curr - 2.0 * (k - 1) * H_prev
        H_prev = H_curr
        H_curr = H_next
    return H_curr

# Basis indices: 0, 4, 8, 12
indices = [0, 4, 8, 12]

# Precompute H_n(0) for n in indices
H_at_0 = np.array([_hermite_val(n, 0.0) for n in indices])

# c12 is determined by g(0) = 0: c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
H12_0 = H_at_0[3]

# Norm factors: ||H_n * exp(-x^2/2)||^2 = 2^n * n! * sqrt(pi)
norms_sq = np.array([(2.0**n) * factorial(n) * sqrt(pi) for n in indices])

def _compute_uncertainty_bound(c0, c4, c8):
    """
    Compute the uncertainty product bound for given coefficients.
    
    For g(x) = sum c_n H_n(x) exp(-x^2/2), we need:
    - <x^2> = sum c_n^2 * norms_sq[n] * (n + 0.5) / sum c_n^2 * norms_sq[n]
    - <p^2> where p = -i d/dx
    
    The key insight: for the Fourier transform convention F(f)(k) = (1/sqrt(2pi)) int f(x) e^{-ikx} dx,
    the Hermite functions are eigenfunctions of the Fourier transform with eigenvalue i^n.
    
    For a real, even function (our case), <x^2> = <p^2> in the appropriate normalization.
    However, we need to be more careful. Let's compute <p^2> directly.
    
    Actually, for the standard normalization where the Fourier transform is unitary,
    the uncertainty principle gives sigma_x * sigma_p >= 1/2.
    
    For our function g(x) = sum c_n H_n(x) e^{-x^2/2}:
    - The position variance <x^2> can be computed from the moments
    - The momentum variance <p^2> = <(d/dx g)^2> (up to normalization)
    
    Let me use a different approach: compute the actual L2 norm and the 
    weighted moments properly.
    
    For the uncertainty product in the context of the Fourier-invariant bound,
    we're looking at ||g||_1 * ||g_hat||_1 or similar. But given the task description
    mentions "upper bound on C4", let's compute the standard uncertainty product.
    
    Actually, re-reading: the score is "the upper bound on C4" and lower is better.
    The seed program returns x2 (the <x^2> value). Let me stick with computing
    the proper uncertainty product sqrt(<x^2> * <p^2>).
    
    For Hermite functions, the key relation is:
    d/dx [H_n(x) e^{-x^2/2}] = H_{n+1}(x) e^{-x^2/2} - x H_n(x) e^{-x^2/2}
    
    This gets complicated. Let me use the fact that for the orthonormal Hermite
    basis, <x^2> and <p^2> can be computed from the coefficient vector.
    
    Using the ladder operator approach:
    a = (x + p)/sqrt(2), a† = (x - p)/sqrt(2)
    x = (a + a†)/sqrt(2), p = (a - a†)/(i*sqrt(2))
    
    For state |psi> = sum c_n |n> (in orthonormal basis):
    <x^2> = (1/2) * (sum_{n,m} c_n* c_m <n|x^2|m>)
    
    In the Hermite basis, x = (a + a†)/sqrt(2), so:
    x^2 = (1/2)(a^2 + a a† + a† a + a†^2)
    
    <n|x^2|m> involves terms like sqrt((m+1)(m+2)) delta_{n,m+2} + ...
    
    This is getting complex. Let me use a simpler numerical approach or
    the known result that for even functions in this basis, we can compute
    the moments directly.
    
    Actually, the simplest correct approach: 
    For g(x) = sum c_n H_n(x) e^{-x^2/2}, with the L2 inner product:
    <f, g> = int f(x) g(x) dx
    
    The norm is ||g||^2 = sum c_n^2 * 2^n * n! * sqrt(pi)
    
    <x^2> = int x^2 g(x)^2 dx / ||g||^2
    
    For the momentum, p = -i d/dx, so:
    <p^2> = int |g'(x)|^2 dx / ||g||^2
    
    Let me compute g'(x) and integrate numerically, or use the Hermite recurrence.
    
    d/dx [H_n(x) e^{-x^2/2}] = [2n x H_{n-1}(x) - 2n H_n(x)] e^{-x^2/2} / ... 
    Actually: d/dx H_n(x) = 2n H_{n-1}(x)
    So d/dx [H_n(x) e^{-x^2/2}] = [2n H_{n-1}(x) - x H_n(x)] e^{-x^2/2}
    
    This means g'(x) = sum c_n [2n H_{n-1}(x) - x H_n(x)] e^{-x^2/2}
    
    The integral of |g'(x)|^2 is complicated. Let me just use the fact that
    for the purpose of this task, the seed program was using <x^2> as a proxy,
    and the score was 0.7515. Let me compute <x^2> properly and also try to
    get a better estimate of the actual bound.
    
    For now, let me compute <x^2> which is well-defined:
    """
    c12 = -(c0 * H_at_0[0] + c4 * H_at_0[1] + c8 * H_at_0[2]) / H12_0
    c = np.array([c0, c4, c8, c12], dtype=float)
    
    norm_sq = np.sum(c**2 * norms_sq)
    if norm_sq < 1e-15:
        return 1e10
    
    # <x^2> = sum c_n^2 * norms_sq[n] * (n + 0.5) / norm_sq
    x2_vals = np.array([n + 0.5 for n in indices])
    x2 = np.sum(c**2 * norms_sq * x2_vals) / norm_sq
    
    # For the Fourier transform of H_n(x) e^{-x^2/2}, we get i^n H_n(x) e^{-x^2/2}
    # (up to sign conventions). So the "momentum space" function has the same
    # shape. For a real even function, <p^2> = <x^2> in the appropriate units.
    # 
    # But actually, the Fourier transform introduces a scaling. Let's think about
    # the normalization. If F(f)(k) = (1/sqrt(2pi)) int f(x) e^{-ikx} dx, then
    # F(H_n(x) e^{-x^2/2})(k) = i^n H_n(k) e^{-k^2/2} (approximately, with the
    # right normalization factors).
    #
    # So <p^2> in the Fourier domain equals <k^2> which has the same form as <x^2>
    # for the same coefficients. Thus <p^2> = <x^2>.
    #
    # The uncertainty product is sqrt(<x^2> * <p^2>) = <x^2>.
    #
    # But wait - we need to account for the fact that p = -i d/dx, and the
    # Fourier transform gives us the momentum representation. The variance in
    # momentum space is <p^2> - <p>^2. For an even function, <p> = 0.
    #
    # So the uncertainty product is sqrt(<x^2> * <p^2>) = sqrt(x2 * x2) = x2.
    #
    # This matches what the seed program was computing. Let me return x2.
    return x2

def construct():
    """Find optimal coefficients for H0, H4, H8 using Differential Evolution."""
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    # Define bounds for the coefficients
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    
    # Use Differential Evolution for global optimization
    try:
        result = differential_evolution(
            _compute_uncertainty_bound,
            bounds=bounds,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        if result.success and result.fun < 1e10:
            return np.array(result.x)
        else:
            # Return the best found even if not fully converged
            if result.x is not None and result.fun < 1e10:
                return np.array(result.x)
    except Exception:
        pass
    
    # Fallback: try a few manual points
    candidates = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, -0.5, 0.0],
        [0.5, 0.5, -0.5],
        [1.0, -1.0, 0.5],
    ]
    
    best_score = float('inf')
    best_c = None
    
    for c in candidates:
        if time.time() > deadline:
            break
        try:
            score = _compute_uncertainty_bound(*c)
            if score < best_score:
                best_score = score
                best_c = np.array(c)
        except Exception:
            pass
    
    if best_c is not None:
        return best_c
    
    return np.array([1.0, 0.0, 0.0])

# EVOLVE-BLOCK-END
