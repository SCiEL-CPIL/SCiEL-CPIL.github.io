"""Rewritten program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import factorial

# Global deadline guard
_START_TIME = time.time()
_TIMEOUT = 210.0
_SAFETY_MARGIN = 5.0

def _time_remaining():
    return _TIMEOUT - (time.time() - _START_TIME) - _SAFETY_MARGIN

def _phys_hermite(n, x):
    """
    Compute the 'physicist's' Hermite polynomial H_n(x) using the recurrence:
    H_0(x) = 1
    H_1(x) = 2x
    H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    """
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    
    h_nm1 = np.ones_like(x)
    h_n = 2.0 * x
    for k in range(1, n):
        h_np1 = 2.0 * x * h_n - 2.0 * k * h_nm1
        h_nm1 = h_n
        h_n = h_np1
    return h_n

def construct():
    # Initial guess: Goncalves et al. 2017
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Discretization parameters
    # We need a grid large enough to capture the tails of the Gaussian and the polynomial growth.
    # The function is roughly exp(-x^2/2) * P(x). P(x) is degree 12.
    # The variance of the Gaussian is 1/2. 5 sigma is ~3.5. Let's use [-10, 10] to be safe.
    # Chebyshev nodes are clustered near the endpoints, which is good for smooth functions on a finite interval.
    N = 512  # Number of points. 512 is a power of 2, good for FFT if we used it, but here we just use for integration.
    # Actually, for Chebyshev integration, we can use Gauss-Chebyshev quadrature or just dense trapezoidal on Chebyshev nodes.
    # Let's use Chebyshev-Gauss nodes and weights for high precision integration.
    # Nodes: x_j = cos((2j-1)pi/(2N))
    # Weights: w_j = pi/N * (1 - (-1)^j) ... wait, standard Chebyshev-Gauss quadrature weights.
    # For N points, the weights are pi/N for all j if we map to [-1,1]?
    # Let's stick to a simple, robust approach: Dense grid with trapezoidal rule, or better, use `scipy.integrate.quad`?
    # No, we need analytic gradients. So we must compute integrals via matrix operations on the grid.
    
    # Let's use a uniform grid first for simplicity and robustness, or Chebyshev for better accuracy with fewer points.
    # Given the 180s limit, N=512 or 1024 is fine for matrix ops.
    
    # Let's use Chebyshev points on [-L, L]
    L = 12.0  # Large enough to make boundary terms negligible
    # Map Chebyshev nodes from [-1, 1] to [-L, L]
    # t_j = cos((2j-1)pi/(2N))
    j = np.arange(1, N + 1)
    t = np.cos((2 * j - 1) * np.pi / (2 * N))
    x = L * t
    
    # Weights for Chebyshev-Gauss quadrature on [-1, 1] are pi/N.
    # Mapping to [-L, L] multiplies weights by L.
    w = np.full(N, np.pi * L / N)
    
    # Sort x in increasing order for clarity (though trapezoidal doesn't care, matrix construction might be cleaner)
    # Chebyshev nodes are not sorted by default in the formula above?
    # cos is decreasing from 1 to -1 as angle goes 0 to pi.
    # So t goes from ~1 to ~-1.
    # Let's sort x and w together.
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    w = w[sort_idx]
    
    # Precompute the Gaussian weight
    # g(x) = exp(-x^2/2) * (a0 H0 + a4 H4 + a8 H8 + a12 H12)
    # We need to compute A(g) = ∫ x^2 g(x)^2 dx / ∫ g(x)^2 dx
    # And A(F(g)).
    
    # To compute A(F(g)), we can use the property that for a Gaussian times a polynomial,
    # the Fourier transform is also a Gaussian times a polynomial (derivatives of the Gaussian).
    # Specifically, F(H_n(x) e^{-x^2/2}) = (-i)^n H_n(x) e^{-x^2/2} (up to normalization).
    # Let's check the convention.
    # If F(f)(k) = ∫ f(x) e^{-ixk} dx, then F(e^{-x^2/2}) = sqrt(2pi) e^{-k^2/2}.
    # F(x^n e^{-x^2/2}) = i^n d^n/dk^n (sqrt(2pi) e^{-k^2/2}) = i^n (-1)^n sqrt(2pi) H_n(k) e^{-k^2/2} / 2^{n/2}? No.
    # The derivative of e^{-k^2/2} is related to Hermite polynomials.
    # d^n/dk^n e^{-k^2/2} = (-1)^n H_n(k) e^{-k^2/2}.
    # So F(x^n e^{-x^2/2}) = i^n (-1)^n sqrt(2pi) H_n(k) e^{-k^2/2} = (-i)^n sqrt(2pi) H_n(k) e^{-k^2/2}.
    
    # Let's verify for n=1.
    # F(x e^{-x^2/2}) = i d/dk (sqrt(2pi) e^{-k^2/2}) = i (-k) sqrt(2pi) e^{-k^2/2} = -i k sqrt(2pi) e^{-k^2/2}.
    # Formula: (-i)^1 sqrt(2pi) H_1(k) e^{-k^2/2} = -i sqrt(2pi) (2k) e^{-k^2/2} = -2i k ...
    # Mismatch by factor of 2?
    # H_1(k) = 2k.
    # d/dk e^{-k^2/2} = -k e^{-k^2/2}.
    # So F(x e^{-x^2/2}) = i (-k) ...
    # Let's re-evaluate H_n definition.
    # Physicist's H_n: H_0=1, H_1=2x.
    # Generating function: exp(2xt - t^2) = sum H_n(x) t^n / n!.
    # e^{-x^2/2} e^{2xt - t^2} = e^{-(x-t)^2/2} e^{t^2/2}.
    # This is getting complicated. Let's just compute the Fourier transform numerically on the grid or use the explicit polynomial relation.
    
    # Actually, there is a known identity:
    # If f(x) = P(x) e^{-x^2/2}, then F(f)(k) = Q(k) e^{-k^2/2}.
    # Q(k) is a polynomial of the same degree.
    # We can determine Q(k) by evaluating F(f) at several points? No, we need an analytic expression for gradients.
    
    # Alternative: Compute the variance in momentum space directly using integration by parts or operator properties.
    # A(F(f)) = ∫ k^2 |F(f)(k)|^2 dk / ∫ |F(f)(k)|^2 dk.
    # By Parseval, ∫ |F(f)(k)|^2 dk = 2π ∫ |f(x)|^2 dx.
    # Also, ∫ k^2 |F(f)(k)|^2 dk = 2π ∫ |f'(x)|^2 dx.
    # So A(F(f)) = ∫ |f'(x)|^2 dx / ∫ |f(x)|^2 dx.
    
    # This is much simpler! We just need to compute the L2 norm of f and f'.
    
    # f(x) = g(x) = e^{-x^2/2} P(x)
    # f'(x) = e^{-x^2/2} (P'(x) - x P(x))
    
    # Let P(x) = a0 H0(x) + a4 H4(x) + a8 H8(x) + a12 H12(x).
    # We have 3 free parameters a0, a4, a8. a12 is determined by P(0) = 0.
    # P(0) = a0 H0(0) + a4 H4(0) + a8 H8(0) + a12 H12(0) = 0.
    # H0(0) = 1
    # H4(0) = -3
    # H8(0) = 105
    # H12(0) = -10395
    # So a0 - 3 a4 + 105 a8 - 10395 a12 = 0
    # a12 = (a0 - 3 a4 + 105 a8) / 10395
    
    # We need to compute:
    # I0 = ∫ |f(x)|^2 dx = ∫ e^{-x^2} P(x)^2 dx
    # I1 = ∫ x^2 |f(x)|^2 dx = ∫ x^2 e^{-x^2} P(x)^2 dx
    # I2 = ∫ |f'(x)|^2 dx = ∫ e^{-x^2} (P'(x) - x P(x))^2 dx
    
    # Objective: C4 = A(g) A(F(g)) = (I1/I0) * (I2/I0) = I1 I2 / I0^2
    
    # We can compute these integrals using the precomputed grid and weights.
    # Since P(x) is a polynomial, P(x)^2 and (P'(x) - x P(x))^2 are also polynomials.
    # We can evaluate them at the grid points x.
    
    # Precompute H0, H4, H8, H12 and their derivatives at grid points x.
    H0 = _phys_hermite(0, x)
    H4 = _phys_hermite(4, x)
    H8 = _phys_hermite(8, x)
    H12 = _phys_hermite(12, x)
    
    # Derivatives: H_n'(x) = 2n H_{n-1}(x)
    H0_p = np.zeros_like(x)
    H4_p = 4 * _phys_hermite(3, x)  # 2*4*H3? No, H_n' = 2n H_{n-1}. So H4' = 8 H3.
    # Wait, H_4' = 8 H_3.
    H4_p = 8 * _phys_hermite(3, x)
    H8_p = 16 * _phys_hermite(7, x)
    H12_p = 24 * _phys_hermite(11, x)
    
    # Values at 0 for constraint
    H0_0 = 1.0
    H4_0 = -3.0
    H8_0 = 105.0
    H12_0 = -10395.0
    
    # Function to compute the score given coefficients a = [a0, a4, a8]
    def score(a):
        a0, a4, a8 = a
        a12 = (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0
        
        # P(x)
        P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        
        # P'(x)
        P_p = a0 * H0_p + a4 * H4_p + a8 * H8_p + a12 * H12_p
        
        # f(x) = e^{-x^2/2} P(x)
        # |f(x)|^2 = e^{-x^2} P(x)^2
        # f'(x) = e^{-x^2/2} (P'(x) - x P(x))
        # |f'(x)|^2 = e^{-x^2} (P'(x) - x P(x))^2
        
        exp_term = np.exp(-x**2)
        
        P_sq = P**2
        Q = P_p - x * P
        Q_sq = Q**2
        
        I0 = np.sum(w * exp_term * P_sq)
        I1 = np.sum(w * exp_term * x**2 * P_sq)
        I2 = np.sum(w * exp_term * Q_sq)
        
        if I0 < 1e-10:
            return 1e10  # Avoid division by zero
        
        return (I1 / I0) * (I2 / I0)
    
    # Gradient of the score
    # We can use finite differences for simplicity first, but analytic is better for SLSQP.
    # Given the complexity, let's try finite differences with `method='SLSQP'` and `jac` provided by finite diff?
    # SLSQP can handle finite differences.
    
    # Let's try to provide an analytic gradient to speed up convergence.
    # Let S = I1 I2 I0^{-2}
    # dS/da_i = I2 I0^{-2} dI1/da_i + I1 I0^{-2} dI2/da_i - 2 I1 I2 I0^{-3} dI0/da_i
    # = (I2/I0^2) I1' + (I1/I0^2) I2' - 2 (I1 I2 / I0^3) I0'
    
    # We need I0', I1', I2'.
    # I0 = ∫ e^{-x^2} P^2 dx
    # I0' = 2 ∫ e^{-x^2} P P' dx (where P' is derivative wrt coeff, not x)
    # Let's denote partial derivatives of P wrt a_i.
    # ∂P/∂a0 = H0 + ∂a12/∂a0 H12 = H0 + (1/10395) H12
    # ∂P/∂a4 = H4 + ∂a12/∂a4 H12 = H4 + (-3/10395) H12
    # ∂P/∂a8 = H8 + ∂a12/∂a8 H12 = H8 + (105/10395) H12
    
    dP_da0 = H0 + (1.0 / 10395.0) * H12
    dP_da4 = H4 - (3.0 / 10395.0) * H12
    dP_da8 = H8 + (105.0 / 10395.0) * H12
    
    # Similarly for P_p
    dPp_da0 = H0_p + (1.0 / 10395.0) * H12_p
    dPp_da4 = H4_p - (3.0 / 10395.0) * H12_p
    dPp_da8 = H8_p + (105.0 / 10395.0) * H12_p
    
    def gradient(a):
        a0, a4, a8 = a
        a12 = (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0
        
        P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        P_p = a0 * H0_p + a4 * H4_p + a8 * H8_p + a12 * H12_p
        
        exp_term = np.exp(-x**2)
        
        P_sq = P**2
        Q = P_p - x * P
        Q_sq = Q**2
        
        I0 = np.sum(w * exp_term * P_sq)
        I1 = np.sum(w * exp_term * x**2 * P_sq)
        I2 = np.sum(w * exp_term * Q_sq)
        
        if I0 < 1e-10:
            return np.zeros(3)
        
        # Compute dI0, dI1, dI2 for each coeff
        dI0 = np.zeros(3)
        dI1 = np.zeros(3)
        dI2 = np.zeros(3)
        
        dP = [dP_da0, dP_da4, dP_da8]
        dPp = [dPp_da0, dPp_da4, dPp_da8]
        
        for i in range(3):
            # dI0 = 2 ∫ e^{-x^2} P dP_i dx
            dI0[i] = 2.0 * np.sum(w * exp_term * P * dP[i])
            
            # dI1 = 2 ∫ e^{-x^2} x^2 P dP_i dx
            dI1[i] = 2.0 * np.sum(w * exp_term * x**2 * P * dP[i])
            
            # dI2 = 2 ∫ e^{-x^2} Q dQ_i dx
            # Q = P_p - x P
            # dQ_i = dPp_i - x dP_i
            dQ_i = dPp[i] - x * dP[i]
            dI2[i] = 2.0 * np.sum(w * exp_term * Q * dQ_i)
        
        # Assemble gradient
        grad = np.zeros(3)
        for i in range(3):
            grad[i] = (I2 / I0**2) * dI1[i] + (I1 / I0**2) * dI2[i] - 2.0 * (I1 * I2 / I0**3) * dI0[i]
            
        return grad
    
    # Optimization
    best_score = score(x0)
    best_x = x0.copy()
    
    # Try a few starting points or just one with SLSQP
    # SLSQP is good for small problems.
    
    if _time_remaining() > 10:
        res = minimize(score, x0, method='SLSQP', jac=gradient, options={'maxiter': 1000, 'ftol': 1e-12})
        if res.success or res.fun < best_score:
            best_score = res.fun
            best_x = res.x
    
    # If time permits, try a second optimization from a perturbed start
    if _time_remaining() > 20:
        x0_pert = x0 + 0.1 * np.random.randn(3)
        res2 = minimize(score, x0_pert, method='SLSQP', jac=gradient, options={'maxiter': 500, 'ftol': 1e-12})
        if res2.success or res2.fun < best_score:
            best_score = res2.fun
            best_x = res2.x
    
    # Return the coefficients
    return best_x
# EVOLVE-BLOCK-END
