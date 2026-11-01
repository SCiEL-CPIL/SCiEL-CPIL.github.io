import numpy as np
import time
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss

START_TIME = time.time()
TIME_LIMIT = 195.0

def construct():
    # GH Quadrature setup
    n_gh = 100
    z, w = hermgauss(n_gh)
    x = np.sqrt(2) * z
    # weights for the e^{-x^2/2} integral
    # I = int f(x) e^{-x^2/2} dx = int f(sqrt(2)z) e^{-z^2} sqrt(2) dz
    # approx = sqrt(2) * sum(w_i * f(sqrt(2) z_i))
    
    # Precompute Hermite polynomials
    # P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # Constraint: P(0) = 0 => c0*1 + c4*3 + c8*105 + c12*945 = 0
    # c12 = -(c0 + 3c4 + 105c8) / 945
    
    # To simplify derivatives, define effective basis functions that satisfy the constraint automatically.
    # P(x) = c0 * Phi0(x) + c4 * Phi4(x) + c8 * Phi8(x)
    # where Phi0 = H0 - (1/945)H12
    # Phi4 = H4 - (3/945)H12
    # Phi8 = H8 - (105/945)H12
    
    # We need to compute integrals for the uncertainty product bound.
    # The problem is to minimize the upper bound on A(f) * A(F(f)).
    # For a function g(x) = P(x) e^{-x^2/4} (standard form for these problems, usually g is the wavefunction),
    # A(g) is typically the variance of position: <x^2> - <x>^2.
    # Since P is even (H0, H4, H8, H12 are even), g is even, so <x> = 0.
    # A(g) = <x^2> = (int x^2 g^2 dx) / (int g^2 dx).
    
    # The Fourier transform of g(x) = P(x) e^{-x^2/4} is related to the derivatives of P.
    # Specifically, if g(x) = P(x) e^{-x^2/4}, then \hat{g}(k) is proportional to Q(k) e^{-k^2/4}
    # where Q(k) involves derivatives of P.
    # For the Hermite-Gaussian basis, the Fourier transform maps H_n(x) e^{-x^2/4} to (i)^n H_n(k) e^{-k^2/4} (up to constants).
    # Actually, the standard Hermite functions psi_n(x) = H_n(x) e^{-x^2/2} are eigenfunctions of the Fourier transform.
    # Here we have e^{-x^2/4}. Let's stick to the general moment calculation.
    
    # Let g(x) = P(x) e^{-x^2/4}.
    # M0 = int g^2 dx = int P^2 e^{-x^2/2} dx.
    # M2 = int x^2 g^2 dx = int x^2 P^2 e^{-x^2/2} dx.
    # A_pos = M2 / M0.
    
    # For the momentum space:
    # \hat{g}(k) = (1/sqrt(2pi)) int P(x) e^{-x^2/4} e^{-ikx} dx.
    # This is the Fourier transform of P(x) e^{-x^2/4}.
    # Using the property that multiplication by x in position space corresponds to i d/dk in momentum space? No.
    # It is easier to use the known result for the variance of the Fourier transform of a Hermite polynomial times Gaussian.
    # However, a robust numerical approach is to compute the moments in k-space directly if we can express \hat{g}.
    
    # Alternative: Use the relation between moments.
    # For g(x) = P(x) e^{-x^2/4}, the Fourier transform \hat{g}(k) can be computed analytically or via quadrature.
    # But we can also use the fact that the variance in k-space, A_mom = <k^2>_k, is related to the derivatives of P.
    # Specifically, for g(x) = P(x) e^{-x^2/4},
    # \hat{g}(k) = C * Q(k) e^{-k^2/4} where Q is a polynomial.
    # Actually, let's just compute the integrals numerically for both position and momentum?
    # No, the momentum space function is not simply a polynomial times Gaussian in the same way unless we transform the basis.
    
    # Let's use the analytic properties.
    # g(x) = P(x) e^{-x^2/4}.
    # \hat{g}(k) = (1/sqrt(2pi)) \int_{-\infty}^{\infty} P(x) e^{-x^2/4} e^{-ikx} dx.
    # This integral is the characteristic function of the distribution proportional to P(x)e^{-x^2/4} squared? No.
    
    # Let's use the derivative method.
    # \int k^2 |\hat{g}(k)|^2 dk = \int (g'(x))^2 dx (up to constants).
    # Plancherel: \int |\hat{g}(k)|^2 dk = \int |g(x)|^2 dx.
    # \int k^2 |\hat{g}(k)|^2 dk = \int |g'(x)|^2 dx.
    # So A_mom = \int (g')^2 dx / \int g^2 dx.
    
    # g(x) = P(x) e^{-x^2/4}
    # g'(x) = P'(x) e^{-x^2/4} + P(x) (-x/2) e^{-x^2/4} = (P'(x) - x/2 P(x)) e^{-x^2/4}
    # Let Q(x) = P'(x) - x/2 P(x).
    # Then \int (g')^2 dx = \int Q(x)^2 e^{-x^2/2} dx.
    
    # So the objective function is:
    # Score = A_pos * A_mom = (M2 / M0) * (N2 / N0)
    # where M0 = \int P^2 e^{-x^2/2} dx
    # M2 = \int x^2 P^2 e^{-x^2/2} dx
    # N0 = \int Q^2 e^{-x^2/2} dx  (Note: N0 should be equal to M0 by Plancherel? No, \int g^2 = \int \hat{g}^2. So N0 = M0. Let's verify.)
    # \int g^2 dx = \int P^2 e^{-x^2/2} dx = M0.
    # \int (g')^2 dx = \int Q^2 e^{-x^2/2} dx = N2.
    # So A_mom = N2 / M0.
    # Wait, the normalization of Fourier transform matters.
    # If \hat{g}(k) = 1/sqrt(2pi) \int g(x) e^{-ikx} dx, then \int |\hat{g}|^2 dk = \int |g|^2 dx.
    # And \int k^2 |\hat{g}|^2 dk = \int |g'|^2 dx.
    # So A_mom = \int (g')^2 dx / \int g^2 dx = N2 / M0.
    
    # So Objective = (M2 / M0) * (N2 / M0) = M2 * N2 / M0^2.
    
    # We need to compute:
    # P(x) = c0 Phi0 + c4 Phi4 + c8 Phi8
    # P'(x) = c0 Phi0' + c4 Phi4' + c8 Phi8'
    # Q(x) = P'(x) - x/2 P(x)
    
    # Let's precompute the basis functions and their derivatives at the quadrature points.
    
    from scipy.special import eval_hermite
    
    H0 = np.ones_like(x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Derivatives
    # H_n' = 2n H_{n-1}
    H0_p = np.zeros_like(x)
    H4_p = 8 * eval_hermite(3, x)
    H8_p = 16 * eval_hermite(7, x)
    H12_p = 24 * eval_hermite(11, x)
    
    # Effective basis
    phi0 = H0 - (1.0/945.0) * H12
    phi4 = H4 - (3.0/945.0) * H12
    phi8 = H8 - (105.0/945.0) * H12
    
    phi0_p = H0_p - (1.0/945.0) * H12_p
    phi4_p = H4_p - (3.0/945.0) * H12_p
    phi8_p = H8_p - (105.0/945.0) * H12_p
    
    # Weights for the integrals: w_i * sqrt(2)
    w_eff = np.sqrt(2.0) * w
    
    def compute_moments(c0, c4, c8):
        P = c0 * phi0 + c4 * phi4 + c8 * phi8
        P_p = c0 * phi0_p + c4 * phi4_p + c8 * phi8_p
        Q = P_p - 0.5 * x * P
        
        P2 = P * P
        Q2 = Q * Q
        
        M0 = np.sum(w_eff * P2)
        M2 = np.sum(w_eff * x**2 * P2)
        N2 = np.sum(w_eff * Q2)
        
        return M0, M2, N2

    def objective_and_grad(coeffs):
        c0, c4, c8 = coeffs
        M0, M2, N2 = compute_moments(c0, c4, c8)
        
        # Objective = M2 * N2 / M0^2
        if M0 < 1e-12:
            return 1e10, np.zeros(3)
            
        obj = (M2 * N2) / (M0 * M0)
        
        # Gradient
        # d/dc_k (M2 N2 / M0^2) = (dM2/dck N2 M0^2 - M2 N2 2 M0 dM0/dck) / M0^4
        # = (N2/M0^2) dM2/dck - 2 (M2 N2 / M0^3) dM0/dck
        # = (N2/M0^2) dM2/dck - 2 (obj/M0) dM0/dck
        
        # We need dM0/dck, dM2/dck, dN2/dck
        # M0 = sum w P^2. dM0/dck = 2 sum w P P'_k
        # M2 = sum w x^2 P^2. dM2/dck = 2 sum w x^2 P P'_k
        # N2 = sum w Q^2. dN2/dck = 2 sum w Q Q'_k
        
        # P'_k = phi_k
        # Q = P'_k_total - x/2 P.
        # Q'_k = phi_k' - x/2 phi_k
        
        P = c0 * phi0 + c4 * phi4 + c8 * phi8
        Q = c0 * (phi0_p - 0.5*x*phi0) + c4 * (phi4_p - 0.5*x*phi4) + c8 * (phi8_p - 0.5*x*phi8)
        
        # Precompute terms for gradient
        term_M0_0 = 2 * np.sum(w_eff * P * phi0)
        term_M0_4 = 2 * np.sum(w_eff * P * phi4)
        term_M0_8 = 2 * np.sum(w_eff * P * phi8)
        
        term_M2_0 = 2 * np.sum(w_eff * x**2 * P * phi0)
        term_M2_4 = 2 * np.sum(w_eff * x**2 * P * phi4)
        term_M2_8 = 2 * np.sum(w_eff * x**2 * P * phi8)
        
        Q_0 = phi0_p - 0.5*x*phi0
        Q_4 = phi4_p - 0.5*x*phi4
        Q_8 = phi8_p - 0.5*x*phi8
        
        term_N2_0 = 2 * np.sum(w_eff * Q * Q_0)
        term_N2_4 = 2 * np.sum(w_eff * Q * Q_4)
        term_N2_8 = 2 * np.sum(w_eff * Q * Q_8)
        
        coeff_M2 = N2 / (M0 * M0)
        coeff_M0 = -2 * obj / M0
        
        grad0 = coeff_M2 * term_M2_0 + coeff_M0 * term_M0_0 + (M2 / M0**2) * term_N2_0
        # Wait, derivative of (M2 * N2 * M0^-2)
        # = dM2 * N2 * M0^-2 + M2 * dN2 * M0^-2 + M2 * N2 * (-2) * M0^-3 * dM0
        # = (N2/M0^2) dM2 + (M2/M0^2) dN2 - (2 M2 N2 / M0^3) dM0
        
        grad0 = (N2 / (M0*M0)) * term_M2_0 + (M2 / (M0*M0)) * term_N2_0 - (2 * obj / M0) * term_M0_0
        grad4 = (N2 / (M0*M0)) * term_M2_4 + (M2 / (M0*M0)) * term_N2_4 - (2 * obj / M0) * term_M0_4
        grad8 = (N2 / (M0*M0)) * term_M2_8 + (M2 / (M0*M0)) * term_N2_8 - (2 * obj / M0) * term_M0_8
        
        return obj, np.array([grad0, grad4, grad8])

    def objective_wrapper(coeffs):
        val, _ = objective_and_grad(coeffs)
        return val

    def grad_wrapper(coeffs):
        _, grad = objective_and_grad(coeffs)
        return grad

    # Initial guess
    # The seed program used [-1.13, 0.04, 1.0/3240.0]
    # Let's start with that and also try 0,0,0 (which gives P=0, undefined) so avoid 0.
    best_coeffs = np.array([-1.13, 0.04, 1.0/3240.0])
    best_score = objective_wrapper(best_coeffs)
    
    np.random.seed(42)
    
    bounds = [(-5.0, 5.0), (-2.0, 2.0), (-0.5, 0.5)]
    
    # Multi-start
    for i in range(30):
        if time.time() > TIME_LIMIT - 10:
            break
            
        x0 = np.array([
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1]),
            np.random.uniform(bounds[2][0], bounds[2][1])
        ])
        
        try:
            res = minimize(objective_wrapper, x0, jac=grad_wrapper, method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass
            
    return best_coeffs
