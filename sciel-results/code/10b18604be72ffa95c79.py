"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

# Precompute Gauss-Hermite quadrature nodes and weights
# Using a high number of points for accuracy
n_points = 200
x_nodes, x_weights = np.polynomial.hermite.hermgauss(n_points)

# Define Hermite polynomials H_0, H_4, H_8, H_12
# Physicists' Hermite polynomials
def H0(x):
    return np.ones_like(x)

def H4(x):
    return 16 * x**4 - 48 * x**2 + 12

def H8(x):
    return 256 * x**8 - 4480 * x**6 + 26880 * x**4 - 40320 * x**2 + 1680

def H12(x):
    return (4096 * x**12 - 122880 * x**10 + 1548288 * x**8 - 
            9254400 * x**6 + 21621600 * x**4 - 20901120 * x**2 + 27720)

def compute_uncertainty_product(c0, c4, c8):
    # Calculate c12 such that g(0) = 0
    # g(0) = (c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)) * exp(0) = 0
    # H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-27720
    # c0 - 12*c4 + 1680*c8 - 27720*c12 = 0
    c12 = (c0 - 12 * c4 + 1680 * c8) / 27720.0
    
    # Evaluate P(x) for g(x)
    # P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    P = c0 * H0(x_nodes) + c4 * H4(x_nodes) + c8 * H8(x_nodes) + c12 * H12(x_nodes)
    
    # Integrate |g|^2 and x^2|g|^2
    # g(x) = P(x) * exp(-x^2/2)
    # |g(x)|^2 = P(x)^2 * exp(-x^2)
    # Integral is sum(weights * P(x)^2)
    norm_sq = np.sum(x_weights * P**2)
    
    if norm_sq <= 0:
        return np.inf
        
    moment_x2 = np.sum(x_weights * x_nodes**2 * P**2)
    var_x = moment_x2 / norm_sq
    std_x = np.sqrt(var_x)
    
    # Fourier transform
    # hat{g}(xi) = sum c_k (-i)^k H_k(xi) exp(-xi^2/2)
    # Let Q(xi) = c0*H0(xi) + c4*(-i)^4*H4(xi) + c8*(-i)^8*H8(xi) + c12*(-i)^12*H12(xi)
    # (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = (-i)^4 * (-i)^8 = 1
    # So Q(xi) = c0*H0(xi) + c4*H4(xi) + c8*H8(xi) + c12*H12(xi)
    # Wait, (-i)^12 = ((-i)^4)^3 = 1^3 = 1.
    # So the coefficients for the Fourier side are the same as the time side!
    # This is because all degrees are multiples of 4.
    # Therefore, |hat{g}(xi)|^2 has the same form as |g(x)|^2 with x replaced by xi.
    # Since the quadrature nodes and weights are the same for the integral over real line,
    # and the polynomial structure is identical, the variance in xi is the same as in x?
    # Let's verify.
    # Q(xi) = P(xi).
    # norm_sq_hat = sum(weights * Q(xi)^2) = sum(weights * P(xi)^2) = norm_sq.
    # moment_xi2 = sum(weights * xi^2 * Q(xi)^2) = sum(weights * xi^2 * P(xi)^2) = moment_x2.
    # So var_xi = var_x.
    # Thus std_xi = std_x.
    # Product = std_x * std_xi = var_x.
    
    # Wait, is this correct?
    # The Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(\xi) e^{-\xi^2/2}.
    # For n=0, (-i)^0 = 1.
    # For n=4, (-i)^4 = 1.
    # For n=8, (-i)^8 = 1.
    # For n=12, (-i)^12 = 1.
    # Yes, the coefficients are identical.
    # So the distribution of |g|^2 and |hat{g}|^2 are identical.
    # Therefore A(g) = A(hat{g}).
    # The product is A(g)^2 = Var(g).
    # We just need to minimize the variance of g(x) subject to g(0)=0.
    
    return var_x

def objective(c):
    c0, c4, c8 = c
    return compute_uncertainty_product(c0, c4, c8)

def construct():
    # Initial guess from the seed program
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    # Perform optimization
    # Use L-BFGS-B for efficiency
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-15})
    
    # Check if optimization improved the score
    if res.success:
        # Return the optimized coefficients
        return res.x
    else:
        # Fallback to seed if optimization fails
        return x0

# EVOLVE-BLOCK-END
