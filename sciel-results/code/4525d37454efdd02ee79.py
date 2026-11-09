# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
from scipy.special import lpmv
import math

# Constants
L = 10
N_THETA = 70
N_PHI = 130

# Precompute quadrature nodes and weights for the sphere
# Gauss-Legendre in cos(theta)
# We need weights for the surface integral: dOmega = dphi * d(cos theta)
# Integrand is f^p. 
# Surface measure dOmega = sin(theta) dtheta dphi. 
# Let u = cos(theta). du = -sin(theta) dtheta.
# So integral over sphere is integral_phi integral_{-1}^{1} f(u, phi) du dphi.
# Gauss-Legendre nodes u_i and weights w_i approximate integral_{-1}^{1} g(u) du.
# Total mass is 4*pi. The quadrature should integrate 1 to 4*pi?
# Standard Gauss-Legendre integrates polynomials exactly. 
# The problem states: "computed by 70-point Gauss-Legendre quadrature in cos theta times 130 equally spaced phi (exact for this degree)."
# This implies: 
# I = (2*pi / N_PHI) * sum_{j=0}^{N_PHI-1} sum_{i=0}^{N_THETA-1} w_i * f(u_i, phi_j)^p
# Wait, standard GL weights sum to 2. 
# If we use standard GL nodes u_i in [-1, 1] and weights w_i, 
# sum w_i = 2.
# The phi integral is 2*pi.
# So the total integral approximation is (2*pi) * (1/N_PHI) * sum_j sum_i w_i * ... ?
# Let's check the "exact for this degree" claim.
# Degree 10 harmonic. f^4 is degree 40.
# Gauss-Legendre with N points integrates polynomials of degree 2N-1 exactly.
# 2*70 - 1 = 139. 139 >= 40. So GL in theta is exact for the angular part.
# Equally spaced phi is exact for periodic functions if N_PHI is a multiple of the frequency?
# The frequency in phi is m, max |m|=10. N_PHI=130. 130 is a multiple of 10. So it's exact.
# 
# The normalization of the integral:
# The standard surface integral is \int_{-1}^1 \int_0^{2\pi} f^p du d\phi.
# Using GL: \approx \sum_i w_i \sum_j f(u_i, \phi_j)^p * (2\pi / N_PHI) ?
# Actually, the quadrature rule for the double integral is:
# Q = \frac{2\pi}{N_{PHI}} \sum_{j=0}^{N_{PHI}-1} \sum_{i=0}^{N_{THETA}-1} w_i f(u_i, \phi_j)^p
# where w_i are the standard Gauss-Legendre weights for [-1, 1].
# Let's verify if this integrates 1 to 4*pi.
# \sum w_i = 2. \sum_j 1 = N_PHI.
# Q = \frac{2\pi}{N_{PHI}} * N_{PHI} * 2 = 4\pi. Correct.

u_nodes, w_weights = np.polynomial.legendre.leggauss(N_THETA)
phi_nodes = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)

# Precompute the Spherical Harmonics Y_m(u, phi) for m = -10..10
# Y_0 = N_0 P_10^0(u)
# Y_m (m>0) = sqrt(2) N_m P_10^m(u) cos(m phi)
# Y_m (m<0) = sqrt(2) N_|m| P_10^{|m|}(u) sin(|m| phi)
# N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)

# We will store Y as a matrix of shape (N_THETA * N_PHI, 21)
# Indexing: flat_index = i * N_PHI + j
# m_index = m + 10

# Calculate factorials
def fact(n):
    return math.factorial(n)

Y_matrix = np.zeros((N_THETA * N_PHI, 21), dtype=np.float64)

for m in range(-L, L + 1):
    abs_m = abs(m)
    # Normalization N_m
    # N_m = sqrt( (2L+1)/(4*pi) * (L-abs_m)! / (L+abs_m)! )
    # L=10, 2L+1 = 21
    N_m = math.sqrt(21.0 / (4.0 * np.pi) * fact(L - abs_m) / fact(L + abs_m))
    
    # P_L^m(u) with Condon-Shortley phase
    # scipy.special.lpmv(m, n, x) includes the Condon-Shortley phase (-1)^m
    # We need P_10^{|m|}(u)
    P_vals = lpmv(abs_m, L, u_nodes) # Shape (N_THETA,)
    
    if m == 0:
        # Y_0 = N_0 P_10^0(u)
        # Broadcast to all phi
        Y_vals = np.tile(P_vals * N_m, (N_PHI, 1)).T # Shape (N_PHI, N_THETA) -> Transpose to (N_THETA, N_PHI) ? 
        # Let's stick to (N_THETA, N_PHI) for clarity, then flatten
        Y_flat = np.tile(P_vals * N_m, (1, N_PHI)).flatten()
    else:
        if m > 0:
            # Y_m = sqrt(2) N_m P_10^m(u) cos(m phi)
            # P_vals shape (N_THETA,), phi_nodes shape (N_PHI,)
            # We need outer product structure
            cos_term = np.cos(m * phi_nodes) # (N_PHI,)
            # Y(i, j) = sqrt(2) N_m P_vals[i] cos_term[j]
            Y_2d = np.sqrt(2.0) * N_m * np.outer(P_vals, cos_term) # (N_THETA, N_PHI)
            Y_flat = Y_2d.flatten()
        else:
            # Y_m = sqrt(2) N_|m| P_10^{|m|}(u) sin(|m| phi)
            sin_term = np.sin(abs_m * phi_nodes) # (N_PHI,)
            Y_2d = np.sqrt(2.0) * N_m * np.outer(P_vals, sin_term) # (N_THETA, N_PHI)
            Y_flat = Y_2d.flatten()
    
    Y_matrix[:, m + 10] = Y_flat

def compute_ratio(c):
    """
    Computes R = ||f||_4 / ||f||_2
    c: array of 21 coefficients
    """
    # f = Y_matrix @ c
    f = Y_matrix @ c
    f2 = f * f
    f4 = f2 * f2
    
    # Quadrature weights
    # The weight for each point (i, j) in the sum is w_i * (2*pi / N_PHI)
    # We can precompute the vector of weights for the flattened array
    # w_flat[i * N_PHI + j] = w_weights[i] * (2*pi / N_PHI)
    
    # To save time in the loop, precompute w_flat outside? 
    # But compute_ratio is called many times. 
    # We can pass w_flat as a global or closure.
    global w_flat
    norm2_sq = np.dot(f2, w_flat)
    norm4_sq = np.dot(f4, w_flat)
    
    if norm2_sq <= 1e-15:
        return 1e15 # Large penalty for zero function
    
    norm2 = math.sqrt(norm2_sq)
    norm4 = math.sqrt(math.sqrt(norm4_sq)) # L4 norm is (integral |f|^4)^{1/4}
    
    return norm4 / norm2

# Precompute flattened weights
w_flat = np.repeat(w_weights, N_PHI) * (2.0 * np.pi / N_PHI)

def objective(c):
    """
    Objective function for minimization.
    We minimize R. 
    """
    return compute_ratio(c)

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0 # 10s safety margin
    
    best_c = None
    best_score = 1e15
    
    # 1. Generate diverse initial seeds
    # Strategy: Mix of random, structured, and Latin Hypercube
    
    seeds = []
    
    # A. Single-m seeds (each m individually)
    for i in range(21):
        c = np.zeros(21)
        c[i] = 1.0
        seeds.append(c)
        
    # B. Pairs of m (symmetric/antisymmetric)
    for i in range(21):
        for j in range(i, 21):
            c = np.zeros(21)
            c[i] = 1.0
            c[j] = 1.0
            seeds.append(c)
            
    # C. Random normal seeds
    rng = np.random.default_rng(42)
    for _ in range(200):
        c = rng.standard_normal(21)
        seeds.append(c)
        
    # D. Latin Hypercube Sampling for better coverage
    # Generate 100 LHS points
    n_lhs = 100
    lhs = np.zeros((n_lhs, 21))
    for i in range(n_lhs):
        for j in range(21):
            lhs[i, j] = rng.random() * 2 - 1 # Uniform in [-1, 1]
    seeds.extend([row for row in lhs])
    
    # Shuffle seeds to ensure variety in time allocation
    rng.shuffle(seeds)
    
    # 2. Optimization Loop
    # Use Nelder-Mead
    # Options: maxiter, xatol, fatol
    # We want to be fast. 
    # Since R is scale invariant (R(kc) = R(c)), we should normalize c or handle it.
    # Nelder-Mead might struggle with scale invariance if it tries to shrink/grow.
    # However, the ratio is homogeneous of degree 0. 
    # To help optimization, we can fix the L2 norm to 1 during optimization?
    # No, the problem asks for coefficients. The ratio is invariant to scaling.
    # If we normalize c to unit L2 norm before evaluating, the optimization landscape is on the sphere.
    # Nelder-Mead works in Euclidean space. 
    # A common trick: minimize R(c) but project c to unit L2 norm at each step?
    # Or just minimize R(c) directly. Since R is scale invariant, the gradient is orthogonal to c.
    # Nelder-Mead should still work, but might drift in scale.
    # Let's stick to direct minimization of R(c).
    
    # To improve stability, we can normalize the initial seeds to have L2 norm = 1.
    # This puts them on the unit sphere in coefficient space, which is a compact manifold.
    
    for c0 in seeds:
        if time.time() > deadline:
            break
            
        # Normalize initial guess to unit L2 norm to help with scale invariance
        # Calculate L2 norm of the harmonic f, not just c.
        # Wait, L2 norm of f is sqrt(c^T G c) where G is the Gram matrix of Y_m.
        # Since Y_m are orthonormal on the sphere, G is Identity?
        # "real orthonormal degree-10 harmonics". Yes, they are orthonormal.
        # So ||f||_2^2 = sum c_m^2.
        # So normalizing c to unit Euclidean norm makes ||f||_2 = 1.
        
        norm_c = np.linalg.norm(c0)
        if norm_c < 1e-6:
            continue
        c0_norm = c0 / norm_c
        
        # Run Nelder-Mead
        try:
            res = scipy.optimize.minimize(
                objective, 
                c0_norm, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            c_opt = res.x
            score = objective(c_opt)
            
            if score < best_score:
                best_score = score
                best_c = c_opt.copy()
                
        except Exception:
            continue
            
    # If no optimization succeeded, fallback to best seed
    if best_c is None:
        # Return a valid default, e.g., m=10
        best_c = np.zeros(21)
        best_c[20] = 1.0
        
    # Ensure validity: max|c| < 1e-3 or > 1e5 is invalid.
    # Our optimization should keep it reasonable, but let's check.
    max_c = np.max(np.abs(best_c))
    if max_c < 1e-3:
        # Scale up
        best_c = best_c / max_c * 1.0
    elif max_c > 1e5:
        # Scale down
        best_c = best_c / max_c * 1.0
        
    return best_c

# EVOLVE-BLOCK-END
