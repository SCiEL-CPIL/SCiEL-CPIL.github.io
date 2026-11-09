# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.special import lpmv
from scipy.optimize import minimize

# Constants
L = 10
N_THETA = 70
N_PHI = 130

# Precompute quadrature nodes and weights
u_nodes, w_weights = np.polynomial.legendre.leggauss(N_THETA)
phi_nodes = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)

# Precompute flattened weights
w_flat = np.repeat(w_weights, N_PHI) * (2.0 * np.pi / N_PHI)

# Precompute Spherical Harmonics Y_matrix
Y_matrix = np.zeros((N_THETA * N_PHI, 21), dtype=np.float64)

def fact(n):
    return math.factorial(n)

for m in range(-L, L + 1):
    abs_m = abs(m)
    N_m = math.sqrt(21.0 / (4.0 * np.pi) * fact(L - abs_m) / fact(L + abs_m))
    P_vals = lpmv(abs_m, L, u_nodes)
    
    if m == 0:
        Y_flat = np.tile(P_vals * N_m, (1, N_PHI)).flatten()
    else:
        if m > 0:
            cos_term = np.cos(m * phi_nodes)
            Y_2d = np.sqrt(2.0) * N_m * np.outer(P_vals, cos_term)
            Y_flat = Y_2d.flatten()
        else:
            sin_term = np.sin(abs_m * phi_nodes)
            Y_2d = np.sqrt(2.0) * N_m * np.outer(P_vals, sin_term)
            Y_flat = Y_2d.flatten()
    
    Y_matrix[:, m + 10] = Y_flat

def compute_ratio(c):
    """
    Computes R = ||f||_4 / ||f||_2
    """
    f = Y_matrix @ c
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.dot(f2, w_flat)
    norm4_sq = np.dot(f4, w_flat)
    
    if norm2_sq <= 1e-15:
        return 1e15
    
    norm2 = math.sqrt(norm2_sq)
    norm4 = math.sqrt(math.sqrt(norm4_sq))
    
    return norm4 / norm2

def gradient_real(c):
    """
    Computes analytic gradient of R(c) in real space.
    """
    f = Y_matrix @ c
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.dot(f2, w_flat)
    norm4_sq = np.dot(f4, w_flat)
    
    if norm2_sq <= 1e-15 or norm4_sq <= 1e-15:
        return np.zeros(21)
        
    norm2 = math.sqrt(norm2_sq)
    norm4 = math.sqrt(math.sqrt(norm4_sq))
    
    # A_k = sum_i w_i f_i^3 Y_{i,k} = (f^3 * w_flat) @ Y_matrix
    # B_k = sum_i w_i f_i Y_{i,k} = (f * w_flat) @ Y_matrix
    
    A = (f**3 * w_flat) @ Y_matrix
    B = (f * w_flat) @ Y_matrix
    
    # dR/dc = (N4^(-3) / N2) * A - (N4 / N2^3) * B
    coeff_A = (norm4 ** -3) / norm2
    coeff_B = norm4 / (norm2 ** 3)
    
    grad = coeff_A * A - coeff_B * B
    
    return grad

def map_complex_to_real(z):
    """
    Map complex vector z (length 11) to real coefficients c (length 21).
    z[0] corresponds to c[10] (m=0)
    z[m] corresponds to c[10+m] (cos) and c[10-m] (sin) for m=1..10
    Note: The task defines c[i] for m = i-10.
    So c[10] is m=0.
    c[10+m] is m>0 (cos).
    c[10-m] is m<0 (sin).
    """
    c = np.zeros(21)
    c[10] = z[0].real
    for m in range(1, 11):
        c[10 + m] = z[m].real
        c[10 - m] = z[m].imag
    return c

def map_real_to_complex(c):
    """
    Map real coefficients c (length 21) to complex vector z (length 11).
    """
    z = np.zeros(11, dtype=np.complex128)
    z[0] = c[10]
    for m in range(1, 11):
        z[m] = c[10 + m] + 1j * c[10 - m]
    return z

def objective_complex(z):
    """
    Objective function in complex space.
    """
    c = map_complex_to_real(z)
    return compute_ratio(c)

def gradient_complex(z):
    """
    Gradient of objective with respect to complex parameters z.
    We use the chain rule: dJ/dz = (dJ/dc) @ (dc/dz)
    """
    c = map_complex_to_real(z)
    grad_c = gradient_real(c)
    
    # grad_z is complex
    grad_z = np.zeros(11, dtype=np.complex128)
    grad_z[0] = grad_c[10]
    for m in range(1, 11):
        # d c[10+m] / d z[m].real = 1
        # d c[10-m] / d z[m].imag = 1
        # So grad_z[m] = grad_c[10+m] + i * grad_c[10-m]
        grad_z[m] = grad_c[10 + m] + 1j * grad_c[10 - m]
        
    return grad_z

def riemannian_gradient_descent(z_init, max_iter=200, tol=1e-8, deadline=0.0):
    """
    Riemannian Gradient Descent on the complex sphere S^20.
    """
    z = z_init / np.linalg.norm(z_init)
    best_z = z.copy()
    best_val = objective_complex(z)
    
    lr = 0.01
    momentum = 0.9
    v = np.zeros(11, dtype=np.complex128)
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Compute gradient in complex space
        g = gradient_complex(z)
        
        # Project gradient to tangent space of sphere
        # proj(g) = g - <g, z> z
        # Inner product for complex is sum(conj(g) * z)
        inner = np.vdot(g, z) # <g, z> = sum(conj(g) * z)
        grad_tan = g - inner * z
        
        # Momentum update
        v = momentum * v + grad_tan
        
        # Update step
        step = -lr * v
        
        # Move and re-normalize
        z_new = z + step
        norm_z = np.linalg.norm(z_new)
        if norm_z < 1e-12:
            break
        z_new = z_new / norm_z
        
        # Line search (Backtracking)
        f_old = objective_complex(z)
        f_new = objective_complex(z_new)
        
        # Simple backtracking if it increases
        if f_new > f_old:
            # Try smaller steps
            accepted = False
            for alpha in [0.5, 0.25, 0.1, 0.05, 0.01]:
                z_trial = z - alpha * lr * v
                norm_trial = np.linalg.norm(z_trial)
                if norm_trial < 1e-12:
                    continue
                z_trial = z_trial / norm_trial
                f_trial = objective_complex(z_trial)
                if f_trial < f_old:
                    z_new = z_trial
                    f_new = f_trial
                    accepted = True
                    break
            if not accepted:
                # If no step accepted, reduce learning rate and break or continue
                lr *= 0.5
                if lr < 1e-6:
                    break
                # Skip this step
                continue
        
        z = z_new
        v = v # Keep momentum
        
        if f_new < best_val:
            best_val = f_new
            best_z = z.copy()
            
        # Convergence check
        if abs(f_old - f_new) < tol:
            break
            
    return best_z, best_val

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0 # 10s safety margin
    
    best_c = None
    best_score = 1e15
    
    # Initialize with diverse starting points in complex space
    rng = np.random.default_rng(42)
    
    # 1. Single m seeds (real and imaginary parts)
    seeds = []
    for i in range(11):
        z = np.zeros(11, dtype=np.complex128)
        z[i] = 1.0
        seeds.append(z)
        z[i] = 1j
        seeds.append(z)
        
    # 2. Random seeds
    for _ in range(20):
        z = rng.standard_normal(11) + 1j * rng.standard_normal(11)
        seeds.append(z)
        
    # 3. Pairs of m (real)
    for i in range(11):
        for j in range(i+1, 11):
            z = np.zeros(11, dtype=np.complex128)
            z[i] = 1.0
            z[j] = 1.0
            seeds.append(z)
            
    for z_seed in seeds:
        if time.time() > deadline:
            break
            
        # Run Riemannian Gradient Descent
        z_opt, val = riemannian_gradient_descent(z_seed, max_iter=100, deadline=deadline)
        
        c_opt = map_complex_to_real(z_opt)
        
        # Polish with L-BFGS-B in real space
        # Define objective and gradient for L-BFGS-B
        def obj_real(c):
            return compute_ratio(c)
            
        def grad_real(c):
            return gradient_real(c)
            
        # Constraint: norm(c) = 1 is not strictly required for L-BFGS-B if we handle scaling,
        # but since the function is scale invariant, we can just optimize c directly.
        # However, L-BFGS-B might drift in scale.
        # Better: Optimize on the sphere? L-BFGS-B doesn't support manifold constraints directly.
        # We can use a penalty or just run it and normalize at the end.
        # Given the scale invariance, the gradient is orthogonal to the radial direction?
        # Let's check: dR/dc = ...
        # If we scale c by k, f scales by k.
        # ||kf||_4 = k ||f||_4, ||kf||_2 = k ||f||_2.
        # R(kc) = R(c).
        # So the gradient should be orthogonal to c?
        # dR/dc . c = 0?
        # Let's verify.
        # If so, L-BFGS-B will stay on the sphere (approximately) if started on it.
        
        # Start L-BFGS-B from z_opt mapped to real
        res = minimize(
            obj_real, 
            c_opt, 
            jac=grad_real, 
            method='L-BFGS-B', 
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        c_polished = res.x
        # Normalize to be safe
        norm_c = np.linalg.norm(c_polished)
        if norm_c > 1e-12:
            c_polished = c_polished / norm_c
            
        score_polished = compute_ratio(c_polished)
        
        if score_polished < best_score:
            best_score = score_polished
            best_c = c_polished.copy()
            
    if best_c is None:
        best_c = np.zeros(21)
        best_c[20] = 1.0
        
    # Ensure validity
    max_c = np.max(np.abs(best_c))
    if max_c < 1e-3:
        best_c = best_c / max_c * 1.0
    elif max_c > 1e5:
        best_c = best_c / max_c * 1.0
        
    return best_c

# EVOLVE-BLOCK-END
