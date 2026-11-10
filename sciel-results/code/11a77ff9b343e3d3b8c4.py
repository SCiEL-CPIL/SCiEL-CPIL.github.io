import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv
import math

def rudin_problem10() -> np.ndarray:
       k = 10
       # Quadrature setup
       x_gl, w_gl = np.polynomial.legendre.leggauss(70)
       n_phi = 130
       phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
       
       w_phi = 2 * np.pi / n_phi
       weights_2d = np.outer(w_gl, np.full(n_phi, w_phi))
       W_flat = weights_2d.flatten()
       
       n_theta = 70
       total_points = n_theta * n_phi
       
       # Precompute N_m
       N = np.zeros(11)
       for m_abs in range(11):
           log_num = math.lgamma(11 - m_abs)
           log_den = math.lgamma(11 + m_abs)
           N[m_abs] = math.sqrt(21.0 / (4.0 * np.pi) * math.exp(log_num - log_den))
           
       # Precompute P_vals
       P_vals = np.zeros((11, 70), dtype=np.float64)
       for m_abs in range(11):
           P_vals[m_abs, :] = lpmv(m_abs, 10, x_gl)
           
       # Compute Y matrix
       Y_vals = np.zeros((total_points, 21), dtype=np.float64)
       sqrt2 = math.sqrt(2.0)
       
       for i in range(n_theta):
           x_i = x_gl[i]
           for j in range(n_phi):
               phi_j = phi[j]
               p = i * n_phi + j
               
               Y_vals[p, 10] = N[0] * P_vals[0, i]
               
               for m in range(1, 11):
                   idx_pos = 10 + m
                   Y_vals[p, idx_pos] = sqrt2 * N[m] * P_vals[m, i] * np.cos(m * phi_j)
                   idx_neg = 10 - m
                   Y_vals[p, idx_neg] = sqrt2 * N[m] * P_vals[m, i] * np.sin(m * phi_j)
                   
       # Compute K tensor
       # K[m1, m2, m3, m4] = sum_p W_p Y[p, m1] Y[p, m2] Y[p, m3] Y[p, m4]
       K = np.zeros((21, 21, 21, 21), dtype=np.float64)
       # Vectorized update
       # Y_vals shape (P, 21)
       # We can do:
       # K += np.einsum('p, pi, pj, pk, pl -> i j k l', W_flat, Y_vals, Y_vals, Y_vals, Y_vals)
       # This might be memory intensive or slow if not optimized.
       # Let's try a loop over points to be safe with memory, or just use einsum.
       # 9100 points.
       # np.einsum('p, pi, pj, pk, pl -> i j k l', W_flat, Y_vals, Y_vals, Y_vals, Y_vals)
       # This creates a 21^4 array.
       # Let's check if it fits in memory. 21^4 * 8 bytes = 1.5 MB. Yes.
       # The operation is 9100 * 21^4 multiplications/additions.
       # This is fast in NumPy.
       
       K = np.einsum('p, pi, pj, pk, pl -> i j k l', W_flat, Y_vals, Y_vals, Y_vals, Y_vals, optimize=True)
       
       # Objective function: minimize J(c) = (c^T K c) / ||c||^4
       # Gradient: grad_k = (4 * (K c)_k * ||c||^4 - 2 * J(c) * c_k * ||c||^4) / ||c||^8
       #           = (4 * (K c)_k - 2 * J(c) * c_k) / ||c||^2
       # where (K c)_k = sum_{i,j,l} K[k,i,j,l] c_i c_j c_l
       
       def objective_and_grad(c):
           c = np.asarray(c)
           norm2 = np.dot(c, c)
           if norm2 < 1e-12:
               return 0.0, np.zeros_like(c)
           
           # Compute Kc
           # Kc_k = sum_{i,j,l} K[k,i,j,l] c_i c_j c_l
           # np.einsum('klij, i, j, l -> k', K, c, c, c)
           Kc = np.einsum('klij, i, j, l -> k', K, c, c, c, optimize=True)
           
           # Numerator of J
           num = np.dot(c, Kc)
           
           # J
           J = num / (norm2 * norm2)
           
           # Gradient
           # grad_k = (4 * Kc_k - 2 * J * c_k) / norm2
           grad = (4.0 * Kc - 2.0 * J * c) / norm2
           
           return J, grad
           
       # Optimization
       # We want to minimize J.
       # Start with several initial guesses.
       
       best_c = None
       best_val = np.inf
       
       # Deterministic starts
       starts = [
           np.ones(21) / np.sqrt(21),
           np.array([1.0 if m == 0 else 0.0 for m in range(-10, 11)]) / 1.0,
           np.array([1.0 if m == 10 else 0.0 for m in range(-10, 11)]) / 1.0,
           np.array([1.0 if m == -10 else 0.0 for m in range(-10, 11)]) / 1.0,
       ]
       
       # Random starts
       np.random.seed(42)
       for _ in range(20):
           starts.append(np.random.randn(21) / np.sqrt(21))
       
       for start in starts:
           try:
               # Use L-BFGS-B or BFGS?
               # L-BFGS-B is good for large N, but N=21 is small.
               # BFGS with analytical gradient is robust.
               res = minimize(
                   objective_and_grad,
                   start,
                   jac=True,
                   method='BFGS',
                   options={'maxiter': 1000, 'gtol': 1e-8}
               )
               if res.success:
                   c_opt = res.x
                   val, _ = objective_and_grad(c_opt)
                   if val < best_val:
                       best_val = val
                       best_c = c_opt
           except Exception:
               pass
               
       if best_c is None:
           # Fallback
           best_c = starts[0]
           
       # Normalize result to satisfy validity checks if needed, 
       # though the objective is scale invariant, so the scale of c doesn't change the score.
       # However, the task says "returning 21 real coefficients c".
       # It doesn't specify the norm of c.
       # But "max|c| < 1e-3 or max|c| > 1e5" is invalid.
       # So we should ensure c is in a reasonable range.
       # Normalizing to unit norm is a safe bet.
       best_c = best_c / np.linalg.norm(best_c)
       
       return best_c
