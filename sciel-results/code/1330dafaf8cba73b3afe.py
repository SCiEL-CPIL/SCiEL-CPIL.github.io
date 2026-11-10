import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Uses spectral-domain formulation with precomputed 4th-order Clebsch-Gordan
    tensor for fast objective evaluation in coefficient space.
    """
    start_time = time.time()
    deadline = start_time + 595

    k = 10
    n_m = 2 * k + 1  # 21 coefficients

    # 1. Setup Quadrature Grid for computing K tensor
    mu, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi

    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    n_points = len(mu_flat)

    # 2. Construct Basis Functions
    A = np.zeros((n_points, n_m))
    
    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs, exact=False) / factorial(10 + m_abs, exact=False))
        return np.sqrt(val)
    
    # m = 0
    P = lpmv(0, 10, mu_flat)
    N = get_norm(0)
    A[:, 10] = N * P
    
    # m > 0 and m < 0
    for m in range(1, k + 1):
        P = lpmv(m, 10, mu_flat)
        N = get_norm(m)
        sqrt2_N = np.sqrt(2.0) * N
        A[:, m + 10] = sqrt2_N * P * np.cos(m * phi_flat)
        A[:, -m + 10] = sqrt2_N * P * np.sin(m * phi_flat)

    # 3. Precompute 4th-order tensor K[m1,m2,m3,m4] = integral Y_m1 * Y_m2 * Y_m3 * Y_m4 dOmega
    # K[i,j,k,l] = sum_p w_p * d_phi * A[p,i] * A[p,j] * A[p,k] * A[p,l]
    # To avoid storing full 21^4 tensor (194481 entries is fine), compute efficiently:
    # For each pair (i,j), precompute S_ij[p] = A[p,i]*A[p,j]
    # Then K[i,j,k,l] = sum_p w_p * S_ij[p] * S_kl[p]
    
    # Precompute all pairwise products
    S = np.zeros((n_m, n_m, n_points))
    for i in range(n_m):
        for j in range(n_m):
            S[i, j, :] = A[:, i] * A[:, j]
    
    # Compute K tensor: K[i,j,k,l] = d_phi * sum_p w_p * S[i,j,p] * S[k,l,p]
    # Use matrix multiplication: for fixed (i,j), K[i,j,k,l] = d_phi * S[i,j,:] @ diag(w) @ S[k,l,:].T
    # More efficiently: K_flat = d_phi * (S_weighted @ S_weighted.T) where S_weighted reshapes
    # Let's do it as: for each (i,j), compute vector v_ij = sqrt(w) * S[i,j,:]
    # Then K[i,j,k,l] = d_phi * v_ij . v_kl
    
    sqrt_w = np.sqrt(w_flat)
    S_weighted = S * sqrt_w[None, None, :]  # shape (n_m, n_m, n_points)
    
    # K[i,j,k,l] = d_phi * sum_p S_weighted[i,j,p] * S_weighted[k,l,p]
    # Reshape to use matmul: K_flat[i*n_m+j, k*n_m+l] = d_phi * S_weighted_flat @ S_weighted_flat.T
    S_flat = S_weighted.reshape(n_m * n_m, n_points)
    K_flat = d_phi * (S_flat @ S_flat.T)  # shape (n_m*n_m, n_m*n_m)
    K = K_flat.reshape(n_m, n_m, n_m, n_m)
    
    # 4. Objective in spectral domain
    # ||f||_2^2 = sum_m c_m^2 = ||c||^2 (since basis is orthonormal)
    # ||f||_4^4 = sum_{m1,m2,m3,m4} c_m1*c_m2*c_m3*c_m4 * K[m1,m2,m3,m4]
    #           = c^T (K contracted) c ... need to be careful
    # Actually: ||f||_4^4 = sum_{i,j,k,l} c_i c_j c_k c_l K[i,j,k,l]
    # This is a 4th-order form. We can write it as:
    # Let M_c[i,j] = sum_k c_k K[i,j,k,l] ... no, that's not right.
    # Better: ||f||_4^4 = sum_{i,j} c_i c_j * (sum_{k,l} c_k c_l K[i,j,k,l])
    # Define Q_c[i,j] = sum_{k,l} c_k c_l K[i,j,k,l]
    # Then ||f||_4^4 = sum_{i,j} c_i c_j Q_c[i,j] = c^T Q_c c
    # where Q_c = K[c,c] (contraction over last two indices)
    
    def compute_Q(c):
        """Compute Q[i,j] = sum_{k,l} c_k c_l K[i,j,k,l]"""
        # Q[i,j] = sum_k c_k * (sum_l c_l K[i,j,k,l])
        # = sum_k c_k * (K[i,j,:,:] @ c)
        # Efficiently: Q = np.einsum('ij kl, k, l -> ij', K, c, c)
        return np.einsum('ijkl,k,l->ij', K, c, c)
    
    def objective_and_grad(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return 1e10, np.zeros(n_m)
        u = c / norm_c
        
        # L2 norm squared = ||u||^2 = 1 (since u is on unit sphere)
        # But in general: ||f||_2^2 = sum c_m^2 for orthonormal basis
        norm2_sq = np.dot(u, u)
        if norm2_sq < 1e-12:
            return 1e10, np.zeros(n_m)
        
        # L4 norm to the 4th power
        Q = compute_Q(u)
        norm4_4 = np.dot(u, Q @ u)
        
        if norm4_4 < 1e-12:
            return 1e10, np.zeros(n_m)
        
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        R = N4 / N2
        
        # Gradient of R with respect to u (on sphere)
        # dR/du = (1/4) * N4^(-3) * d(norm4_4)/du / N2 - N4 * (1/(2*N2^3)) * d(norm2_sq)/du
        # d(norm2_sq)/du = 2*u
        # d(norm4_4)/du[i] = 2 * sum_j u_j * dQ[j,i]/du ... need to be careful
        # norm4_4 = sum_{i,j} u_i u_j Q_ij(u) where Q_ij = sum_{k,l} u_k u_l K[i,j,k,l]
        # d(norm4_4)/du[i] = 2*sum_j u_j * dQ[j,i]/du[i] + 2*Q[i,:].u ... this is getting complex
        # Let's use: norm4_4 = einsum('i,ij,j', u, Q, u) where Q = einsum('ijkl,k,l->ij', K, u, u)
        # d(norm4_4)/du[m] = 2*sum_{i,j} u_i * dQ[i,j]/du[m] * u_j + 2*sum_j Q[m,j]*u_j + 2*sum_i u_i*Q[i,m]
        # dQ[i,j]/du[m] = K[i,j,m,:].u + K[i,j,:,m].u
        # So d(norm4_4)/du[m] = 2*sum_{i,j} u_i*u_j*(K[i,j,m,:].u + K[i,j,:,m].u) + 4*sum_j Q[m,j]*u_j
        # By symmetry of K, K[i,j,m,:].u = K[i,j,:,m].u, so:
        # d(norm4_4)/du[m] = 4*sum_{i,j} u_i*u_j*K[i,j,m,:].u + 4*sum_j Q[m,j]*u_j
        # Let's define T[m] = sum_{i,j} u_i*u_j*K[i,j,m,:]  (vector of length n_m)
        # Then d(norm4_4)/du[m] = 4*(T[m].u + sum_j Q[m,j]*u_j)
        
        # Compute T: T[m] = einsum('ij,ijm->m', outer(u,u), K[:,:,m,:]) ... 
        # T[m] = sum_{i,j} u_i*u_j*K[i,j,m,l] for each l... no, T is a vector
        # T[m,l] = sum_{i,j} u_i*u_j*K[i,j,m,l]
        # Then d(norm4_4)/du[m] = 4*(sum_l T[m,l]*u_l + sum_j Q[m,j]*u_j)
        
        # Compute T: shape (n_m, n_m)
        T = np.einsum('ij,ijml->ml', np.outer(u, u), K)  # T[m,l] = sum_{i,j} u_i*u_j*K[i,j,m,l]
        
        d_norm4_4 = 4.0 * (T @ u + Q @ u)
        
        dR_du = (0.25 * N4**(-3) * d_norm4_4 / N2) - (N4 * u / (N2**3))
        
        # Project to tangent space
        u_dot_dR = np.dot(u, dR_du)
        grad_tangent = dR_du - u * u_dot_dR
        
        # Gradient with respect to c (accounting for u = c/||c||)
        grad_c = (grad_tangent - u * np.dot(u, grad_tangent)) / norm_c
        # Simplify: grad_c = grad_tangent / norm_c (since grad_tangent is already orthogonal to u)
        grad_c = grad_tangent / norm_c
        
        return R, grad_c

    # 5. Sphere-constrained gradient descent polish
    def sphere_polish(c, max_iter=2000, lr_init=0.05):
        c = c.copy()
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return c
        u = c / norm_c
        lr = lr_init
        
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            R_curr, grad_u = objective_and_grad(u)
            
            grad_tangent = grad_u - u * np.dot(u, grad_u)
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-14:
                break
            
            # Adaptive step size with backtracking
            step = lr
            improved = False
            for _ in range(25):
                u_new = u - step * grad_tangent
                u_new_norm = np.linalg.norm(u_new)
                if u_new_norm < 1e-12:
                    step *= 0.5
                    continue
                u_new = u_new / u_new_norm
                
                R_new, _ = objective_and_grad(u_new)
                
                if R_new < R_curr:
                    u = u_new
                    R_curr = R_new
                    improved = True
                    break
                else:
                    step *= 0.5
            
            if not improved or step < 1e-14:
                break
            lr = max(lr * 0.9995, 0.0005)
        
        return u

    # 6. Generate diverse initial candidates
    rng = np.random.default_rng(42)
    candidates = []
    
    # Pure modes
    for m in range(n_m):
        c = np.zeros(n_m)
        c[m] = 1.0
        candidates.append(c)
    
    # Pairs
    for i in range(n_m):
        for j in range(i+1, n_m):
            c = np.zeros(n_m)
            c[i] = 1.0
            c[j] = 1.0
            c /= np.linalg.norm(c)
            candidates.append(c)
    
    # Random
    for _ in range(200):
        c = rng.standard_normal(n_m)
        c /= np.linalg.norm(c)
        candidates.append(c)
    
    # Random with structure (emphasize certain m values)
    for _ in range(100):
        c = np.zeros(n_m)
        for m in range(n_m):
            if rng.random() < 0.5:
                c[m] = rng.standard_normal()
        if np.linalg.norm(c) > 1e-12:
            c /= np.linalg.norm(c)
            candidates.append(c)
    
    # 7. Evaluate all candidates
    candidate_vals = []
    for c_init in candidates:
        if time.time() > deadline - 60:
            break
        val, _ = objective_and_grad(c_init)
        candidate_vals.append((val, c_init))
    
    candidate_vals.sort(key=lambda x: x[0])
    
    # 8. Optimize top candidates
    best_c = None
    best_val = np.inf
    n_to_optimize = min(20, len(candidate_vals))
    
    for i in range(n_to_optimize):
        if time.time() > deadline - 15:
            break
        val_init, c_init = candidate_vals[i]
        
        try:
            res = minimize(
                objective_and_grad, 
                c_init, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
            )
            val = res.fun
            if val < best_val and np.all(np.isfinite(res.x)):
                best_val = val
                best_c = res.x.copy()
        except Exception:
            if val_init < best_val:
                best_val = val_init
                best_c = c_init.copy()
        
        # Sphere polish
        if best_c is not None and time.time() < deadline - 10:
            polished = sphere_polish(best_c, max_iter=1500, lr_init=0.05)
            val_p, _ = objective_and_grad(polished)
            if val_p < best_val:
                best_val = val_p
                best_c = polished.copy()

    # 9. Final high-precision polish
    if best_c is not None and time.time() < deadline - 5:
        for lr in [0.03, 0.01, 0.005, 0.001]:
            if time.time() > deadline - 5:
                break
            polished = sphere_polish(best_c, max_iter=2000, lr_init=lr)
            val_p, _ = objective_and_grad(polished)
            if val_p < best_val:
                best_val = val_p
                best_c = polished.copy()

    # 10. Final Validity Check and Scaling
    if best_c is None or not np.all(np.isfinite(best_c)):
        best_c = np.zeros(n_m)
        best_c[10] = 1.0
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c * (1e-2 / max_abs)
    elif max_abs > 1e5:
        best_c = best_c * (1e4 / max_abs)
        
    return best_c
