import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 180.0  # Safety margin before 210s limit
    
    # Factorials needed: 0! = 1, 4! = 24, 8! = 40320, 12! = 479001600
    fact = {0: 1.0, 4: 24.0, 8: 40320.0, 12: 479001600.0}
    
    def compute_exact_integrals(c0, c4, c8):
        """Compute I0, Ix2, Ideriv exactly using Hermite orthogonality."""
        c12 = -(c0 + 3.0*c4 + 105.0*c8) / 10395.0
        c = {0: c0, 4: c4, 8: c8, 12: c12}
        degrees = [0, 4, 8, 12]
        
        # I0 = sum_k c_k^2 * k!
        I0 = 0.0
        for k in degrees:
            I0 += c[k]**2 * fact[k]
        
        # Ix2 = sum over k,l of c_k c_l * <H_k x^2 H_l>
        # x^2 H_k = 0.5*H_{k+2} + k*H_k + 0.5*k*(k-1)*H_{k-2}
        Ix2 = 0.0
        for k in degrees:
            ck = c[k]
            if abs(ck) < 1e-30:
                continue
            # Contribution from 0.5*H_{k+2}
            if (k+2) in fact:
                Ix2 += ck * c[k+2] * 0.5 * fact[k+2]
            # Contribution from k*H_k
            Ix2 += ck * c[k] * k * fact[k]
            # Contribution from 0.5*k*(k-1)*H_{k-2}
            if k >= 2 and (k-2) in fact:
                Ix2 += ck * c[k-2] * 0.5 * k * (k-1) * fact[k-2]
        
        # Ideriv = sum over k,l of c_k c_l * <H_k' H_l'>
        # H_k' = 2k H_{k-1}, so H_k' H_l' = 4kl H_{k-1} H_{l-1}
        # <H_{k-1} H_{l-1}> = (k-1)! delta_{k-1,l-1} = (k-1)! delta_{k,l}
        Ideriv = 0.0
        for k in degrees:
            if k == 0:
                continue  # H_0' = 0
            ck = c[k]
            Ideriv += ck**2 * 4.0 * k * k * fact[k-1]
        
        return I0, Ix2, Ideriv
    
    def objective_and_grad(coeffs):
        """Compute the uncertainty product and its gradient analytically."""
        c0, c4, c8 = coeffs
        try:
            I0, Ix2, Ideriv = compute_exact_integrals(c0, c4, c8)
            if I0 <= 0 or not np.isfinite(I0):
                return 1e10, np.zeros(3)
            if Ix2 < 0 or not np.isfinite(Ix2):
                return 1e10, np.zeros(3)
            if Ideriv < 0 or not np.isfinite(Ideriv):
                return 1e10, np.zeros(3)
            
            A_f_sq = Ix2 / I0
            A_gf_sq = (1.0 / (4.0 * np.pi**2)) * (Ideriv / I0)
            
            if A_f_sq <= 0 or A_gf_sq <= 0:
                return 1e10, np.zeros(3)
            
            val = np.sqrt(A_f_sq * A_gf_sq)
            
            # Compute gradient analytically
            c12 = -(c0 + 3.0*c4 + 105.0*c8) / 10395.0
            c = {0: c0, 4: c4, 8: c8, 12: c12}
            degrees = [0, 4, 8, 12]
            
            dc12_dc = np.array([-1.0/10395.0, -3.0/10395.0, -105.0/10395.0])
            
            dI0_dc = np.zeros(3)
            for i, k in enumerate([0, 4, 8]):
                dI0_dc[i] = 2.0 * c[k] * fact[k] + 2.0 * c[12] * fact[12] * dc12_dc[i]
            
            M = np.zeros((4, 4))
            for idx_k, k in enumerate(degrees):
                for idx_l, l in enumerate(degrees):
                    M_val = 0.0
                    if k + 2 == l:
                        M_val += 0.5 * fact[k+2]
                    if k == l:
                        M_val += k * fact[k]
                    if k >= 2 and k - 2 == l:
                        M_val += 0.5 * k * (k-1) * fact[k-2]
                    M[idx_k, idx_l] = M_val
            
            dIx2_dc = np.zeros(3)
            c_arr = np.array([c[0], c[4], c[8], c[12]])
            for i in range(3):
                s1 = 0.0
                for b in range(4):
                    s1 += c_arr[b] * M[i, b]
                s2 = 0.0
                for a in range(4):
                    s2 += c_arr[a] * M[3, a]
                dIx2_dc[i] = 2.0 * s1 + 2.0 * s2 * dc12_dc[i]
            
            dIderiv_dc = np.zeros(3)
            for i, k in enumerate([0, 4, 8]):
                term1 = 0.0
                if k > 0:
                    term1 = 2.0 * c[k] * 4.0 * k * k * fact[k-1]
                term2 = 2.0 * c[12] * 4.0 * 12 * 12 * fact[11] * dc12_dc[i]
                dIderiv_dc[i] = term1 + term2
            
            sqrt_term = np.sqrt(Ix2 * Ideriv)
            if sqrt_term <= 0 or not np.isfinite(sqrt_term):
                return val, np.zeros(3)
            
            grad_corrected = np.zeros(3)
            for i in range(3):
                term1 = (1.0 / (2.0 * np.pi * I0 * 2.0 * sqrt_term)) * (Ideriv * dIx2_dc[i] + Ix2 * dIderiv_dc[i])
                term2 = -sqrt_term / (2.0 * np.pi * I0**2) * dI0_dc[i]
                grad_corrected[i] = term1 + term2
            
            return val, grad_corrected
        except Exception:
            return 1e10, np.zeros(3)
    
    def objective(coeffs):
        val, _ = objective_and_grad(coeffs)
        return val

    # Initial guess from seed (Goncalves et al. 2017)
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    best_score, _ = objective_and_grad(x0)
    best_coeffs = x0.copy()
    
    # Phase 1: Structured Grid Search
    # We explore a box around the seed. The seed is known to be near the optimum.
    # We use a coarse grid to ensure we don't miss a better local minimum.
    # Box: c0 in [-1.5, -0.8], c4 in [-0.1, 0.2], c8 in [-0.1, 0.2]
    # This is a reasonable range based on the seed and typical variations.
    
    n_grid = 25 # 25x25x25 = 15625 evaluations. Each is fast (analytic).
    # 15625 * ~0.001s = ~15s. Safe.
    
    c0_vals = np.linspace(-1.5, -0.8, n_grid)
    c4_vals = np.linspace(-0.1, 0.2, n_grid)
    c8_vals = np.linspace(-0.1, 0.2, n_grid)
    
    # We will store the top K candidates from the grid
    top_candidates = []
    K = 10
    
    # To speed up the grid, we can vectorize the integral computation if possible, 
    # but the current function is scalar. Let's just loop.
    # 15k iterations is fine in Python if the body is light.
    
    # Optimization: Pre-compute constants
    # Actually, let's just call objective.
    
    for c0 in c0_vals:
        if time.time() > deadline - 60: # Keep 60s for refinement
            break
        for c4 in c4_vals:
            for c8 in c8_vals:
                coeffs = np.array([c0, c4, c8])
                val, _ = objective_and_grad(coeffs)
                
                if np.isfinite(val) and val < best_score:
                    best_score = val
                    best_coeffs = coeffs.copy()
                
                # Maintain top K
                top_candidates.append((val, coeffs.copy()))
                if len(top_candidates) > K:
                    top_candidates.sort(key=lambda x: x[0])
                    top_candidates.pop()
    
    # Phase 2: Refine the top candidates from the grid
    # We take the unique best points from the grid and polish them.
    
    # Remove duplicates or very close points
    unique_candidates = []
    for val, coeffs in top_candidates:
        is_unique = True
        for _, c2 in unique_candidates:
            if np.linalg.norm(coeffs - c2) < 1e-4:
                is_unique = False
                break
        if is_unique:
            unique_candidates.append((val, coeffs))
    
    # Also include the seed and the current best
    unique_candidates.append((best_score, best_coeffs))
    
    # Sort by score
    unique_candidates.sort(key=lambda x: x[0])
    
    # Refine top 5 unique candidates
    for val, coeffs in unique_candidates[:5]:
        if time.time() > deadline - 10:
            break
        
        # Use L-BFGS-B for high precision
        try:
            res = opt.minimize(objective, coeffs, jac=lambda x: objective_and_grad(x)[1],
                              method='L-BFGS-B', 
                              options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
        
        # Follow up with Nelder-Mead for any remaining flatness
        try:
            res = opt.minimize(objective, best_coeffs, method='Nelder-Mead', 
                              options={'xatol': 1e-15, 'fatol': 1e-16, 'maxiter': 1000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Phase 3: Final Polish on the absolute best
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(objective, best_coeffs, jac=lambda x: objective_and_grad(x)[1],
                              method='L-BFGS-B', 
                              options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 200})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
