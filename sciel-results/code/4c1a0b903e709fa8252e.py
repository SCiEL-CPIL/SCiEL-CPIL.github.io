# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.special import lpmv

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0

    L = 10
    N_THETA = 70
    N_PHI = 130

    u_nodes, w_weights = np.polynomial.legendre.leggauss(N_THETA)
    phi_nodes = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)

    # Precompute Y_matrix: shape (N_THETA * N_PHI, 21)
    Y_matrix = np.zeros((N_THETA * N_PHI, 21), dtype=np.float64)
    w_flat = np.repeat(w_weights, N_PHI) * (2.0 * np.pi / N_PHI)

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

    # Precompute Y^2 and Y^4 matrices for faster evaluation
    # Y_sq_matrix[i, j] = Y_matrix[i, j]^2
    # Y_four_matrix[i, j] = Y_matrix[i, j]^4
    # But we need f = Y @ c, so f^2 = (Y @ c)^2.
    # Let's just use Y_matrix.
    
    def compute_ratio_and_grad(c):
        """
        Computes R = ||f||_4 / ||f||_2 and its gradient w.r.t c.
        f = Y @ c
        ||f||_2^2 = c^T G_2 c, where G_2 = Y^T W Y
        ||f||_4^4 = (c^T Y W Y^T (Y c)^3)^T ... complicated.
        
        Let's use the chain rule.
        R = N4 / N2
        N2 = (c^T G_2 c)^0.5
        N4 = (sum w_i f_i^4)^0.25
        
        dR/dc = (1/N2) dN4/dc - (N4/N2^2) dN2/dc
        
        dN2/dc = (1/(2 N2)) * 2 G_2 c = G_2 c / N2
        dN4/dc = (1/(4 N4^3)) * 4 sum w_i f_i^3 Y_i = (1/N4^3) Y^T (w * f^3)
        
        So grad_R = (1/N2) * (1/N4^3) Y^T (w * f^3) - (N4/N2^2) * (G_2 c / N2)
        """
        f = Y_matrix @ c
        f2 = f * f
        f4 = f2 * f2
        
        norm2_sq = np.dot(f2, w_flat)
        norm4_sq = np.dot(f4, w_flat)
        
        if norm2_sq <= 1e-15:
            return 1e15, np.zeros_like(c)
        
        norm2 = math.sqrt(norm2_sq)
        norm4 = math.pow(norm4_sq, 0.25)
        
        R = norm4 / norm2
        
        # Gradient
        # G_2 c = Y^T (w * f)
        G2_c = Y_matrix.T @ (w_flat * f)
        
        # Y^T (w * f^3)
        Yt_wf3 = Y_matrix.T @ (w_flat * (f * f2))
        
        term1 = Yt_wf3 / (norm2 * norm4**3)
        term2 = (norm4 / (norm2**3)) * G2_c
        
        grad = term1 - term2
        
        return R, grad

    def project_to_sphere(c):
        n = np.linalg.norm(c)
        if n < 1e-9:
            return np.zeros_like(c)
        return c / n

    best_c = None
    best_score = 1e15
    
    # 1. Generate diverse seeds
    rng = np.random.default_rng(42)
    seeds = []
    
    # Single m
    for i in range(21):
        c = np.zeros(21)
        c[i] = 1.0
        seeds.append(c)
        
    # Pairs
    for i in range(21):
        for j in range(i+1, 21):
            c = np.zeros(21)
            c[i] = 1.0
            c[j] = 1.0
            seeds.append(c)
            
    # Random
    for _ in range(100):
        c = rng.standard_normal(21)
        seeds.append(c)
        
    # Evaluate seeds
    seed_scores = []
    for c0 in seeds:
        if time.time() > deadline - 10:
            break
        c_norm = project_to_sphere(c0)
        score, _ = compute_ratio_and_grad(c_norm)
        seed_scores.append((score, c_norm))
        
    seed_scores.sort(key=lambda x: x[0])
    
    # 2. Conjugate Gradient on the Sphere
    # We will run CG from the top 10 seeds.
    # At each step:
    # 1. Compute gradient g at current point x on sphere.
    # 2. Project g onto tangent space: g_tan = g - (g . x) x
    # 3. Compute step direction p.
    # 4. Perform line search for step size alpha.
    # 5. Update x_new = (x + alpha * p) / ||x + alpha * p||
    
    def cg_optimize(x0, max_iter=500):
        x = x0.copy()
        score, g = compute_ratio_and_grad(x)
        best_local_score = score
        best_local_x = x.copy()
        
        # Project gradient
        proj_g = g - np.dot(g, x) * x
        # Negative gradient direction
        p = -proj_g
        norm_p = np.linalg.norm(p)
        if norm_p < 1e-10:
            return best_local_x, best_local_score
            
        p = p / norm_p
        
        alpha = 0.01
        prev_score = score
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Line search
            # Try a few step sizes
            best_alpha = alpha
            best_val = 1e15
            
            for _ in range(5):
                x_trial = x + best_alpha * p
                x_trial = project_to_sphere(x_trial)
                val, _ = compute_ratio_and_grad(x_trial)
                if val < best_val:
                    best_val = val
                    best_alpha = alpha # Actually we want to find the best alpha in a small range
                # Let's do a simple backtracking or golden section? 
                # Let's just try a few fixed multipliers of alpha
                # alpha * 0.5, alpha * 1.0, alpha * 2.0, alpha * 0.1, alpha * 5.0
                pass
            
            # More robust line search
            alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
            best_alpha_val = 1e15
            best_alpha_found = 0.1
            best_x_trial = None
            
            for a in alphas:
                x_trial = x + a * p
                x_trial = project_to_sphere(x_trial)
                val, _ = compute_ratio_and_grad(x_trial)
                if val < best_alpha_val:
                    best_alpha_val = val
                    best_alpha_found = a
                    best_x_trial = x_trial
            
            if best_x_trial is None:
                break
                
            if best_alpha_val < best_local_score:
                best_local_score = best_alpha_val
                best_local_x = best_x_trial.copy()
                
            x = best_x_trial
            score = best_alpha_val
            
            # Update gradient
            _, g = compute_ratio_and_grad(x)
            proj_g = g - np.dot(g, x) * x
            norm_g = np.linalg.norm(proj_g)
            if norm_g < 1e-8:
                break
            
            # Fletcher-Reeves beta
            # beta = (g_new . g_new) / (g_old . g_old)
            # We need to store old proj_g
            # For simplicity, let's restart CG every few steps (non-linear CG)
            if it % 10 == 0:
                p = -proj_g
                norm_p = np.linalg.norm(p)
                if norm_p < 1e-10:
                    break
                p = p / norm_p
            else:
                # Simple momentum
                p_new = -proj_g + 0.5 * p # Simple heuristic
                norm_p = np.linalg.norm(p_new)
                if norm_p < 1e-10:
                    break
                p = p_new / norm_p

        return best_local_x, best_local_score

    # Run CG from top seeds
    for score, c0 in seed_scores[:10]:
        if time.time() > deadline - 30:
            break
        
        c_opt, score_opt = cg_optimize(c0)
        
        if score_opt < best_score:
            best_score = score_opt
            best_c = c_opt.copy()
            
    # 3. Final Polish with Nelder-Mead if time permits
    if time.time() < deadline - 20 and best_c is not None:
        try:
            from scipy.optimize import minimize
            
            def obj_nm(x):
                n = np.linalg.norm(x)
                if n < 1e-9:
                    return 1e15
                c = x / n
                val, _ = compute_ratio_and_grad(c)
                return val
            
            res = minimize(
                obj_nm, 
                best_c, 
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            c_nm = res.x
            c_nm = project_to_sphere(c_nm)
            score_nm, _ = compute_ratio_and_grad(c_nm)
            
            if score_nm < best_score:
                best_score = score_nm
                best_c = c_nm.copy()
                
        except Exception:
            pass

    if best_c is None:
        best_c = np.zeros(21)
        best_c[20] = 1.0

    # Validation
    max_c = np.max(np.abs(best_c))
    if max_c < 1e-3:
        best_c = best_c / max_c * 1.0
    elif max_c > 1e5:
        best_c = best_c / max_c * 1.0
        
    return best_c

# EVOLVE-BLOCK-END
