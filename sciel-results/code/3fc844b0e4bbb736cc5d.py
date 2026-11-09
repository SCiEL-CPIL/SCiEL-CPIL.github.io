# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.special import lpmv
from scipy.optimize import minimize, differential_evolution

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0

    # 1. Precompute Spherical Harmonics
    L = 10
    N_THETA = 70
    N_PHI = 130

    u_nodes, w_weights = np.polynomial.legendre.leggauss(N_THETA)
    phi_nodes = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)

    # Matrix of Y_m values at quadrature points
    # Shape: (N_THETA * N_PHI, 21)
    # m = -10..10 maps to index 0..20
    Y_matrix = np.zeros((N_THETA * N_PHI, 21), dtype=np.float64)
    
    # Precompute factorials for normalization
    def fact(n):
        return math.factorial(n)

    for m in range(-L, L + 1):
        abs_m = abs(m)
        # N_m = sqrt( (2L+1)/(4*pi) * (L-abs_m)! / (L+abs_m)! )
        N_m = math.sqrt(21.0 / (4.0 * np.pi) * fact(L - abs_m) / fact(L + abs_m))
        
        # P_L^m(u) with Condon-Shortley phase
        # scipy.special.lpmv(m, n, x) includes Condon-Shortley phase
        P_vals = lpmv(abs_m, L, u_nodes)  # Shape (N_THETA,)
        
        if m == 0:
            # Y_0 = N_0 P_L^0(u)
            # Repeat for all phi
            Y_flat = np.tile(P_vals * N_m, (1, N_PHI)).flatten()
        else:
            if m > 0:
                # Y_m = sqrt(2) N_m P_L^m(u) cos(m phi)
                cos_term = np.cos(m * phi_nodes)  # (N_PHI,)
                Y_2d = np.sqrt(2.0) * N_m * np.outer(P_vals, cos_term)
                Y_flat = Y_2d.flatten()
            else:
                # Y_m = sqrt(2) N_|m| P_L^{|m|}(u) sin(|m| phi)
                sin_term = np.sin(abs_m * phi_nodes)  # (N_PHI,)
                Y_2d = np.sqrt(2.0) * N_m * np.outer(P_vals, sin_term)
                Y_flat = Y_2d.flatten()
        
        Y_matrix[:, m + 10] = Y_flat

    # Precompute quadrature weights for flattened array
    # Weight for point (i, j) is w_weights[i] * (2*pi / N_PHI)
    w_flat = np.repeat(w_weights, N_PHI) * (2.0 * np.pi / N_PHI)

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
            return 1e15  # Penalty for zero function
        
        norm2 = math.sqrt(norm2_sq)
        norm4 = math.pow(norm4_sq, 0.25)  # L4 norm is (integral |f|^4)^{1/4}
        
        return norm4 / norm2

    def objective_sphere(x):
        """
        Objective function for optimization on the unit sphere.
        x is a vector of 21 coefficients. We normalize it to unit L2 norm.
        """
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-9:
            return 1e15
        c = x / norm_x
        return compute_ratio(c)

    # 2. Optimization Strategy
    best_c = None
    best_score = 1e15
    
    # Generate diverse seeds
    seeds = []
    rng = np.random.default_rng(42)
    
    # A. Single m seeds
    for i in range(21):
        c = np.zeros(21)
        c[i] = 1.0
        seeds.append(c)
        
    # B. Pairs of m
    for i in range(21):
        for j in range(i+1, 21):
            c = np.zeros(21)
            c[i] = 1.0
            c[j] = 1.0
            seeds.append(c)
            
    # C. Random normal
    for _ in range(200):
        c = rng.standard_normal(21)
        seeds.append(c)
        
    # Shuffle
    rng.shuffle(seeds)
    
    # First pass: Evaluate all seeds quickly
    for c0 in seeds:
        if time.time() > deadline:
            break
        norm_c = np.linalg.norm(c0)
        if norm_c < 1e-6:
            continue
        c_norm = c0 / norm_c
        score = compute_ratio(c_norm)
        if score < best_score:
            best_score = score
            best_c = c_norm.copy()
            
    # Second pass: Local refinement using Nelder-Mead on the best few seeds
    # Select top 20 seeds
    seed_scores = []
    for c0 in seeds:
        if time.time() > deadline - 100:  # Reserve time for refinement
            break
        norm_c = np.linalg.norm(c0)
        if norm_c < 1e-6:
            continue
        c_norm = c0 / norm_c
        score = compute_ratio(c_norm)
        seed_scores.append((score, c_norm))
        
    seed_scores.sort(key=lambda x: x[0])
    
    top_seeds = seed_scores[:20]
    
    for score, c0 in top_seeds:
        if time.time() > deadline:
            break
            
        try:
            # Optimize starting from c0
            # Since we are on the sphere, we can just run Nelder-Mead on the objective_sphere
            # Nelder-Mead doesn't constrain, but if we start on the sphere and step size is small,
            # it might drift. However, the ratio is scale-invariant, so the gradient is tangent to the sphere.
            # Actually, Nelder-Mead works on R^n. If we minimize R(x) where R is scale invariant,
            # the minimum is flat along the ray. This is bad for NM.
            # Better to project back to sphere at each step? Or use a constrained optimizer?
            # Let's use a trick: Minimize R(x) + lambda * (||x||^2 - 1)^2 ?
            # Or just use scipy.optimize.minimize with bounds? No, sphere is not box.
            
            # Let's try a simple projection method:
            # Run NM, but at each function evaluation, normalize the input.
            # This effectively makes the search happen on the sphere.
            def obj_proj(x):
                n = np.linalg.norm(x)
                if n < 1e-9:
                    return 1e15
                return compute_ratio(x / n)
            
            res = minimize(
                obj_proj, 
                c0, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            c_opt = res.x
            n_opt = np.linalg.norm(c_opt)
            if n_opt > 1e-9:
                c_opt = c_opt / n_opt
            score_opt = compute_ratio(c_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_c = c_opt.copy()
                
        except Exception:
            continue

    # Third pass: Global search using Differential Evolution if time permits
    if time.time() < deadline - 60:
        try:
            # DE requires bounds. Let's bound coefficients to [-1, 1].
            # Since the ratio is scale invariant, the magnitude doesn't matter, only direction.
            # So bounding to [-1, 1] is safe as long as we normalize at the end.
            bounds = [(-1.0, 1.0)] * 21
            
            def de_objective(x):
                n = np.linalg.norm(x)
                if n < 1e-9:
                    return 1e15
                return compute_ratio(x / n)
            
            # Use a small population and limited iterations to fit in time
            res_de = differential_evolution(
                de_objective, 
                bounds, 
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                init='sobol'
            )
            
            if res_de.success or (res_de.fun < best_score):
                c_de = res_de.x
                n_de = np.linalg.norm(c_de)
                if n_de > 1e-9:
                    c_de = c_de / n_de
                score_de = compute_ratio(c_de)
                
                if score_de < best_score:
                    best_score = score_de
                    best_c = c_de.copy()
        except Exception:
            pass

    # Final check and validation
    if best_c is None:
        best_c = np.zeros(21)
        best_c[20] = 1.0
        
    max_c = np.max(np.abs(best_c))
    if max_c < 1e-3:
        best_c = best_c / max_c * 1.0
    elif max_c > 1e5:
        best_c = best_c / max_c * 1.0
        
    return best_c

# EVOLVE-BLOCK-END
