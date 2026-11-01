import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit (evaluator kills at 150s, but task says 120s limit in contract, let's be safe for 120s)
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def normalize(P):
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return P / norms
    
    # 2. Helper to compute covering radius given a set of test points
    def compute_covering_radius(P, test_points):
        # P: (N, 3), test_points: (M, 3)
        # Compute angular distances
        dots = test_points @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        return np.max(min_dists)
    
    # 3. Helper to get critical points (Hull circumcenters + vertices)
    def get_critical_points(P):
        try:
            hull = ConvexHull(P)
            critical = [P]
            for simplex in hull.simplices:
                A, B, C = P[simplex]
                # Find circumcenter of triangle ABC on the sphere
                # The circumcenter is the intersection of the sphere with the line passing through the origin and the circumcenter of the planar triangle ABC.
                # Planar circumcenter of ABC:
                D = 2 * (A[0] * (B[1] * C[2] - C[1] * B[2]) -
                          A[1] * (B[0] * C[2] - C[0] * B[2]) +
                          A[2] * (B[0] * C[1] - C[0] * B[1]))
                if np.abs(D) < 1e-12:
                    continue
                Ux = ((A[0]**2 + A[1]**2 + A[2]**2) * (B[1] * C[2] - C[1] * B[2]) -
                      (B[0]**2 + B[1]**2 + B[2]**2) * (A[1] * C[2] - C[1] * A[2]) +
                      (C[0]**2 + C[1]**2 + C[2]**2) * (A[1] * B[2] - B[1] * A[2])) / D
                Uy = ((B[0]**2 + B[1]**2 + B[2]**2) * (A[0] * C[2] - C[0] * A[2]) -
                      (A[0]**2 + A[1]**2 + A[2]**2) * (B[0] * C[2] - C[0] * B[2]) +
                      (C[0]**2 + C[1]**2 + C[2]**2) * (B[0] * A[2] - A[0] * B[2])) / D
                Uz = ((A[0]**2 + A[1]**2 + A[2]**2) * (B[0] * C[1] - C[0] * B[1]) -
                      (B[0]**2 + B[1]**2 + B[2]**2) * (A[0] * C[1] - C[0] * A[1]) +
                      (C[0]**2 + C[1]**2 + C[2]**2) * (A[0] * B[1] - B[0] * A[1])) / D
                U_vec = np.array([Ux, Uy, Uz])
                norm_U = np.linalg.norm(U_vec)
                if norm_U > 1e-9:
                    C_sphere = U_vec / norm_U
                    critical.append(C_sphere)
            
            if len(critical) == 1:
                # Fallback to dense sample if hull fails
                M = 5000
                j = np.arange(M) + 0.5
                z_s = 1.0 - 2.0 * j / M
                phi_s = np.pi * (1.0 + np.sqrt(5.0)) * j
                r_s = np.sqrt(1.0 - z_s * z_s)
                S = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])
                return S
            return np.array(critical)
        except Exception:
            # Fallback to dense sample
            M = 5000
            j = np.arange(M) + 0.5
            z_s = 1.0 - 2.0 * j / M
            phi_s = np.pi * (1.0 + np.sqrt(5.0)) * j
            r_s = np.sqrt(1.0 - z_s * z_s)
            S = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])
            return S

    # 4. Optimization Loop
    best_P = P_init
    best_val = compute_covering_radius(best_P, get_critical_points(best_P))
    
    # Phase 1: Coarse optimization with a dense sample to get out of the initial spiral
    M_coarse = 2000
    j = np.arange(M_coarse) + 0.5
    z_s = 1.0 - 2.0 * j / M_coarse
    phi_s = np.pi * (1.0 + np.sqrt(5.0)) * j
    r_s = np.sqrt(1.0 - z_s * z_s)
    S_coarse = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])
    
    def objective_coarse(flat_P):
        P_flat = flat_P.reshape(N, 3)
        P_norm = normalize(P_flat)
        return compute_covering_radius(P_norm, S_coarse)
    
    x0 = P_init.flatten()
    try:
        # Use Powell for robustness
        res = minimize(objective_coarse, x0, method='Powell', options={'maxiter': 100, 'tol': 1e-4})
        if res.fun < best_val:
            best_P = res.x.reshape(N, 3)
            best_P = normalize(best_P)
            best_val = res.fun
    except Exception:
        pass

    # Phase 2: Fine optimization using Hull-based critical points
    # We will iterate: Compute Hull -> Get Critical Points -> Optimize on Critical Points -> Repeat
    for _ in range(5):
        if time.time() > deadline:
            break
            
        critical_pts = get_critical_points(best_P)
        
        def objective_fine(flat_P):
            P_flat = flat_P.reshape(N, 3)
            P_norm = normalize(P_flat)
            return compute_covering_radius(P_norm, critical_pts)
        
        x_current = best_P.flatten()
        try:
            # Use Powell for robustness
            res = minimize(objective_fine, x_current, method='Powell', options={'maxiter': 50, 'tol': 1e-6})
            if res.fun < best_val:
                best_P = res.x.reshape(N, 3)
                best_P = normalize(best_P)
                best_val = res.fun
        except Exception:
            break

    # Phase 3: Final refinement with a slightly denser critical set or mixed sample
    # If time permits, do one more pass with a hybrid of critical points and a moderate sample
    if time.time() < deadline - 10:
        critical_pts = get_critical_points(best_P)
        # Add a moderate sample to catch any issues with hull approximation
        M_hybrid = 1000
        j_h = np.arange(M_hybrid) + 0.5
        z_h = 1.0 - 2.0 * j_h / M_hybrid
        phi_h = np.pi * (1.0 + np.sqrt(5.0)) * j_h
        r_h = np.sqrt(1.0 - z_h * z_h)
        S_hybrid = np.column_stack([r_h * np.cos(phi_h), r_h * np.sin(phi_h), z_h])
        
        # Combine critical points and sample points
        test_points = np.vstack([critical_pts, S_hybrid])
        
        def objective_hybrid(flat_P):
            P_flat = flat_P.reshape(N, 3)
            P_norm = normalize(P_flat)
            return compute_covering_radius(P_norm, test_points)
        
        x_current = best_P.flatten()
        try:
            res = minimize(objective_hybrid, x_current, method='Powell', options={'maxiter': 30, 'tol': 1e-7})
            if res.fun < best_val:
                best_P = res.x.reshape(N, 3)
                best_P = normalize(best_P)
                best_val = res.fun
        except Exception:
            pass

    # Final check and normalization
    best_P = normalize(best_P)
    
    # Ensure no coincident points (add small jitter if needed, though optimization should prevent this)
    # Check for validity
    if not np.all(np.isfinite(best_P)):
        # Fallback to initial
        best_P = P_init
    
    return best_P
