import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def normalize(P):
        return P / np.linalg.norm(P, axis=1, keepdims=True)
    
    # 2. Helper to compute covering radius given a set of test points
    def compute_covering_radius(P, test_points):
        # P: (N, 3), test_points: (M, 3)
        # Compute angular distances
        dots = test_points @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        return np.max(min_dists)

    # 3. Helper to get critical points (Hull circumcenters)
    def get_critical_points(P):
        try:
            hull = ConvexHull(P)
            critical = [P]
            for simplex in hull.simplices:
                A, B, C = P[simplex]
                # Find circumcenter of triangle ABC on the sphere
                # The circumcenter is the intersection of the plane through ABC and the line perpendicular to the plane through the origin?
                # No, for points on a sphere, the circumcenter of the spherical triangle is the point on the sphere equidistant from A, B, C.
                # This is the normalized cross product of (B-A) and (C-A)? No.
                # The circumcenter of a triangle on a sphere is the intersection of the sphere with the line passing through the origin and the circumcenter of the planar triangle ABC.
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
                # The vector (Ux, Uy, Uz) is the circumcenter in 3D space.
                # Since A,B,C are on the unit sphere, the circumcenter of the spherical triangle is normalize(U).
                # Wait, the planar circumcenter is NOT on the sphere. The spherical circumcenter is the intersection of the sphere and the line from origin to the planar circumcenter?
                # Actually, the point on the sphere equidistant from A, B, C is the normalized vector sum of A, B, C? No, that's the centroid.
                # The spherical circumcenter is the intersection of the sphere with the line passing through the origin and the planar circumcenter of the triangle formed by A, B, C?
                # Yes. The plane containing A, B, C does not pass through the origin (unless they are collinear with origin, which is not the case for a hull facet).
                # The normal to the plane is n = (B-A) x (C-A).
                # The circumcenter C_s satisfies C_s . A = C_s . B = C_s . C = r_c.
                # C_s is in the span of A, B, C.
                # A simpler way: The spherical circumcenter is the normalized vector:
                # v = (B-A) x (C-A) is the normal. This is the centroid of the *great circle* triangle? No.
                # Let's use the property that the circumcenter is the point on the sphere that maximizes the minimum distance? No, it's the point equidistant.
                # The vector u = A + B + C is the centroid.
                # The correct formula for the circumcenter of a spherical triangle with vertices A, B, C (unit vectors) is:
                # If the triangle is small, it's approx the planar circumcenter.
                # Generally, the circumcenter is the normalized vector:
                # w = (A x B) x (A x C) ? No.
                # Let's solve for w = x*A + y*B + z*C such that w.w = 1 and w.A = w.B = w.C.
                # w.A = w.B => x(A.A - B.A) + y(B.A - B.B) + z(C.A - C.B) = 0
                # This is getting complex. 
                # Alternative: The circumcenter of the facet is the point on the sphere furthest from the interior of the facet?
                # Actually, for a convex hull facet, the "worst" point is the circumcenter of the triangle formed by the three vertices, IF the circumradius is the largest.
                # The circumcenter of a triangle on a unit sphere can be found by normalizing the vector:
                # n = (B-A) x (C-A)
                # The circumcenter is NOT simply normalize(n).
                # However, we can use a numerical solver or a geometric formula.
                # Formula: The circumcenter C is the intersection of the sphere and the plane defined by A, B, C? No, the plane of the triangle is not the plane of the circumcenter.
                # The circumcenter lies on the line passing through the origin and the circumcenter of the planar triangle?
                # Yes. Let C_planar be the circumcenter of the planar triangle ABC.
                # Then C_sphere = C_planar / ||C_planar||.
                # We already computed U = (Ux, Uy, Uz) which is C_planar.
                U_vec = np.array([Ux, Uy, Uz])
                norm_U = np.linalg.norm(U_vec)
                if norm_U > 1e-9:
                    C_sphere = U_vec / norm_U
                    # Check if C_sphere is "outside" the triangle (i.e., the cap is empty)
                    # For a convex hull facet, the circumcenter is always on the opposite side of the plane from the origin?
                    # Actually, we just need to include it as a candidate for the worst point.
                    critical.append(C_sphere)
            
            # Also include the original points as candidates (though they are distance 0)
            # And maybe some random points if the hull is degenerate?
            # For now, just use the circumcenters.
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
    best_P = P
    best_val = np.inf
    
    # Phase 1: Coarse optimization with a dense sample to get out of the initial spiral
    M_coarse = 2000
    j = np.arange(M_coarse) + 0.5
    z_s = 1.0 - 2.0 * j / M_coarse
    phi_s = np.pi * (1.0 + np.sqrt(5.0)) * j
    r_s = np.sqrt(1.0 - z_s * z_s)
    S_coarse = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])
    
    def objective_coarse(flat_P):
        P_flat = flat_P.reshape(N, 3)
        P_norm = P_flat / np.linalg.norm(P_flat, axis=1, keepdims=True)
        return compute_covering_radius(P_norm, S_coarse)
    
    x0 = P.flatten()
    try:
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
            P_norm = P_flat / np.linalg.norm(P_flat, axis=1, keepdims=True)
            return compute_covering_radius(P_norm, critical_pts)
        
        x_current = best_P.flatten()
        try:
            # Use Powell for robustness, or L-BFGS-B if we had gradients
            res = minimize(objective_fine, x_current, method='Powell', options={'maxiter': 50, 'tol': 1e-6})
            if res.fun < best_val:
                best_P = res.x.reshape(N, 3)
                best_P = normalize(best_P)
                best_val = res.fun
        except Exception:
            break

    # Final check and normalization
    best_P = normalize(best_P)
    
    # Ensure no coincident points (add small jitter if needed, though optimization should prevent this)
    # Check for validity
    if not np.all(np.isfinite(best_P)):
        # Fallback to initial
        best_P = P
    
    return best_P

# EVOLVE-BLOCK-END
