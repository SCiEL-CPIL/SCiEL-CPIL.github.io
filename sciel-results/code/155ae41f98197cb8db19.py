import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimization parameters
    step_size = 0.01
    iterations = 2000
    
    for it in range(iterations):
        # Compute Convex Hull
        try:
            hull = ConvexHull(P)
            faces = hull.simplices
        except Exception:
            # If hull fails (e.g., points coplanar or inside hemisphere), add small noise
            P += np.random.normal(0, 0.01, P.shape)
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            continue
        
        # Calculate circumradius for each face
        # We want to find the face with the maximum circumradius
        max_radius = -1
        worst_face_idx = -1
        worst_circumcenter = None
        
        for f_idx, face in enumerate(faces):
            p1, p2, p3 = P[face[0]], P[face[1]], P[face[2]]
            
            # Compute side lengths (angular distances)
            # a is opposite p1 (between p2, p3)
            # b is opposite p2 (between p1, p3)
            # c is opposite p3 (between p1, p2)
            
            def ang_dist(u, v):
                d = np.dot(u, v)
                d = np.clip(d, -1.0, 1.0)
                return np.arccos(d)
            
            a = ang_dist(p2, p3)
            b = ang_dist(p1, p3)
            c = ang_dist(p1, p2)
            
            # Spherical Law of Cosines for angles
            # cos(alpha) = (cos(a) - cos(b)cos(c)) / (sin(b)sin(c))
            
            sin_b = np.sin(b)
            sin_c = np.sin(c)
            sin_a = np.sin(a)
            
            if sin_b * sin_c < 1e-9:
                # Degenerate
                continue
                
            cos_a = np.cos(a)
            cos_b = np.cos(b)
            cos_c = np.cos(c)
            
            cos_alpha = (cos_a - cos_b * cos_c) / (sin_b * sin_c)
            cos_beta = (cos_b - cos_a * cos_c) / (sin_a * sin_c)
            cos_gamma = (cos_c - cos_a * cos_b) / (sin_a * sin_b)
            
            # Clip for stability
            cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
            cos_beta = np.clip(cos_beta, -1.0, 1.0)
            cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
            
            alpha = np.arccos(cos_alpha)
            beta = np.arccos(cos_beta)
            gamma = np.arccos(cos_gamma)
            
            # Circumcenter vector direction
            # C = p1 * sin(2*alpha) + p2 * sin(2*beta) + p3 * sin(2*gamma)
            # Note: This formula works if the circumcenter is on the same side of the great circle.
            # For acute spherical triangles, this is standard.
            # If the triangle is obtuse, the circumcenter might be "far away", but the formula still gives the point equidistant.
            
            s2a = np.sin(2 * alpha)
            s2b = np.sin(2 * beta)
            s2g = np.sin(2 * gamma)
            
            C_vec = p1 * s2a + p2 * s2b + p3 * s2g
            C_norm = np.linalg.norm(C_vec)
            
            if C_norm < 1e-9:
                # Degenerate case, e.g., right triangle where circumcenter is midpoint of hypotenuse?
                # Or points are antipodal?
                continue
                
            C = C_vec / C_norm
            
            # The circumradius is the angular distance from C to any vertex
            # R = arccos(C . p1)
            dot_cp1 = np.clip(np.dot(C, p1), -1.0, 1.0)
            R = np.arccos(dot_cp1)
            
            # We want to minimize the MAX R. So we track the max R.
            if R > max_radius:
                max_radius = R
                worst_face_idx = f_idx
                worst_circumcenter = C
                worst_face = face
        
        if worst_face_idx == -1:
            # No valid faces found?
            break
            
        # Move the vertices of the worst face away from the circumcenter
        # The gradient of the distance from a point p to C on the sphere is - (C - p) projected to tangent?
        # Actually, we want to move p away from C to increase the distance from C to p?
        # No, the covering radius is the distance from the *farthest point* (which is C for this facet) to the *nearest point*.
        # The nearest point to C for this facet is one of the vertices (if the triangle is acute).
        # If we move the vertices *away* from C, the circumradius increases?
        # Wait. The circumcenter is the point equidistant to the vertices.
        # If we move a vertex $p_i$ away from $C$, the distance $d(p_i, C)$ increases.
        # But the circumcenter $C$ will also move.
        # Generally, to *reduce* the circumradius of a triangle, we should move the vertices *towards* the circumcenter?
        # No, that shrinks the triangle, but might make it more "obtuse" or change the circumcenter.
        # Let's think: If we have an equilateral triangle, and we move all 3 vertices towards the center, the triangle shrinks, and the circumradius decreases.
        # So we should move vertices *towards* the circumcenter.
        
        # Correction:
        # The "worst" point on the sphere is the circumcenter $C$.
        # The distance from $C$ to the nearest configuration point is $R$.
        # To reduce $R$, we need to bring some points closer to $C$.
        # The points defining the facet are the closest to $C$.
        # So we should move the 3 vertices of the worst facet *towards* $C$.
        
        p1, p2, p3 = P[worst_face[0]], P[worst_face[1]], P[worst_face[2]]
        
        # Move towards C
        # p_new = p + step * (C - p)
        # Then normalize
        
        P[worst_face[0]] = p1 + step_size * (worst_circumcenter - p1)
        P[worst_face[1]] = p2 + step_size * (worst_circumcenter - p2)
        P[worst_face[2]] = p3 + step_size * (worst_circumcenter - p3)
        
        # Normalize all points to prevent drift
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        
        # Decay step size
        step_size *= 0.999
        
    return P
