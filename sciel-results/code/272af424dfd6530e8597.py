import numpy as np
import scipy.spatial as sp
import scipy.optimize as opt
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Helper: Compute covering radius and face data
    def compute_metrics(pts):
        # Ensure points are normalized
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        try:
            hull = sp.ConvexHull(pts)
        except:
            return None, None, None
            
        faces = hull.simplices
        R_sq_sum = 0.0
        face_centroids = np.zeros((len(faces), 3))
        face_areas = np.zeros(len(faces))
        
        for idx, face in enumerate(faces):
            a, b, c = pts[face[0]], pts[face[1]], pts[face[2]]
            
            # Gram matrix
            G = np.array([
                [a.dot(a), a.dot(b), a.dot(c)],
                [b.dot(a), b.dot(b), b.dot(c)],
                [c.dot(a), c.dot(b), c.dot(c)]
            ])
            
            # Solve for circumradius R
            # cos(R) = 1 / sqrt(1^T G^-1 1)
            try:
                G_inv = np.linalg.inv(G)
                ones = np.ones(3)
                val = np.dot(ones, np.dot(G_inv, ones))
                if val <= 1e-12:
                    continue
                cos_R = 1.0 / np.sqrt(val)
                # Clamp cos_R to [-1, 1] for safety
                cos_R = np.clip(cos_R, -1.0, 1.0)
                R = np.arccos(cos_R)
                R_sq_sum += R * R
                
                # Centroid of spherical triangle
                # Approximation: normalized sum of vertices weighted by something?
                # Better: The centroid of the spherical triangle is the normalized vector:
                # c = (a*sin(A) + b*sin(B) + c*sin(C)) / (2 * sin(A/2)sin(B/2)sin(C/2) ... )?
                # Simpler approximation for Lloyd: The "mass center" of the triangle.
                # For small triangles, (a+b+c)/3 is good.
                # For better accuracy, we can use the formula for the vector integral of the triangle.
                # Integral of x over spherical triangle T is:
                # I = (a x b + b x c + c x a) * (1 / (2 * (1 + a.b + b.c + c.a))) ? No.
                # Let's use the normalized sum of vertices for now, it's robust for Lloyd.
                # To improve, weight by the "area" contribution?
                # Actually, the centroid of the spherical triangle is the pole of the great circle 
                # that is the "center of mass" of the arc?
                # Let's stick to the normalized sum of vertices for the Lloyd step, 
                # but weighted by the edge lengths? No, just simple average.
                centroid = a + b + c
                norm_c = np.linalg.norm(centroid)
                if norm_c > 1e-10:
                    centroid = centroid / norm_c
                else:
                    centroid = a # Fallback
                face_centroids[idx] = centroid
                
                # Area of spherical triangle
                # Area = 2 * (sin(A/2)sin(B/2)sin(C/2) ... )?
                # Area = A + B + C - pi
                # Angles:
                # cos(A) = (b.c - a.b*a.c) / (sqrt(1-(a.b)^2) * sqrt(1-(a.c)^2))
                # This is expensive. Approximate area by planar area for small triangles?
                # Or just use 1.0 as weight for simplicity in Lloyd.
                # Using equal weights is standard for simple Lloyd.
                face_areas[idx] = 1.0 
                
            except:
                R_sq_sum += 1.0 # Penalty for degenerate
                face_centroids[idx] = a
                face_areas[idx] = 1.0
                
        # Aggregate centroids for each point
        point_centroids = np.zeros((N, 3))
        for idx, face in enumerate(faces):
            i, j, k = face[0], face[1], face[2]
            w = face_areas[idx]
            point_centroids[i] += w * face_centroids[idx]
            point_centroids[j] += w * face_centroids[idx]
            point_centroids[k] += w * face_centroids[idx]
            
        # Normalize
        norms = np.linalg.norm(point_centroids, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        point_centroids = point_centroids / norms
        
        return R_sq_sum, point_centroids, hull

    # Phase 1: Lloyd's Iterations
    for _ in range(50):
        if time.time() > deadline:
            break
        metrics, centroids, hull = compute_metrics(points)
        if metrics is None:
            break
        points = centroids

    # Phase 2: L-BFGS Optimization
    # Objective: Sum of squared circumradii
    def objective(x):
        pts = x.reshape(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        R_sq_sum, _, _ = compute_metrics(pts)
        return R_sq_sum

    x0 = points.flatten()
    
    # Use L-BFGS-B
    try:
        res = opt.minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        if time.time() < deadline:
            points = res.x.reshape(N, 3)
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
    except:
        pass

    # Final normalization
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
# EVOLVE-BLOCK-END
