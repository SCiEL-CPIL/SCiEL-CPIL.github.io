import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def construct():
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimization
    start_time = time.time()
    max_time = 110.0 # Leave some buffer for 120s limit
    
    # Parameters
    step_size = 0.1
    decay = 0.99
    
    while time.time() - start_time < max_time:
        try:
            hull = ConvexHull(points)
            # Get the triangular faces
            faces = hull.simplices
            
            max_radius = 0
            max_face_idx = -1
            max_circumcenter = None
            
            for idx, face in enumerate(faces):
                v1, v2, v3 = points[face]
                
                # Calculate circumcenter and circumradius
                # Vector normal to the plane
                n = np.cross(v2 - v1, v3 - v1)
                if np.linalg.norm(n) < 1e-10:
                    continue # Degenerate face
                n = n / np.linalg.norm(n)
                
                # The circumcenter of the spherical triangle is the normalized vector
                # that is equidistant to v1, v2, v3.
                # It lies in the plane defined by v1, v2, v3.
                # The center of the circumcircle in the plane is given by solving:
                # c = v1 + a * (v2-v1) + b * (v3-v1)
                # |c - v1|^2 = |c - v2|^2 = |c - v3|^2
                
                # Alternatively, the circumcenter direction is proportional to:
                # c_dir = (|v2|^2 * (v3 - v1) - |v3|^2 * (v2 - v1)) x (v2 - v1) x (v3 - v1) ??
                # Since points are on unit sphere, |v1|=|v2|=|v3|=1.
                # Let's use the formula for circumcenter of a triangle on the unit sphere.
                # The circumcenter is the normalized vector:
                # c = (v1 + v2 + v3) is NOT correct in general.
                
                # Correct method:
                # The circumcenter $c$ satisfies $c \cdot v_1 = c \cdot v_2 = c \cdot v_3 = \cos(R)$.
                # And $c$ is in the plane of $v_1, v_2, v_3$.
                # So $c$ is a linear combination of $v_1, v_2, v_3$.
                # Actually, it's easier to find the plane equation $n \cdot x = d$.
                # $d = n \cdot v_1$.
                # The circumcenter in the plane is the projection of the origin? No.
                # The circumcenter of the triangle in 3D space (planar circumcenter) is $c_p$.
                # The spherical circumcenter is $c_s = c_p / |c_p|$.
                # And the angular radius is $R = \arccos(d / |c_p|)$? No.
                # $\cos(R) = \frac{c_p \cdot v_1}{|c_p| \cdot 1} = \frac{d}{|c_p|}$.
                
                # Let's find the planar circumcenter $c_p$.
                # Let $u = v_2 - v_1$, $w = v_3 - v_1$.
                # $c_p = v_1 + a u + b w$
                # $|c_p - v_1|^2 = |c_p - v_2|^2 \implies 2 a (u \cdot u) + 2 b (w \cdot u) = u \cdot u$
                # $|c_p - v_1|^2 = |c_p - v_3|^2 \implies 2 a (u \cdot w) + 2 b (w \cdot w) = w \cdot w$
                
                A = np.array([[np.dot(u, u), np.dot(w, u)], [np.dot(u, w), np.dot(w, w)]])
                B = np.array([np.dot(u, u), np.dot(w, w)])
                
                try:
                    coeffs = np.linalg.solve(A, B)
                except np.linalg.LinAlgError:
                    continue
                a, b = coeffs
                c_p = v1 + a * u + b * w
                
                if np.linalg.norm(c_p) < 1e-10:
                    continue
                    
                c_s = c_p / np.linalg.norm(c_p)
                
                # Calculate angular circumradius
                # cos(R) = c_s . v1
                cos_R = np.dot(c_s, v1)
                # Clip for numerical stability
                cos_R = np.clip(cos_R, -1.0, 1.0)
                R = np.arccos(cos_R)
                
                if R > max_radius:
                    max_radius = R
                    max_face_idx = idx
                    max_circumcenter = c_s
            
            if max_face_idx == -1:
                break
                
            # Move the vertices of the worst face towards the circumcenter
            face = faces[max_face_idx]
            for i in face:
                # Move towards circumcenter
                # v_new = v + alpha * (c_s - v)
                # This moves the point along the great circle towards c_s
                v = points[i]
                points[i] = v + step_size * (max_circumcenter - v)
                
            # Normalize
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis]
            
            # Decay step size
            step_size *= decay
            
        except Exception:
            break
            
    return points
