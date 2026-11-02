import numpy as np
from scipy.optimize import minimize

N = 12

def _hex_normals():
    """Return the 6 outward normals of a unit regular hexagon (flat-top orientation)."""
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    return np.column_stack((np.cos(angles), np.sin(angles)))

def construct():
    normals = _hex_normals()
    sqrt3 = np.sqrt(3.0)
    
    best_L = float('inf')
    best_inner = None
    best_params = None
    
    # Radial Slicing: 1 center, 5 inner ring, 6 outer ring
    # Parameters: [r_inner, r_outer, rot_inner, rot_outer]
    # Inner ring: 5 hexagons at angles 0, 72, 144, 216, 288 + rot_inner
    # Outer ring: 6 hexagons at angles 0, 60, 120, 180, 240, 300 + rot_outer
    
    def build_centers(params):
        r_in, r_out, rot_in, rot_out = params
        rot_in_rad = np.deg2rad(rot_in)
        rot_out_rad = np.deg2rad(rot_out)
        centers = np.zeros((N, 2))
        # Center hexagon
        centers[0] = [0.0, 0.0]
        
        # Inner ring: 5 hexagons
        inner_angles = np.deg2rad([0, 72, 144, 216, 288]) + rot_in_rad
        for k in range(5):
            centers[1 + k] = [r_in * np.cos(inner_angles[k]), r_in * np.sin(inner_angles[k])]
            
        # Outer ring: 6 hexagons
        outer_angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + rot_out_rad
        for k in range(6):
            centers[6 + k] = [r_out * np.cos(outer_angles[k]), r_out * np.sin(outer_angles[k])]
            
        return centers
    
    def compute_violations(centers):
        """Compute maximum violation across all pairs.
        For two unit hexagons with same orientation, they are disjoint iff
        max_k |normal_k . (c_i - c_j)| >= 2*sqrt(3)/2 = sqrt(3)
        Wait, let me reconsider. The inradius of a unit hexagon is sqrt(3)/2.
        Two hexagons are disjoint if the distance between centers projected onto any normal
        is at least 2 * inradius = sqrt(3).
        So violation = sqrt(3) - max_k |normal_k . diff|
        """
        max_viol = 0.0
        for i in range(N):
            for j in range(i+1, N):
                diff = centers[i] - centers[j]
                d_dots = np.abs(diff @ normals.T)
                max_d_dot = np.max(d_dots)
                viol = sqrt3 - max_d_dot
                if viol > max_viol:
                    max_viol = viol
        return max_viol
    
    def objective(params):
        centers = build_centers(params)
        # L is determined by the maximum extent of any center from origin
        # plus the inradius of the inner hexagon
        # For a hexagon centered at c with side length 1, the support function in direction u is
        # h(u) = |u . c| + sqrt(3)/2 (for flat-top, this is approximate)
        # Actually, the maximum distance from center to any point is sqrt(3)/2 (inradius)
        # No wait, the circumradius is 1, inradius is sqrt(3)/2.
        # The outer hexagon must contain all inner hexagons.
        # The outer hexagon has side length L, inradius L*sqrt(3)/2, circumradius L.
        # For the outer hexagon to contain an inner hexagon at center c,
        # we need: for all normals n of outer hex, n . c + sqrt(3)/2 <= L * sqrt(3)/2
        # => n . c <= L * sqrt(3)/2 - sqrt(3)/2 = (L-1) * sqrt(3)/2
        # => |n . c| <= (L-1) * sqrt(3)/2 for all n
        # => max_n |n . c| <= (L-1) * sqrt(3)/2
        # => L >= 1 + max_n |n . c| * 2/sqrt(3)
        
        max_abs_dot = np.max(np.abs(centers @ normals.T))
        L_req = 1.0 + 2.0 * max_abs_dot / sqrt3
        
        # Penalty for violations
        max_viol = compute_violations(centers)
        penalty = 1000.0 * max(max_viol, 0.0)
        
        return L_req + penalty
    
    # Multi-start optimization
    n_starts = 300
    for seed in range(n_starts):
        np.random.seed(seed)
        # Initial guesses based on geometric intuition
        # Inner ring: 5 hexagons around center. Minimum r_in to avoid overlap with center:
        # Center at origin, inner at r_in. Violation = sqrt(3) - max_n |n . (0 - c)| = sqrt(3) - r_in * max_n |n . unit_vector|
        # For r_in to be valid, we need r_in >= sqrt(3) (when aligned with a normal)
        # But with rotation, the worst case is when the vector aligns with a normal.
        # So r_in >= sqrt(3) ≈ 1.732
        
        # Outer ring: 6 hexagons at r_out. They must not overlap with inner ring or each other.
        # Between adjacent outer hexagons: distance = 2 * r_out * sin(30°) = r_out
        # Need r_out >= sqrt(3)
        # Between outer and inner: more complex, depends on angles.
        
        r_in = np.random.uniform(1.7, 2.5)
        r_out = np.random.uniform(3.0, 4.5)
        rot_in = np.random.uniform(0, 72)
        rot_out = np.random.uniform(0, 60)
        
        x0 = np.array([r_in, r_out, rot_in, rot_out])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            params = result.x
            
            # Verify validity
            centers = build_centers(params)
            max_viol = compute_violations(centers)
            
            if max_viol > 1e-4:
                continue
            
            max_abs_dot = np.max(np.abs(centers @ normals.T))
            L_final = 1.0 + 2.0 * max_abs_dot / sqrt3
            
            if L_final < best_L:
                best_L = L_final
                best_params = params.copy()
                best_inner = [[float(centers[i, 0]), float(centers[i, 1]), 0.0] for i in range(N)]
                
        except Exception:
            continue
            
    if best_inner is None:
        # Fallback: spread out
        inner = [[5.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 50.0, 0.0
        
    return best_inner, (0.0, 0.0), float(best_L), 0.0
