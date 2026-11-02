import numpy as np
from scipy.optimize import minimize

N = 12

def _hex_normals():
    """Return the 6 outward normals of a unit regular hexagon (pointy-top)."""
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    return np.column_stack((np.cos(angles), np.sin(angles)))

def construct():
    normals = _hex_normals()
    sqrt3 = np.sqrt(3.0)
    
    # The 6 normals correspond to directions 30, 90, 150, 210, 270, 330 degrees.
    # A regular hexagon with side length L centered at origin has inradius L*sqrt(3)/2.
    # The support function h(u) for the outer hexagon is L*sqrt(3)/2.
    # For an inner unit hexagon centered at c, its support in direction u is u·c + sqrt(3)/2.
    # Containment condition: u·c + sqrt(3)/2 <= L*sqrt(3)/2 for all u.
    # This implies L >= (u·c + sqrt(3)/2) / (sqrt(3)/2) = 2*u·c/sqrt(3) + 1.
    # Since we must satisfy this for all 6 normals (both positive and negative directions),
    # we need L >= max_u (2*|u·c|/sqrt(3) + 1) for all centers c.
    # So L = 1 + (2/sqrt(3)) * max_{c, u} |u·c|.

    def build_centers(params):
        r_in, r_out, rot_deg = params
        rot_rad = np.deg2rad(rot_deg)
        centers = np.zeros((N, 2))
        # Center hexagon
        centers[0] = [0.0, 0.0]
        # Inner ring: 5 hexagons
        # To fit 5 in a ring, the angles are 72 degrees apart.
        # We align them such that one is at 0 + rot, etc.
        inner_angles = np.deg2rad([0, 72, 144, 216, 288]) + rot_rad
        for k in range(5):
            centers[1 + k] = [r_in * np.cos(inner_angles[k]), r_in * np.sin(inner_angles[k])]
        # Outer ring: 6 hexagons
        # 6 hexagons at 60 degrees apart.
        outer_angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + rot_rad
        for k in range(6):
            centers[6 + k] = [r_out * np.cos(outer_angles[k]), r_out * np.sin(outer_angles[k])]
        return centers

    def check_separation(centers):
        """
        Check if centers are valid (disjoint hexagons).
        Returns 0.0 if valid, else the amount of violation (positive).
        """
        min_viol = 0.0
        n = len(centers)
        for i in range(n):
            for j in range(i+1, n):
                diff = centers[i] - centers[j]
                d_dots = np.abs(diff @ normals.T)
                max_d_dot = np.max(d_dots)
                if max_d_dot < sqrt3:
                    viol = sqrt3 - max_d_dot
                    min_viol = max(min_viol, viol)
        return min_viol

    def objective(params):
        centers = build_centers(params)
        
        # Calculate required L
        dots = centers @ normals.T
        max_abs_dot = np.max(np.abs(dots))
        L_req = 1.0 + 2.0 * max_abs_dot / sqrt3
        
        # Check separation
        viol = check_separation(centers)
        
        # Penalty for separation violation
        # We want to minimize L, but if viol > 0, the configuration is invalid.
        # We use a large penalty to push the optimizer away from invalid regions.
        return L_req + 1000.0 * viol * viol

    # Initial guesses based on geometric reasoning
    # For 5 hexagons around 1, the inner ring radius r_in should be at least 2*sqrt(3) ~ 3.46?
    # No, distance between centers of adjacent hexagons in a ring of 5:
    # Chord length = 2 * r_in * sin(36 deg).
    # Min distance between centers for disjoint hexagons is 2*sqrt(3) ~ 3.46.
    # So 2 * r_in * sin(36) >= 2*sqrt(3) => r_in >= sqrt(3)/sin(36) ~ 3.46/0.588 ~ 5.88.
    # Wait, the "distance" condition is max_u |u·diff| >= sqrt(3).
    # If diff is along a normal direction, distance is sqrt(3).
    # If diff is perpendicular to a normal (along a flat side normal), the required distance is larger?
    # The Minkowski sum is a hexagon of inradius sqrt(3). The minimum distance from origin to boundary is sqrt(3).
    # So the centers must be at Euclidean distance >= sqrt(3) ONLY IF the vector is along a normal.
    # The maximum Euclidean distance allowed inside the "forbidden" hexagon is the circumradius 2*sqrt(3).
    # So Euclidean distance must be >= sqrt(3) is not sufficient.
    # The condition is that the vector c_i - c_j is outside the hexagon of inradius sqrt(3).
    # The closest point on the boundary to the origin is sqrt(3).
    # So if the vector is aligned with a normal, dist >= sqrt(3) is enough.
    # If aligned with a vertex, dist >= 2*sqrt(3) is required? No.
    # The hexagon of Minkowski sum has vertices at distance 2*sqrt(3) and edge midpoints at sqrt(3).
    # So the condition is: The vector d must not be inside this hexagon.
    
    # Let's estimate r_in.
    # If we place 5 hexagons symmetrically, the angle between adjacent centers is 72 deg.
    # Let's try to make the vector between adjacent centers align with a normal direction to minimize distance.
    # Normals are at 30, 90, 150...
    # If we rotate the ring, we can align the chord with a normal.
    # Chord vector angle: theta_k + 36 (bisector).
    # We want theta_k + 36 to be 30 + 180*m or 90 + 180*m etc.
    # Let's just use the optimizer to find the best rotation.
    
    # Rough estimates:
    # r_in needs to be large enough. Let's start around 4.0 - 6.0.
    # r_out needs to be larger.
    
    best_L = float('inf')
    best_params = None
    best_inner = None

    # Generate a few good starting points
    starts = []
    # Symmetric start
    starts.append([5.0, 8.0, 0.0])
    starts.append([5.5, 8.5, 30.0])
    starts.append([6.0, 9.0, 15.0])
    
    # Add some random perturbations to the symmetric start to escape local minima
    np.random.seed(42)
    for _ in range(10):
        r_in = np.random.uniform(4.5, 6.5)
        r_out = np.random.uniform(7.5, 10.0)
        rot = np.random.uniform(0, 60)
        starts.append([r_in, r_out, rot])

    for x0 in starts:
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            params = result.x
            centers = build_centers(params)
            
            # Verify validity strictly
            viol = check_separation(centers)
            if viol > 1e-6:
                continue
            
            # Compute final L
            dots = centers @ normals.T
            max_abs_dot = np.max(np.abs(dots))
            L_final = 1.0 + 2.0 * max_abs_dot / sqrt3
            
            if L_final < best_L:
                best_L = L_final
                best_params = params.copy()
                best_inner = [[centers[i, 0], centers[i, 1], 0.0] for i in range(N)]
        except Exception:
            continue

    if best_inner is None:
        # Fallback if optimization fails
        # Use a simple grid search for robustness
        best_L = float('inf')
        for r_in in np.linspace(4.0, 7.0, 10):
            for r_out in np.linspace(7.0, 12.0, 10):
                for rot in np.linspace(0, 60, 10):
                    centers = build_centers(np.array([r_in, r_out, rot]))
                    viol = check_separation(centers)
                    if viol > 1e-5:
                        continue
                    dots = centers @ normals.T
                    max_abs_dot = np.max(np.abs(dots))
                    L_final = 1.0 + 2.0 * max_abs_dot / sqrt3
                    if L_final < best_L:
                        best_L = L_final
                        best_inner = [[centers[i, 0], centers[i, 1], 0.0] for i in range(N)]
        
        if best_inner is None:
            # Last resort: place them far apart
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, (0.0, 0.0), best_L, 0.0
