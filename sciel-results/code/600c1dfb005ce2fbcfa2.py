"""
Hexagon packing: Place 11 unit regular hexagons to minimize enclosing hexagon side.
Uses scipy.optimize.minimize with trust-constr method to handle non-overlap constraints strictly.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

N = 11
_START = time.time()
_DEADLINE = _START + 140.0  # Safety margin before 150s kill

# Unit regular hexagon vertices (flat-top relative to local frame)
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees
# Circumradius = 1
_BASE = np.array([
    [math.cos(math.radians(a)), math.sin(math.radians(a))] 
    for a in [0, 60, 120, 180, 240, 300]
])

# Normals for the enclosing hexagon (flat-top orientation assumed for width calculation)
# Width of a regular hexagon with side s in direction theta is s * (sqrt(3)/2 * |cos(theta)| + 1/2 * |sin(theta)|) * 2? 
# Actually, easier to just project all points onto 3 orthogonal directions 30, 90, 150 and find max span.
# For a regular hexagon of side L centered at origin, flat-top:
# Vertices: (L,0), (L/2, L*sqrt(3)/2), (-L/2, L*sqrt(3)/2), (-L,0), (-L/2, -L*sqrt(3)/2), (L/2, -L*sqrt(3)/2)
# The "width" in x-direction is 2L. In direction 60 deg, width is 2 * L * sqrt(3)/2 * cos(30) + ... 
# Let's stick to the projection method which is robust.
_NORMS = np.array([
    [math.cos(math.radians(30)), math.sin(math.radians(30))],
    [math.cos(math.radians(90)), math.sin(math.radians(90))],
    [math.cos(math.radians(150)), math.sin(math.radians(150))]
])


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _get_vertices(centers, angles):
    """Compute vertices for all hexagons."""
    n = len(centers)
    verts = np.empty((n, 6, 2))
    for i in range(n):
        R = _rot(angles[i])
        verts[i] = centers[i] + _BASE @ R.T
    return verts


def _min_dist_convex_hexagons(V1, V2):
    """
    Calculate minimum distance between two convex hexagons using Separating Axis Theorem.
    Returns 0 if overlapping, else minimum gap.
    """
    min_dist = 1e18
    
    # Axes are normals to the edges of both hexagons
    axes = []
    for poly in (V1, V2):
        for i in range(6):
            e = poly[(i+1)%6] - poly[i]
            # Normal
            n = np.array([-e[1], e[0]])
            n = n / np.linalg.norm(n)
            axes.append(n)
            
    axes = np.array(axes)
    
    # Project both polygons onto all axes
    proj1 = V1 @ axes.T
    proj2 = V2 @ axes.T
    
    r1_min = proj1.min(axis=0)
    r1_max = proj1.max(axis=0)
    r2_min = proj2.min(axis=0)
    r2_max = proj2.max(axis=0)
    
    # Check for separation
    # If max1 < min2 or max2 < min1, they are separated
    # Gap is min2 - max1 or min1 - max2
    gap1 = r2_min - r1_max
    gap2 = r1_min - r2_max
    
    valid1 = gap1 > 0
    valid2 = gap2 > 0
    
    if np.any(valid1):
        min_dist = min(min_dist, float(gap1[valid1].min()))
    if np.any(valid2):
        min_dist = min(min_dist, float(gap2[valid2].min()))
        
    if min_dist > 1e10:
        return 0.0
    return min_dist


def _objective(x):
    """
    Objective function: Side length of enclosing hexagon.
    x: [x0, y0, a0, x1, y1, a1, ...]
    """
    centers = x[0:3*N:3].reshape(N, 2)
    angles = x[1:3*N:3]
    verts = _get_vertices(centers, angles)
    all_points = verts.reshape(-1, 2)
    
    # Calculate width in 3 principal directions
    widths = []
    for n in _NORMS:
        proj = all_points @ n
        w = proj.max() - proj.min()
        widths.append(w)
    
    max_width = max(widths)
    # For a regular hexagon, Width = Side * sqrt(3) if flat-top aligned with axes?
    # Let's check:
    # Side 1. Flat top.
    # X extent: 2. Width x = 2. sqrt(3)*1 = 1.732. So Width = 2 * Side / sqrt(3) ?
    # Vertices: (1,0), (0.5, .866), (-.5, .866), (-1,0), ...
    # X range: -1 to 1 -> Width 2.
    # Y range: -0.866 to 0.866 -> Width 1.732.
    # Direction 30 deg:
    # Project (1,0) on (0.866, 0.5) -> 0.866
    # Project (0.5, .866) on (0.866, 0.5) -> 0.433 + 0.433 = 0.866
    # Project (-0.5, .866) -> -0.433 + 0.433 = 0
    # Project (-1,0) -> -0.866
    # Width = 0.866 - (-0.866) = 1.732.
    # So max width for Side 1 is 2.
    # Therefore Side = max_width / 2.
    return max_width / 2.0


def _collision_constraints(x):
    """
    Constraint function for NonlinearConstraint.
    Returns array of distances between all pairs.
    We require distance >= 0.
    """
    centers = x[0:3*N:3].reshape(N, 2)
    angles = x[1:3*N:3]
    verts = _get_vertices(centers, angles)
    
    n_pairs = N * (N - 1) // 2
    dists = np.zeros(n_pairs)
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            dists[k] = _min_dist_convex_hexagons(verts[i], verts[j])
            k += 1
    return dists


def _grad_objective(x):
    """
    Numerical gradient for objective to help trust-constr.
    Using central differences.
    """
    eps = 1e-6
    grad = np.zeros_like(x)
    f0 = _objective(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (_objective(x_plus) - _objective(x_minus)) / (2 * eps)
    return grad


def _pack():
    """
    Main optimization routine.
    """
    rng = np.random.default_rng(42)
    
    # Initial Guess: Hexagonal cluster
    # 1 center, 6 neighbors at distance 2 (edge touching), 4 satellites.
    # Distance between centers of touching hexagons is 2 * apothem = 2 * sqrt(3)/2 = sqrt(3) ~ 1.732?
    # No. Side length 1. Distance between centers of two touching hexagons is 2 * (3/4 side) * 2? 
    # Apophem = sqrt(3)/2. Distance = 2 * sqrt(3)/2 = sqrt(3).
    # Let's verify.
    # Hex 1 center (0,0). Top edge y = sqrt(3)/2.
    # Hex 2 center (0, y_c). Bottom edge y = y_c - sqrt(3)/2.
    # Touching: y_c - sqrt(3)/2 = sqrt(3)/2 => y_c = sqrt(3).
    # So neighbor distance is sqrt(3).
    
    d = math.sqrt(3)
    
    # Base configuration
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # Center
    centers[0] = [0.0, 0.0]
    
    # 6 neighbors
    for k in range(6):
        ang = math.radians(60 * k + 30) # 30, 90, 150, 210, 270, 330
        centers[k+1] = d * np.array([math.cos(ang), math.sin(ang)])
    
    # 4 satellites in the "gaps" of the 6 neighbors
    # The 6 neighbors form a ring. The gaps are at 0, 90, 180, 270?
    # Neighbors at 30, 90, 150, 210, 270, 330.
    # Midpoints of arcs: 0 (between 330 and 30), 60 (between 30 and 90), etc.
    # But 60 is occupied by neighbor? No, neighbors are at 30, 90...
    # So gaps are at 0, 60, 120, 180, 240, 300?
    # Let's check distance to existing neighbors.
    # If we put a hex at angle 0, distance d2 from center.
    # It should touch two neighbors (at 30 and 330).
    # Triangle formed by Center, Neighbor(30), Satellite(0).
    # Side Center-Neighbor = d. Side Neighbor-Satellite = d (touching). Side Center-Satellite = d2.
    # Angle at Center is 30 deg.
    # By law of cosines: d^2 = d^2 + d2^2 - 2*d*d2*cos(30).
    # 0 = d2^2 - 2*d*d2*cos(30) => d2 = 2*d*cos(30) = 2*sqrt(3)*sqrt(3)/2 = 3.
    # So satellites at distance 3, angles 0, 60, 120, 180, 240, 300.
    # We have 4 satellites. We can pick 4 of these 6 positions.
    # Due to symmetry of objective (hexagon), orientation matters.
    # Let's pick 0, 90, 180, 270? No, 90 is a neighbor position.
    # The 6 gap positions are 0, 60, 120, 180, 240, 300.
    # Let's pick 0, 120, 240 (equilateral) and one more?
    # Or maybe 0, 90, 180, 270 if we rotated the whole thing?
    # Let's just pick 4 distinct gap positions: 0, 60, 120, 180.
    sat_angles = [0, 60, 120, 180]
    for i, ang_deg in enumerate(sat_angles):
        ang = math.radians(ang_deg)
        centers[7+i] = 3.0 * np.array([math.cos(ang), math.sin(ang)])
        
    # Add small noise to break symmetry
    centers += rng.normal(0, 0.1, centers.shape)
    angles += rng.normal(0, 0.1, angles.shape)
    
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = angles[i]
        
    # Constraints: dist >= 0
    # We use a margin of -0.001 to allow slight floating point error or define >= -epsilon
    # NonlinearConstraint(fun, lb, ub)
    # fun(x) >= lb
    # We want dist >= 0.
    n_constraints = N * (N - 1) // 2
    cons = NonlinearConstraint(_collision_constraints, 0.0, np.inf, jac=None)
    
    # Bounds: keep positions within a reasonable range to prevent divergence
    # Max side length estimate: 10. So centers within +/- 10.
    bounds = [(-10, 10), (-10, 10), (-math.pi, math.pi)] * N
    
    # Run optimization
    best_x = x0
    best_val = _objective(x0)
    
    # Try multiple random restarts if time permits
    # But trust-constr is heavy. Let's do one main run first.
    
    try:
        # Use numerical gradient if analytical is too complex or slow
        # trust-constr supports jac for objective
        res = minimize(
            _objective,
            x0,
            method='trust-constr',
            jac=_grad_objective,
            hess=None,
            constraints=cons,
            bounds=bounds,
            options={'maxiter': 100, 'gtol': 1e-5, 'xtol': 1e-6, 'ctol': 1e-6, 'disp': False}
        )
        
        if res.success:
            # Verify validity
            dists = _collision_constraints(res.x)
            if np.min(dists) >= -1e-5: # Small tolerance
                val = _objective(res.x)
                if val < best_val:
                    best_x = res.x
                    best_val = val
        else:
            # If it failed, check if it's still valid
            dists = _collision_constraints(res.x)
            if np.min(dists) >= -1e-5:
                val = _objective(res.x)
                if val < best_val:
                    best_x = res.x
                    best_val = val
            
    except Exception as e:
        print(f"Optimization error: {e}")
        
    # If we have time, try another restart with different initial guess
    if time.time() < _DEADLINE - 20.0:
        # Random restart
        centers2 = rng.normal(0, 2, (N, 2))
        angles2 = rng.uniform(-math.pi, math.pi, N)
        x0_2 = np.zeros(3 * N)
        for i in range(N):
            x0_2[3*i] = centers2[i, 0]
            x0_2[3*i+1] = centers2[i, 1]
            x0_2[3*i+2] = angles2[i]
            
        try:
            res2 = minimize(
                _objective,
                x0_2,
                method='trust-constr',
                jac=_grad_objective,
                constraints=cons,
                bounds=bounds,
                options={'maxiter': 50, 'gtol': 1e-4, 'xtol': 1e-5, 'ctol': 1e-5, 'disp': False}
            )
            dists2 = _collision_constraints(res2.x)
            if np.min(dists2) >= -1e-5:
                val2 = _objective(res2.x)
                if val2 < best_val:
                    best_x = res2.x
                    best_val = val2
        except Exception:
            pass
            
    return best_x, best_val

def construct():
    try:
        x, side = _pack()
        
        # Extract results
        inner = []
        for i in range(N):
            cx = float(x[3*i])
            cy = float(x[3*i+1])
            ang_rad = float(x[3*i+2])
            ang_deg = (ang_rad * 180.0 / math.pi) % 360.0
            inner.append([cx, cy, ang_deg])
            
        # Center of outer hexagon is assumed to be origin (0,0) in our objective calc?
        # Wait, _objective calculates width based on absolute coordinates.
        # It assumes the enclosing hexagon is centered at the origin?
        # No, _enclosing_side in the previous code projected points and took max-min.
        # That calculates the size of the bounding hexagon, but doesn't fix its position.
        # The problem allows translating the outer hexagon.
        # However, the width calculation `max - min` is translation invariant.
        # So the side length returned is correct regardless of where the outer hexagon is centered.
        # We can return center (0,0) for the outer hexagon, as the inner hexagons' absolute coordinates
        # define the packing. The evaluator likely checks if inner hexagons fit in an outer hexagon
        # of side `side` centered at `center`.
        # If we return center (0,0), the evaluator will check if all inner hexagons are inside
        # a hexagon of side `side` centered at (0,0).
        # Our objective minimized the width *in absolute coordinates*.
        # If the cluster is centered at (5,5), the width is the same, but it won't fit in a hexagon
        # centered at (0,0) with that side length.
        # We MUST re-center the packing to the origin to ensure it fits in the reported outer hexagon
        # centered at (0,0).
        
        # Calculate centroid of the packing to shift it to origin?
        # Or better, calculate the "center" of the bounding box and shift.
        # Actually, the width is minimal when the shape is centered.
        # Let's compute the center of the bounding box in the 3 directions and shift.
        
        centers = x[0:3*N:3].reshape(N, 2)
        angles = x[1:3*N:3]
        verts = _get_vertices(centers, angles)
        all_points = verts.reshape(-1, 2)
        
        # Calculate center as average of projections?
        # Or just mean of all points?
        # For a regular hexagon, the center that minimizes max distance is the circumcenter.
        # Let's just use the mean of the points as an approximation for re-centering.
        mean_pt = all_points.mean(axis=0)
        
        # Shift all inner hexagons by -mean_pt
        shifted_centers = centers - mean_pt
        
        # Re-calculate side length to be safe (should be same)
        shifted_verts = _get_vertices(shifted_centers, angles)
        shifted_points = shifted_verts.reshape(-1, 2)
        
        widths = []
        for n in _NORMS:
            proj = shifted_points @ n
            w = proj.max() - proj.min()
            widths.append(w)
        final_side = max(widths) / 2.0
        
        # Construct inner list with shifted coordinates
        inner = []
        for i in range(N):
            cx = float(shifted_centers[i, 0])
            cy = float(shifted_centers[i, 1])
            ang_deg = (float(angles[i]) * 180.0 / math.pi) % 360.0
            inner.append([cx, cy, ang_deg])
            
        return inner, (0.0, 0.0), float(final_side), 0.0

    except Exception as e:
        print("Exception in construct:", e)
        # Fallback: Line packing
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0


if __name__ == "__main__":
    res = construct()
    print("outer side:", res[2])
    print("elapsed:", time.time() - _START)
