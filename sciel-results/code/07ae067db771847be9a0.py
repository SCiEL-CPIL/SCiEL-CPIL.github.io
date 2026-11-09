"""
Hexagon packing: Place 11 unit regular hexagons to minimize enclosing hexagon side.
Uses Simulated Annealing with a penalty-based objective function to handle overlaps.
"""
import math
import time
import numpy as np

N = 11
_START = time.time()
_DEADLINE = _START + 145.0  # Safety margin before 150s kill

# Unit regular hexagon vertices (flat-top relative to local frame)
_BASE = np.array([
    [math.cos(math.radians(a)), math.sin(math.radians(a))] 
    for a in [0, 60, 120, 180, 240, 300]
])

# Normals for the enclosing hexagon (flat-top orientation assumed for width calculation)
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


def _min_separation_convex_hexagons(V1, V2):
    """
    Calculate minimum separation between two convex hexagons using Separating Axis Theorem.
    Returns negative value if overlapping (depth of penetration), positive if separated.
    """
    min_separation = -1e18
    
    # Axes are normals to the edges of both hexagons
    axes = []
    for poly in (V1, V2):
        for i in range(6):
            e = poly[(i+1)%6] - poly[i]
            # Normal
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                n = n / norm
                axes.append(n)
                
    if not axes:
        return 0.0
        
    axes = np.array(axes)
    
    # Project both polygons onto all axes
    proj1 = V1 @ axes.T
    proj2 = V2 @ axes.T
    
    r1_min = proj1.min(axis=0)
    r1_max = proj1.max(axis=0)
    r2_min = proj2.min(axis=0)
    r2_max = proj2.max(axis=0)
    
    # Separation on each axis
    # If r1_max < r2_min, separation is r2_min - r1_max (positive)
    # If r2_max < r1_min, separation is r1_min - r2_max (positive)
    # Otherwise, they overlap on this axis. The "overlap" amount is not simply defined for SAT depth,
    # but we can approximate the penetration depth by the minimum of the overlaps?
    # Actually, for penalty methods, we want a smooth measure.
    # A common approach: 
    # sep = min( r2_min - r1_max, r1_min - r2_max )
    # If sep > 0, they are separated by sep.
    # If sep < 0, they are overlapping. The value -sep is an upper bound on penetration depth?
    # Not exactly, but it's a standard proxy.
    
    sep1 = r2_min - r1_max
    sep2 = r1_min - r2_max
    
    # The minimum separation is the minimum of the positive separations if any exist.
    # If no positive separation exists, they overlap.
    # We return the minimum of (sep1, sep2) if we consider the "worst case" overlap?
    # No, the minimum separation between two convex sets is the maximum of the minimum separations over all axes?
    # Wait. The distance between two convex polygons is the maximum of the separations over all separating axes.
    # If there is no separating axis, distance is 0 (or negative for penetration).
    # To get a penalty for overlap, we can use the minimum of the "gaps" where gap = min(r2_min - r1_max, r1_min - r2_max).
    # If any gap is positive, the distance is at least that gap.
    # If all gaps are negative, the objects overlap.
    # A simple penalty is sum of max(0, -gap)^2 for all axes? No, that's too strong.
    # Let's use the standard distance: 
    # dist = max(0, min_axis_sep) where min_axis_sep is the minimum separation found?
    # No, if they are separated, dist = max(sep1, sep2) over all axes?
    # Actually, if they are separated, there is at least one axis where they are separated.
    # The true distance is the maximum of the separations over all axes that separate them?
    # No, the distance is the minimum distance between any point in P1 and any point in P2.
    # SAT gives: if separated, dist >= max separation on separating axes? No.
    # If separated on axis A by d_A, then dist >= d_A? No, dist is the Euclidean distance.
    # The projection distance is a lower bound on Euclidean distance.
    # So, if we find an axis where they are separated by d, the Euclidean distance is at least d.
    # The true Euclidean distance is the maximum of these lower bounds? No, it's the maximum of the projections?
    # Actually, for convex polygons, the Euclidean distance is the maximum of the separations over all axes?
    # Yes, if the polygons are separated.
    # If they overlap, the separation is negative on all axes.
    
    # For penalty purposes, we want a value that is 0 if separated, and positive if overlapping.
    # Let's define overlap_penalty = max(0, -min_separation_on_any_axis)?
    # If separated, min_separation > 0, so penalty = 0.
    # If overlapping, min_separation < 0, so penalty = -min_separation.
    # This is a reasonable proxy for penetration depth.
    
    all_seps = np.minimum(sep1, sep2)
    min_sep = np.max(all_seps) # Wait, if separated, we want the largest gap? No, we want to know if they are separated.
    # If ANY axis shows separation, they are separated.
    # The "distance" is the maximum of the separations over all axes?
    # Let's check: If they are far apart, many axes show large separation. The max separation is a good estimate of distance?
    # Actually, the distance is the minimum over all pairs of points.
    # SAT: If max(r1_min - r2_max, r2_min - r1_max) > 0 for some axis, they are separated.
    # The value of that separation is a lower bound on the Euclidean distance.
    # The maximum of these lower bounds is the best lower bound provided by SAT.
    # So, dist_lower_bound = max( max(r2_min - r1_max), max(r1_min - r2_max) )
    
    max_sep = max(np.max(sep1), np.max(sep2))
    
    if max_sep > 0:
        return max_sep
    else:
        # Overlapping. Return a negative value indicating overlap.
        # The "depth" is roughly -max_sep? No, max_sep is the maximum of the minimums?
        # If they overlap, all seps are negative.
        # Let's return -(-max_sep) = max_sep? No.
        # Let's return the most negative separation? 
        # A common penalty is the sum of squares of the negative separations.
        # But we want a single scalar per pair.
        # Let's return -(-max_sep) is confusing.
        # If max_sep is -0.5, it means the "best" separation was -0.5.
        # So they are overlapping by at least 0.5?
        # Let's return max_sep. It will be negative.
        return max_sep


def _objective(x):
    """
    Objective function: Side length of enclosing hexagon + Penalty for overlaps.
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
    side_length = max_width / 2.0
    
    # Calculate overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            sep = _min_separation_convex_hexagons(verts[i], verts[j])
            if sep < 0:
                # Penalty is quadratic in the overlap depth
                penalty += (sep * sep)
                
    # Weight the penalty heavily to ensure feasibility
    return side_length + 10.0 * penalty


def _pack():
    """
    Simulated Annealing optimization routine.
    """
    rng = np.random.default_rng(42)
    
    # Initial Guess: Hexagonal cluster
    d = math.sqrt(3)
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # Center
    centers[0] = [0.0, 0.0]
    
    # 6 neighbors
    for k in range(6):
        ang = math.radians(60 * k + 30)
        centers[k+1] = d * np.array([math.cos(ang), math.sin(ang)])
    
    # 4 satellites
    sat_angles = [0, 60, 120, 180]
    for i, ang_deg in enumerate(sat_angles):
        ang = math.radians(ang_deg)
        centers[7+i] = 3.0 * np.array([math.cos(ang), math.sin(ang)])
        
    # Add small noise
    centers += rng.normal(0, 0.1, centers.shape)
    angles += rng.normal(0, 0.1, angles.shape)
    
    x = np.zeros(3 * N)
    for i in range(N):
        x[3*i] = centers[i, 0]
        x[3*i+1] = centers[i, 1]
        x[3*i+2] = angles[i]
        
    best_x = x.copy()
    best_val = _objective(x)
    current_val = best_val
    
    # SA Parameters
    T = 1.0
    cooling_rate = 0.995
    min_T = 1e-4
    max_iters = 50000
    
    iter_count = 0
    while time.time() < _DEADLINE and T > min_T:
        # Generate neighbor
        x_new = x.copy()
        
        # Perturb 1 or 2 random hexagons
        num_perturb = rng.integers(1, 3)
        for _ in range(num_perturb):
            idx = rng.integers(0, N)
            # Perturb x, y, angle
            x_new[3*idx] += rng.normal(0, 0.5)
            x_new[3*idx+1] += rng.normal(0, 0.5)
            x_new[3*idx+2] += rng.normal(0, 0.5)
            
        new_val = _objective(x_new)
        
        # Acceptance criterion
        if new_val < current_val:
            x = x_new
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_x = x_new.copy()
        else:
            delta = new_val - current_val
            if delta < 0:
                x = x_new
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_x = x_new.copy()
            else:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    x = x_new
                    current_val = new_val
        
        T *= cooling_rate
        iter_count += 1
        
    return best_x, best_val


def construct():
    try:
        x, side = _pack()
        
        # Extract results
        centers = x[0:3*N:3].reshape(N, 2)
        angles = x[1:3*N:3]
        
        # Re-center to origin
        verts = _get_vertices(centers, angles)
        all_points = verts.reshape(-1, 2)
        mean_pt = all_points.mean(axis=0)
        
        shifted_centers = centers - mean_pt
        
        # Re-calculate side length
        shifted_verts = _get_vertices(shifted_centers, angles)
        shifted_points = shifted_verts.reshape(-1, 2)
        
        widths = []
        for n in _NORMS:
            proj = shifted_points @ n
            w = proj.max() - proj.min()
            widths.append(w)
        final_side = max(widths) / 2.0
        
        # Construct inner list
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
