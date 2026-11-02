import numpy as np
from math import pi, cos, sin, sqrt
import time

N = 12
T0 = time.time()
DEADLINE = 120.0

# Convert hex-grid axial coords to Cartesian (flat-top, side=1)
def axial_to_xy(q, r):
    x = 1.5 * q
    y = sqrt(3) / 2 * (q + 2 * r)
    return x, y

# Define a compact 12-hex cluster: 3-wide by 4-tall parallelogram in axial coords
# q in {0,1,2}, r in {0,1,2,3}
# This gives 12 hexagons in a parallelogram shape
def get_cluster_centers():
    centers = []
    for q in range(3):
        for r in range(4):
            x, y = axial_to_xy(q, r)
            centers.append((x, y))
    return centers

# Compute the 6 support directions of a regular hexagon with flat-top orientation
# A regular hexagon with flat-top has vertices at angles 0, 60, 120, 180, 240, 300 degrees
# The support function in direction u is max over vertices of dot(vertex, u)
# For a hexagon centered at origin with side length s, the support in direction theta
# can be computed from the convex hull.

# Actually, let's use a more direct approach:
# The width of a set S in direction u (unit vector) is max_{p in S} dot(p,u) - min_{p in S} dot(p,u)
# For the Minkowski sum S ⊕ H (where H is a unit hexagon), the support in direction u is
# h_S(u) + h_H(u)
# The width in direction u is 2 * max_k h_cluster(u_k) + 2 * h_hex(u_k) ... no.

# Let me think more carefully.
# A regular hexagon of side s has 6 vertices. The support function h(u) for a convex set is max_{x in set} x·u.
# For the Minkowski sum S ⊕ H, h_{S⊕H}(u) = h_S(u) + h_H(u).
# The width of S⊕H in direction u is h_{S⊕H}(u) - h_{S⊕H}(-u) = [h_S(u) + h_H(u)] - [h_S(-u) + h_H(-u)]
# = [h_S(u) - h_S(-u)] + [h_H(u) - h_H(-u)] = width_S(u) + width_H(u).

# For a regular hexagon of side length s, the width in any direction is 2*s (the distance between opposite sides).
# Wait, that's not right. The width of a regular hexagon of side s:
# - In the direction perpendicular to a pair of parallel sides (i.e., through midpoints of opposite sides): 2 * (sqrt(3)/2 * s) = sqrt(3)*s
# - In the direction through opposite vertices: 2s
# So the width varies with direction.

# The constraint is that the Minkowski sum must fit inside the outer hexagon of side S.
# The outer hexagon of side S has support h_outer(u) = (S/2) * f(u) where f(u) depends on direction.
# Actually, for a regular hexagon of side S centered at origin with flat-top:
# The vertices are at angles 0, 60, 120, 180, 240, 300 degrees, at distance S from center.
# h_outer(u) = max over the 6 vertices of dot(vertex, u)
# = S * max(cos(theta_v - theta_u)) for the 6 vertex angles.

# For the Minkowski sum to be contained in the outer hexagon, we need:
# h_{S⊕H}(u) <= h_outer(u) for ALL directions u.

# The critical directions are where the support functions have "corners".
# For a regular hexagon, the support function is piecewise linear, with breakpoints at the 6 vertex directions.
# So we need to check the constraint at 6 directions (the vertex directions of the outer hexagon)
# and also at the directions of the inner hexagon's support.

# Actually, the support of the Minkowski sum is the sum of supports. The constraint
# h_{S⊕H}(u) <= h_outer(u) for all u is equivalent to checking at all "critical" directions
# where either h_{S⊕H} or h_outer has a corner.

# For a regular hexagon of side S (flat-top, centered at origin), the vertices are at:
# V_k = (S * cos(k*pi/3), S * sin(k*pi/3)) for k=0..5
# The support function h_outer(u) for u at angle theta is:
# h_outer(u) = S * max_k cos(theta - k*pi/3)

# For the inner cluster (12 unit hexagons), the Minkowski sum with a unit hexagon gives a convex set.
# Its support in direction u is h_cluster(u) + h_unit_hex(u).

# h_unit_hex(u) for a unit regular hexagon (side=1, flat-top):
# The vertices are at distance 1 from center.
# h_unit_hex(u) = max_k cos(theta - k*pi/3)  [since side=1, the circumradius is 1]
# Wait, for a regular hexagon of side s, the circumradius is s. So for side=1, circumradius=1.
# h_unit_hex(u) = max_{k=0..5} cos(theta - k*pi/3)

# For the cluster, h_cluster(u) = max over all 12 hexagon centers p_i of (p_i · u) + h_unit_hex(u)
# Wait, each inner hexagon is a unit hexagon centered at p_i. The Minkowski sum of the cluster
# with a unit hexagon is the union of (p_i + unit_hex) for all i. But the convex hull of this
# union is the Minkowski sum of the convex hull of {p_i} with the unit hexagon.
# So h_cluster_minkowski(u) = h_convex_hull_of_centers(u) + h_unit_hex(u)

# So the constraint is:
# h_convex_hull_of_centers(u) + h_unit_hex(u) <= h_outer_hex(S, u)  for all u

# Where h_outer_hex(S, u) = S * max_k cos(theta - k*pi/3) for the outer hexagon orientation.

# The outer hexagon has a fixed orientation (let's say flat-top, same as inner hexagons initially).
# But we can rotate the entire cluster relative to the outer hexagon.

# Let's parameterize: the outer hexagon is fixed (flat-top, side S, centered at origin).
# The cluster has an orientation angle alpha. The centers of the inner hexagons are rotated by alpha.
# Also, each inner hexagon can have its own rotation, but since they are all unit regular hexagons,
# their Minkowski sum contribution is always a unit hexagon regardless of individual rotations
# (because the convex hull of rotated hexagons is still a hexagon of the same size... actually no,
# the Minkowski sum of the cluster with a unit hexagon assumes all inner hexagons are aligned.
# If they can rotate independently, the problem is more complex.)

# For simplicity and a strong baseline, let's assume all inner hexagons are aligned with the outer hexagon.
# Then the problem reduces to: find orientation alpha of the cluster and side length S such that
# the Minkowski sum fits in the outer hexagon.

# Actually, let me re-read the task. "each inner hexagon may translate and rotate."
# So each can rotate independently. But for a first pass, let's assume they all share the same orientation
# as the outer hexagon (alpha=0 relative). We can optimize the cluster's overall rotation.

# Let's implement the direct computation:

def support_hexagon(theta, circumradius=1.0):
    """Support function of a regular hexagon with given circumradius, in direction theta."""
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    max_val = 0
    for k in range(6):
        angle = k * pi / 3
        val = circumradius * cos(theta - angle)
        if val > max_val:
            max_val = val
    return max_val

def h_convex_hull(centers, theta):
    """Support function of the convex hull of centers in direction theta."""
    max_val = -float('inf')
    for (x, y) in centers:
        val = x * cos(theta) + y * sin(theta)
        if val > max_val:
            max_val = val
    return max_val

def min_side_length(centers, alpha, outer_theta_offset=0.0):
    """Compute the minimum side length S of the outer hexagon (flat-top) to contain
    the Minkowski sum of the cluster (rotated by alpha) with unit hexagons."""
    # Rotate centers by alpha
    rotated_centers = []
    for (x, y) in centers:
        xr = x * cos(alpha) - y * sin(alpha)
        yr = x * sin(alpha) + y * cos(alpha)
        rotated_centers.append((xr, yr))
    
    # The outer hexagon is flat-top, centered at origin, side S.
    # Its support in direction theta is S * support_hexagon(theta, 1.0)
    # (since circumradius = side length for a regular hexagon)
    
    # We need: h_convex_hull(rotated_centers, theta) + support_hexagon(theta, 1.0) <= S * support_hexagon(theta, 1.0)
    # for all theta.
    # So: S >= [h_convex_hull(rotated_centers, theta) + support_hexagon(theta, 1.0)] / support_hexagon(theta, 1.0)
    # = h_convex_hull(rotated_centers, theta) / support_hexagon(theta, 1.0) + 1.0
    
    # We need the max over all theta of the RHS.
    # The critical directions are where the support functions have corners.
    # Let's sample many directions.
    
    max_s = 0.0
    n_samples = 720
    for i in range(n_samples):
        theta = i * pi / n_samples  # 0 to pi (symmetric)
        h_outer_unit = support_hexagon(theta, 1.0)
        if h_outer_unit < 1e-10:
            continue
        h_cluster = h_convex_hull(rotated_centers, theta)
        h_inner_hex = support_hexagon(theta, 1.0)
        s_needed = h_cluster / h_outer_unit + 1.0
        if s_needed > max_s:
            max_s = s_needed
    return max_s

def optimize():
    centers = get_cluster_centers()
    
    # Grid search over alpha in [0, pi/3] (hexagon symmetry period)
    n_alpha = 360
    best_s = float('inf')
    best_alpha = 0.0
    
    for i in range(n_alpha):
        alpha = i * (pi / 3) / n_alpha
        s = min_side_length(centers, alpha)
        if s < best_s:
            best_s = s
            best_alpha = alpha
    
    # Local refinement around best_alpha
    for _ in range(3):
        alpha_lo = max(0.0, best_alpha - (pi / 3) / n_alpha)
        alpha_hi = min(pi / 3, best_alpha + (pi / 3) / n_alpha)
        n_fine = 100
        for i in range(n_fine + 1):
            alpha = alpha_lo + i * (alpha_hi - alpha_lo) / n_fine
            s = min_side_length(centers, alpha)
            if s < best_s:
                best_s = s
                best_alpha = alpha
        alpha_lo = best_alpha - (pi / 3) / (n_alpha * 4)
        alpha_hi = best_alpha + (pi / 3) / (n_alpha * 4)
    
    # Compute the optimal center of the outer hexagon
    # The Minkowski sum should be centered in the outer hexagon.
    # The centroid of the convex hull of the rotated centers should map to the center.
    # For now, let's center the cluster at the origin.
    
    # Translate the cluster so its centroid is at the origin
    centroid_x = sum(x for (x, y) in centers) / len(centers)
    centroid_y = sum(y for (x, y) in centers) / len(centers)
    
    # Rotate and translate
    inner = []
    for (x, y) in centers:
        x_rot = (x - centroid_x) * cos(best_alpha) - (y - centroid_y) * sin(best_alpha)
        y_rot = (x - centroid_x) * sin(best_alpha) + (y - centroid_y) * cos(best_alpha)
        inner.append([x_rot, y_rot, 0.0])
    
    return inner, (0.0, 0.0), best_s, 0.0

def construct():
    elapsed = time.time() - T0
    if elapsed > DEADLINE - 10.0:
        # Fallback: return a valid but not optimal solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    try:
        result = optimize()
        return result
    except Exception:
        # Fallback on error
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
