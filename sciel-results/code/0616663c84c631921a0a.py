import math
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N=11 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Returns (inner, center, side, angle) where:
      inner: list of [x, y, angle_degrees] for each inner hexagon
      center: (cx, cy) of outer hexagon
      side: outer hexagon side length
      angle: outer hexagon rotation in degrees
    """
    start_time = time.time()
    DEADLINE = start_time + 140.0  # 10s safety margin before 150s kill

    def hex_vertices(cx, cy, theta, scale=1.0):
        """Return 6 vertices of a regular hexagon centered at (cx,cy), rotated by theta radians, side=scale."""
        # Flat-top orientation at theta=0: vertices at angles 0, 60, 120, 180, 240, 300 degrees
        # Pointy-top would be offset by 30 deg. We use flat-top for consistency.
        verts = []
        for k in range(6):
            a = theta + k * math.pi / 3.0
            x = cx + scale * math.cos(a)
            y = cy + scale * math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def sat_min_depth(A, B):
        """
        Compute minimum translation depth to separate convex polygons A and B (both n-gons).
        Returns (depth, direction) where depth >= 0 (0 if separated), direction is unit vector
        to push A away from B.
        Uses SAT: check all edge normals of A and B.
        """
        n_a, n_b = len(A), len(B)
        min_depth = float('inf')
        min_dir = np.array([0.0, 0.0])

        def project(poly, axis):
            dots = poly @ axis
            return dots.min(), dots.max()

        def edge_normals(poly):
            normals = []
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                # Normal: perpendicular to edge
                # Rotate edge by 90 deg CCW
                nx = -edge[1]
                ny = edge[0]
                norm = math.hypot(nx, ny)
                if norm < 1e-12:
                    continue
                normals.append(np.array([nx / norm, ny / norm]))
            return normals

        all_normals = edge_normals(A) + edge_normals(B)
        for axis in all_normals:
            a_min, a_max = project(A, axis)
            b_min, b_max = project(B, axis)
            # Check for gap
            if a_max < b_min:
                # Gap along axis
                depth = b_min - a_max
                if depth > min_depth:
                    min_depth = depth
                    min_dir = -axis  # Push A in -axis direction to increase gap? 
                    # Actually if a_max < b_min, A is "before" B along axis.
                    # To separate, we can push A in -axis or B in +axis.
                    # We want direction to push A away from B.
                    # If A is at lower projection, pushing A in -axis direction separates them.
                    # But the "overlap" is zero here (gap > 0). 
                    # SAT: if gap exists on any axis, polys are separated.
                    # Return 0.
                    return 0.0, np.array([0.0, 0.0])
            if b_max < a_min:
                depth = a_min - b_max
                if depth > min_depth:
                    min_depth = depth
                    min_dir = axis
                # Separated
                return 0.0, np.array([0.0, 0.0])
            # Overlap on this axis
            # The overlap interval is [max(a_min, b_min), min(a_max, b_max)]
            # The "penetration" along this axis is the amount one must move to separate.
            # For the minimum translation vector, we consider the overlap depth on each axis.
            # The Minkowski difference approach: the depth of overlap along axis = 
            #   min(a_max, b_max) - max(a_min, b_min)
            # But for the MTV, we need to find the axis with minimum overlap depth and direction.
            # Actually, for convex polys, the minimum translation vector to separate is along
            # the axis of minimum overlap (in terms of the required displacement).
            # The required displacement along axis 'a' to separate is:
            #   if a_max < b_min: a_max - b_min (negative, meaning push A in -a direction)
            #   if b_max < a_min: b_max - a_min (positive, meaning push A in +a direction)
            #   else (overlap): the amount of overlap, but direction depends on which is "inside" which.
            # For our purpose, we just need a penalty for overlap. We can use the minimum
            # overlap depth across all axes as a measure of penetration.
            overlap = min(a_max, b_max) - max(a_min, b_min)
            if overlap < 0:
                # This shouldn't happen if we didn't return early
                return 0.0, np.array([0.0, 0.0])
            # The penetration depth along this axis
            # We want to minimize the maximum penetration over all axes? No, SAT says
            # polys overlap iff all axes have overlap. The minimum translation to separate
            # is along the axis with the smallest overlap, and the depth is that overlap.
            if overlap < min_depth:
                min_depth = overlap
                # Direction: we need to determine which way to push A.
                # Center of A vs center of B projected on axis
                ca = np.mean(A, axis=0)
                cb = np.mean(B, axis=0)
                diff = (ca - cb) @ axis
                if diff > 0:
                    min_dir = axis  # Push A in +axis direction
                else:
                    min_dir = -axis

        return min_depth, min_dir / (np.linalg.norm(min_dir) + 1e-12) if np.linalg.norm(min_dir) > 1e-12 else np.array([0.0, 0.0])

    def point_in_hex(v, cx, cy, S, theta):
        """
        Check if point v is inside regular hexagon centered at (cx,cy) with side S, rotation theta.
        Returns signed distance: negative if inside, 0 on boundary, positive if outside.
        For a regular hexagon, the inradius is S * sqrt(3)/2.
        """
        # Transform point to hexagon's local frame
        dx = v[0] - cx
        dy = v[1] - cy
        # Rotate by -theta
        c, s = math.cos(-theta), math.sin(-theta)
        lx = c * dx - s * dy
        ly = s * dx + c * dy

        # Flat-top hexagon: vertices at (S,0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S,0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
        # The hexagon is the intersection of 6 half-planes.
        # For flat-top, the 6 edge normals point at angles 0, 60, 120, 180, 240, 300 degrees? No.
        # Edge normals for flat-top hexagon point at angles 90, 150, 210, 270, 330, 30 degrees.
        # Actually, let's compute the distance to each of the 6 edges.
        # Vertices in local frame:
        # v0 = (S, 0)
        # v1 = (S/2, S*sqrt(3)/2)
        # v2 = (-S/2, S*sqrt(3)/2)
        # v3 = (-S, 0)
        # v4 = (-S/2, -S*sqrt(3)/2)
        # v5 = (S/2, -S*sqrt(3)/2)
        # Edge from v0 to v1: direction = (-S/2, S*sqrt(3)/2), normal pointing inward = (S*sqrt(3)/2, S/2) normalized
        # Inradius = S * sqrt(3) / 2
        # The 6 edge normals (inward) in local frame:
        # Edge v0-v1: normal angle = 30 deg
        # Edge v1-v2: normal angle = 90 deg
        # Edge v2-v3: normal angle = 150 deg
        # Edge v3-v4: normal angle = 210 deg
        # Edge v4-v5: normal angle = 270 deg
        # Edge v5-v0: normal angle = 330 deg
        # Distance from point to edge k = dot(local_point, normal_k) - inradius
        # If all distances <= 0, point is inside.
        # Signed distance = max of (dot - inradius) over all 6 edges.
        sqrt3 = math.sqrt(3)
        inradius = S * sqrt3 / 2.0
        normals = []
        for k in range(6):
            angle = math.pi / 6.0 + k * math.pi / 3.0  # 30, 90, 150, 210, 270, 330 degrees
            normals.append((math.cos(angle), math.sin(angle)))

        max_dist = -float('inf')
        for nx, ny in normals:
            d = lx * nx + ly * ny - inradius
            if d > max_dist:
                max_dist = d
        return max_dist

    def compute_objective(state, S):
        """
        state: flat array [x1,y1,theta1, x2,y2,theta2, ..., xN,yN,thetaN]
        S: outer side length
        Returns: penalty value (0 if feasible, >0 if violations)
        """
        n = len(state) // 3
        penalty = 0.0
        hexes = []
        for i in range(n):
            x = state[3*i]
            y = state[3*i+1]
            theta = state[3*i+2]
            verts = hex_vertices(x, y, theta, 1.0)
            hexes.append(verts)
            # Containment check
            for v in verts:
                d = point_in_hex(v, 0.0, 0.0, S, 0.0)
                if d > 0:
                    penalty += d * d

        # Collision check
        for i in range(n):
            for j in range(i+1, n):
                depth, _ = sat_min_depth(hexes[i], hexes[j])
                if depth > 0:
                    penalty += depth * depth

        return penalty

    def optimize(state_init, S_init):
        """
        Nelder-Mead style optimization of [x1,y1,theta1, ..., xN,yN,thetaN, S]
        """
        # Variables: 3*N positions/angles + 1 S = 3*11 + 1 = 34
        n_vars = 3 * N + 1
        # Initial simplex
        simplex = [np.array(state_init + [S_init])]
        for i in range(n_vars):
            perturbed = np.array(state_init + [S_init])
            perturbed[i] += 0.1 if i < 3*N else 0.05
            simplex.append(perturbed)

        def obj(x):
            S = x[-1]
            if S < 0.1:
                return 1e6
            return compute_objective(x[:3*N], S) + S

        best_x = simplex[0].copy()
        best_f = obj(best_x)

        max_iter = 5000
        for it in range(max_iter):
            if time.time() > DEADLINE:
                break

            # Sort simplex
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    fi = obj(simplex[i])
                    fj = obj(simplex[j])
                    if fi > fj:
                        simplex[i], simplex[j] = simplex[j], simplex[i]

            # Check convergence
            if it > 0:
                f_max = obj(simplex[-1])
                f_min = obj(simplex[0])
                if abs(f_max - f_min) < 1e-10:
                    break

            # Centroid of all but worst
            centroid = np.mean([simplex[i] for i in range(len(simplex)-1)], axis=0)

            # Reflection
            worst = simplex[-1]
            reflected = centroid + (centroid - worst)
            f_ref = obj(reflected)

            if f_ref < obj(simplex[0]):
                # Expansion
                expanded = centroid + 2*(reflected - centroid)
                f_exp = obj(expanded)
                if f_exp < f_ref:
                    simplex[-1] = expanded
                else:
                    simplex[-1] = reflected
            elif f_ref < obj(simplex[-2]):
                simplex[-1] = reflected
            else:
                # Contraction
                if f_ref < obj(worst):
                    contracted = centroid + 0.5*(reflected - centroid)
                else:
                    contracted = centroid + 0.5*(worst - centroid)
                f_con = obj(contracted)
                if f_con < min(f_ref, obj(worst)):
                    simplex[-1] = contracted
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5*(simplex[i] - simplex[0])

            if obj(simplex[0]) < best_f:
                best_f = obj(simplex[0])
                best_x = simplex[0].copy()

        return best_x

    # Initial configuration: hexagonal cluster
    # Center + 6 neighbors + 4 more
    # Hex lattice: neighbors at distance sqrt(3) (for unit hexes, center-to-center distance for touching = sqrt(3))
    d = math.sqrt(3)  # center-to-center distance for touching unit hexagons (flat-top)
    # Actually, for two unit flat-top hexagons to touch, center distance = sqrt(3)
    # Let's verify: inradius = sqrt(3)/2. Two hexagons touch when distance = 2 * inradius = sqrt(3). Yes.

    positions = []
    # Center
    positions.append((0.0, 0.0, 0.0))
    # 6 neighbors
    for k in range(6):
        angle = k * math.pi / 3.0
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        positions.append((x, y, 0.0))
    # 4 more: place them at distance 2*d from center, at angles 30, 150, 210, 330 degrees
    for k in range(4):
        angle = math.pi / 6.0 + k * math.pi / 2.0  # 30, 150, 270, 390=30? No: 30, 150, 270, 30+360?
        # Let's use 30, 150, 270, 390->30. That's only 3 distinct. 
        # Better: 30, 90, 150, 270? No, let's just use 4 positions at distance 2d.
        pass

    # Let me reconsider: 11 hexagons. Center + 6 ring = 7. Need 4 more.
    # Place 4 more at distance 2*d from center, at angles 0, 120, 240 degrees? That's 3.
    # Or at the "corners" of the hexagonal cluster.
    # The hexagonal cluster of 7 has a hexagonal shape. The next ring would have 12 hexes.
    # We need 4 more. Place them at 4 of the 12 positions of the second ring.
    # Second ring positions: distance 2*d, angles: 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330 degrees
    # Pick 4 symmetrically: 0, 120, 240 degrees (3) and one more? Or 30, 150, 270, 30+360?
    # Let's use 4 positions: angles 0, 120, 240, and... maybe just 4 at 90-degree intervals? 
    # Actually, for symmetry, let's use 4 positions at angles 45, 135, 225, 315 degrees at distance 2*d.
    for k in range(4):
        angle = math.pi / 4.0 + k * math.pi / 2.0
        x = 2 * d * math.cos(angle)
        y = 2 * d * math.sin(angle)
        positions.append((x, y, 0.0))

    # Flatten
    state_init = []
    for x, y, theta in positions:
        state_init.extend([x, y, theta])

    # Initial S estimate: the bounding hexagon needs to contain all these hexes.
    # Max distance from center: 2*d + 1 (inradius of outer hex needs to cover 2*d + inradius of inner hex)
    # inradius_outer >= 2*d + sqrt(3)/2
    # S_outer * sqrt(3)/2 >= 2*sqrt(3) + sqrt(3)/2
    # S_outer >= 2 + 0.5 = 2.5
    # So S ~ 2.5 to 3.0
    S_init = 3.0

    # Run optimization
    best_state = optimize(state_init, S_init)
    best_S = best_state[-1]
    best_inner = []
    for i in range(N):
        x = best_state[3*i]
        y = best_state[3*i+1]
        theta = best_state[3*i+2]
        best_inner.append([x, y, math.degrees(theta)])

    # Check if feasible
    penalty = compute_objective(best_state[:3*N], best_S)
    if penalty > 1e-6:
        # Try a few more random restarts
        np.random.seed(42)
        for restart in range(20):
            if time.time() > DEADLINE:
                break
            # Random perturbation of initial config
            state_rand = []
            for i in range(N):
                x = positions[i][0] + np.random.randn() * 0.5
                y = positions[i][1] + np.random.randn() * 0.5
                theta = positions[i][2] + np.random.randn() * 0.2
                state_rand.extend([x, y, theta])
            S_rand = S_init + np.random.randn() * 0.2
            state_result = optimize(state_rand, S_rand)
            S_res = state_result[-1]
            pen = compute_objective(state_result[:3*N], S_res)
            if pen < penalty:
                penalty = pen
                best_state = state_result
                best_S = S_res
                best_inner = []
                for i in range(N):
                    x = best_state[3*i]
                    y = best_state[3*i+1]
                    theta = best_state[3*i+2]
                    best_inner.append([x, y, math.degrees(theta)])

    # Final polish: if feasible, try to reduce S
    if penalty < 1e-6:
        # Binary search on S
        S_low = best_S * 0.5
        S_high = best_S
        for _ in range(50):
            if time.time() > DEADLINE:
                break
            S_mid = (S_low + S_high) / 2
            # Try to find feasible config with S_mid
            state_test = optimize(state_init, S_mid)
            pen_test = compute_objective(state_test[:3*N], S_mid)
            if pen_test < 1e-6:
                S_high = S_mid
                best_state = state_test
                best_S = S_mid
                best_inner = []
                for i in range(N):
                    x = best_state[3*i]
                    y = best_state[3*i+1]
                    theta = best_state[3*i+2]
                    best_inner.append([x, y, math.degrees(theta)])
            else:
                S_low = S_mid

    return best_inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
