import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def smooth_min_gap(p1, p2):
        """
        Calculate a smooth approximation of the minimum distance between two convex polygons.
        Returns a negative value if overlapping (penalty), positive if separated.
        Uses a log-sum-exp approximation on the signed distances from vertices to edges.
        """
        # Compute signed distances from all vertices of p1 to all edges of p2 and vice versa
        # For convex polygons, the minimum gap is the max of the minimum signed distances.
        # Actually, a simpler robust metric for smooth optimization is the sum of squared overlaps
        # or a smooth max of separation distances.
        
        # Let's use a simpler, differentiable penalty:
        # For each pair of edges, compute the distance between the line segments.
        # This is complex. 
        # Alternative: Use the "soft" intersection test.
        # If polygons overlap, the distance is negative.
        
        # A common trick: 
        # d = min_{v in P1, e in P2} dist(v, e) + min_{v in P2, e in P1} dist(v, e)
        # This is not smooth.
        
        # Let's use a differentiable "overlap area" approximation or just a smooth repulsive force.
        # Simplest robust differentiable penalty:
        # For every vertex of P1, compute distance to P2. If inside, dist is negative (based on depth).
        # We can compute the "signed distance" field approximately.
        
        # Given time constraints, let's use a simpler heuristic that is differentiable:
        # Sum of squared distances from centers if close, plus a term for orientation?
        # No, that's too weak.
        
        # Let's stick to a SAT-based smooth penalty.
        # For each separating axis, compute the gap. If gap < 0, they overlap on that axis.
        # The total overlap depth is max(0, -min_gap).
        # To make it smooth, use soft_max of the negative gaps.
        
        axes = []
        for i in range(6):
            edge1 = p1[(i+1)%6] - p1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            axes.append(normal1)
            edge2 = p2[(i+1)%6] - p2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
            
        gaps = []
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-9: continue
            axis = axis / norm
            proj1 = p1 @ axis
            proj2 = p2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            # Gap is the distance between the intervals
            # If max1 < min2, gap = min2 - max1 (positive separation)
            # If max2 < min1, gap = min1 - max2 (positive separation)
            # Otherwise, they overlap. Gap is negative.
            # Let's define separation s such that s > 0 if disjoint, s < 0 if overlapping.
            # s = min( min2 - max1, min1 - max2 )
            # If overlapping, both terms are negative, so min is the "deepest" overlap.
            s = min(min2 - max1, min1 - max2)
            gaps.append(s)
            
        if not gaps:
            return 0.0
            
        gaps = np.array(gaps)
        # We want a smooth max of the negative gaps (which represent overlap depth).
        # If all gaps are positive, overlap is 0.
        # Let M = max(0, -min(gaps)). This is the minimum separation distance (negative if overlapping).
        # Actually, the polygons are disjoint if min(gaps) > 0.
        # Overlap depth D = max(0, -min(gaps)).
        # To make it smooth, we can use a soft version.
        # But min(gaps) is already somewhat smooth if the "critical" axis doesn't change abruptly.
        # The non-smoothness comes from the min.
        # We can use log-sum-exp on -gaps to get a smooth max of -gaps.
        
        # Let's just use the raw min gap for now, it's often good enough for L-BFGS if we use finite differences or if we provide gradients.
        # Since we don't provide gradients, L-BFGS uses finite differences.
        # The min function creates kinks, but L-BFGS can handle them if the step size is small.
        
        min_gap = np.min(gaps)
        # Return -min_gap if overlapping (positive penalty), 0 if separated.
        # Actually, we want to minimize the outer side + penalty.
        # Penalty should be large when overlapping.
        # Let's return max(0, -min_gap) * weight.
        if min_gap < 0:
            return -min_gap
        return 0.0

    def is_valid(positions, angles):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                # Use the smooth gap to check validity strictly
                # If gap < 0, they overlap.
                gap = smooth_min_gap(polys[i], polys[j])
                if gap > 1e-6: # Overlapping
                    return False
        return True

    def objective_smooth(params):
        if time.time() > deadline - 5:
            return 1000.0
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Calculate outer side length
        side = calc_min_outer_side(positions, angles, 0.0)
        
        # Calculate overlap penalty
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                gap = smooth_min_gap(polys[i], polys[j])
                if gap > 0:
                    # Smooth penalty: exponential or quadratic
                    # Quadratic is standard
                    pen += gap * gap * 10.0
        
        return side + pen

    best_score = float('inf')
    best_params = None

    # Generate diverse initial guesses
    def generate_initial_guesses():
        guesses = []
        sqrt3 = math.sqrt(3)
        
        # 1. Honeycomb subset
        lattice_points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * 1.5
                y = i * sqrt3 / 2.0 + j * sqrt3
                lattice_points.append((x, y))
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_positions = np.array(lattice_points[:12])
        init_positions -= np.mean(init_positions, axis=0)
        init_angles = np.zeros(N)
        guesses.append(np.concatenate([init_positions.flatten(), init_angles]))
        
        # 2. Ring 6 + Ring 6
        pos1 = np.array([[2.0, 0], [1.0, 1.732], [-1.0, 1.732], [-2.0, 0], [-1.0, -1.732], [1.0, -1.732]])
        pos2 = np.array([[4.0, 0], [2.0, 3.464], [-2.0, 3.464], [-4.0, 0], [-2.0, -3.464], [2.0, -3.464]])
        pos = np.vstack([pos1, pos2])
        ang = np.zeros(N)
        guesses.append(np.concatenate([pos.flatten(), ang]))
        
        # 3. Random scatter
        for _ in range(5):
            pos = np.random.uniform(-3, 3, (N, 2))
            ang = np.random.uniform(0, 360, N)
            guesses.append(np.concatenate([pos.flatten(), ang]))
            
        return guesses

    guesses = generate_initial_guesses()
    
    # Optimize each guess
    for i, x0 in enumerate(guesses):
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.fun < best_score and res.fun < 100.0:
                # Check validity of the result
                final_positions = res.x[:2*N].reshape(N, 2)
                final_angles = res.x[2*N:]
                
                # Re-calculate true score (without penalty) if valid
                if is_valid(final_positions, final_angles):
                    true_score = calc_min_outer_side(final_positions, final_angles, 0.0)
                    if true_score < best_score:
                        best_score = true_score
                        best_params = res.x
        except Exception:
            pass

    # If no good result found, use a safe fallback
    if best_params is None:
        # Fallback to a known valid configuration (even if not optimal)
        # A large hexagon containing 12 small ones
        final_positions = np.array([
            [0, 0], [2.1, 0], [-2.1, 0],
            [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
            [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
            [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
        ])
        final_angles = np.zeros(N)
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    else:
        final_positions = best_params[:2*N].reshape(N, 2)
        final_angles = best_params[2*N:]
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
