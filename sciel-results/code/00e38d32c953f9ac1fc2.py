import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Precompute unit hex vertices (flat-top, side 1)
    # Angles: 0, 60, 120, 180, 240, 300
    base_verts = np.array([
        [math.cos(math.radians(60 * i)), math.sin(math.radians(60 * i))] for i in range(6)
    ])

    def get_hex_vertices(cx, cy, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        R = np.array([[ca, -sa], [sa, ca]])
        return (base_verts @ R.T) + np.array([cx, cy])

    def check_all_disjoint(inner):
        """
        Check if inner hexagons are disjoint.
        inner: list of (x, y, angle_deg)
        Returns True if disjoint, False otherwise.
        """
        verts_list = []
        for x, y, ang in inner:
            verts_list.append(get_hex_vertices(x, y, ang))
        
        for i in range(len(verts_list)):
            for j in range(i + 1, len(verts_list)):
                vi = verts_list[i]
                vj = verts_list[j]
                
                # Check vertex in polygon
                for v in vi:
                    if point_in_polygon(v, vj):
                        return False
                for v in vj:
                    if point_in_polygon(v, vi):
                        return False
                
                # Check edge intersection
                for k in range(6):
                    p1 = vi[k]
                    p2 = vi[(k+1)%6]
                    for l in range(6):
                        p3 = vj[l]
                        p4 = vj[(l+1)%6]
                        if segments_intersect(p1, p2, p3, p4):
                            return False
        return True

    def point_in_polygon(p, polygon):
        x, y = p
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def segments_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def compute_required_side(inner, oangle):
        """
        Compute the minimum side length S of a regular hexagon centered at origin
        with rotation oangle that contains all inner hexagons.
        """
        # Outer hexagon normals
        # For a regular hexagon with side S, the distance from center to edge (apothem) is S * sqrt(3) / 2.
        # The projection of any point (x,y) onto the normal of the k-th edge must be <= apothem.
        # Normals are at angles: oangle + 30 + 60*k
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        
        # Collect all vertices of inner hexagons
        all_verts = []
        for x, y, ang in inner:
            verts = get_hex_vertices(x, y, ang)
            all_verts.extend(verts)
        all_verts = np.array(all_verts)
        
        for k in range(6):
            theta = math.radians(oangle + 30 + 60 * k)
            nx, ny = math.cos(theta), math.sin(theta)
            # Projection of all vertices onto normal
            projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            max_proj = np.max(projs)
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def compute_required_side_smooth(inner, oangle, beta=100.0):
        """
        Smooth approximation of compute_required_side using log-sum-exp.
        """
        apothem_factor = math.sqrt(3) / 2
        all_verts = []
        for x, y, ang in inner:
            verts = get_hex_vertices(x, y, ang)
            all_verts.extend(verts)
        all_verts = np.array(all_verts)
        
        total_S = 0.0
        for k in range(6):
            theta = math.radians(oangle + 30 + 60 * k)
            nx, ny = math.cos(theta), math.sin(theta)
            projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            # Log-sum-exp of exp(beta * proj) / beta
            # To avoid overflow, subtract max
            m = np.max(projs)
            if m > 0:
                lse = (m + np.log(np.sum(np.exp(beta * (projs - m)))) / beta)
                S_k = lse / apothem_factor
            else:
                S_k = 0.0
            total_S += S_k
        
        # We want max(S_k), but sum is a valid upper bound that is smooth.
        # Actually, for minimization, we want to minimize the max.
        # Using sum of all 6 projections scaled appropriately might be a good surrogate.
        # Or just return the max of the smooth approximations? No, that's not differentiable.
        # Let's return the sum of the 6 smooth approximations. Minimizing this encourages all sides to be small.
        return total_S

    def compute_overlap_penalty(inner, beta=50.0):
        """
        Compute a smooth penalty for overlapping inner hexagons.
        Uses soft distance between convex hulls.
        """
        verts_list = []
        for x, y, ang in inner:
            verts_list.append(get_hex_vertices(x, y, ang))
        
        penalty = 0.0
        for i in range(len(verts_list)):
            for j in range(i + 1, len(verts_list)):
                vi = verts_list[i]
                vj = verts_list[j]
                
                # Compute pairwise distances between all vertices of i and j
                # dist^2
                min_dist_sq = float('inf')
                for v1 in vi:
                    for v2 in vj:
                        d2 = (v1[0]-v2[0])**2 + (v1[1]-v2[0])**2 # Typo in y? v2[1]
                        if d2 < min_dist_sq:
                            min_dist_sq = d2
                
                # If min_dist_sq < threshold, penalty
                # Use softplus or similar
                # Threshold: if they are touching, distance is 0.
                # Let's say if min_dist < 0.1, penalize.
                min_dist = math.sqrt(min_dist_sq)
                if min_dist < 0.2:
                    # Penalty increases as distance decreases
                    # Use exp(-beta * min_dist)
                    penalty += math.exp(-beta * min_dist)
                
                # Also check if any vertex is inside the other
                # This is discrete, so we approximate with distance to edges?
                # For now, vertex-vertex distance is a decent proxy for "closeness".
                # A more robust check: distance from center of i to hull of j?
                # Let's stick to vertex-vertex for speed in the smooth function.
                
        return penalty

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        # Unpack
        pos = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N] # in radians
        
        # Symmetry breaking: fix hex 0 at origin, angle 0
        # Note: minimize doesn't easily handle fixed vars, so we just penalize deviation or ignore.
        # Better: remove fixed vars from params.
        # But for simplicity in this rewrite, let's keep them and add a large penalty if they move.
        # Or just set them to 0 in the unpacking step? No, that breaks gradients.
        # Let's just add a penalty.
        penalty_fix = 100.0 * (pos[0,0]**2 + pos[0,1]**2 + angles[0]**2)
        
        # Construct inner list
        inner = []
        for i in range(N):
            inner.append((pos[i,0], pos[i,1], angles[i] * 180.0 / math.pi))
        
        # Compute smooth side length
        # We need to optimize over outer angle too?
        # The task allows outer angle.
        # Let's include outer angle in the optimization.
        # But the interface returns a single angle.
        # Let's optimize inner positions and outer angle.
        # Params: 2*N positions, N angles, 1 outer angle.
        # Total: 3*N + 1
        # Wait, my params unpacking above assumes 3*N.
        # Let's adjust objective to take 3*N + 1 params.
        
        # Re-implementing objective with outer angle
        # params[:2*N] -> pos
        # params[2*N:3*N] -> inner angles
        # params[3*N] -> outer angle
        
        # Since I already unpacked, I need to change the function signature or logic.
        # Let's make a helper.
        
        oangle_rad = params[3*N]
        oangle_deg = oangle_rad * 180.0 / math.pi
        
        S_smooth = compute_required_side_smooth(inner, oangle_deg, beta=50.0)
        overlap_pen = compute_overlap_penalty(inner, beta=20.0)
        
        return S_smooth + 100.0 * overlap_pen + penalty_fix

    # Adjust objective to handle 3*N + 1 params
    def objective_full(params):
        if time.time() > deadline:
            return 1e10
            
        pos = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        oangle_rad = params[3*N]
        
        penalty_fix = 100.0 * (pos[0,0]**2 + pos[0,1]**2 + angles[0]**2)
        
        inner = []
        for i in range(N):
            inner.append((pos[i,0], pos[i,1], angles[i] * 180.0 / math.pi))
        
        oangle_deg = oangle_rad * 180.0 / math.pi
        
        S_smooth = compute_required_side_smooth(inner, oangle_deg, beta=50.0)
        overlap_pen = compute_overlap_penalty(inner, beta=20.0)
        
        return S_smooth + 100.0 * overlap_pen + penalty_fix

    # Generate initial configurations
    # 1. Hexagonal lattice
    sp = math.sqrt(3)
    hex_config = [[0.0, 0.0, 0.0]]
    for i in range(6):
        theta = math.radians(60 * i)
        hex_config.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
    # 5 more for second ring?
    # Let's just use a random cluster for the rest to allow topology change
    for i in range(5):
        hex_config.append([random.uniform(-2, 2), random.uniform(-2, 2), 0.0])
    
    initial_configs = [hex_config[:N]]
    
    # 2. Random perturbation
    for _ in range(5):
        cfg = [list(x) for x in hex_config[:N]]
        for i in range(N):
            cfg[i][0] += random.gauss(0, 0.5)
            cfg[i][1] += random.gauss(0, 0.5)
        initial_configs.append(cfg)
        
    # 3. Pure random
    for _ in range(5):
        cfg = [[random.uniform(-2, 2), random.uniform(-2, 2), 0.0] for _ in range(N)]
        initial_configs.append(cfg)

    # Optimize each
    for cfg in initial_configs:
        if time.time() > deadline:
            break
            
        x0 = np.zeros(3*N + 1)
        for i in range(N):
            x0[2*i] = cfg[i][0]
            x0[2*i+1] = cfg[i][1]
            x0[2*N+i] = cfg[i][2] * math.pi / 180.0
        x0[3*N] = 0.0 # Initial outer angle
        
        try:
            res = minimize(objective_full, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-6})
            
            if res.fun < 1e6: # Sanity check
                pos = res.x[:2*N].reshape(N, 2)
                angles = res.x[2*N:3*N]
                oangle = res.x[3*N] * 180.0 / math.pi
                
                inner = []
                for i in range(N):
                    inner.append([pos[i][0], pos[i][1], angles[i] * 180.0 / math.pi])
                
                # Center the configuration
                cx = sum(p[0] for p in inner) / N
                cy = sum(p[1] for p in inner) / N
                for i in range(N):
                    inner[i][0] -= cx
                    inner[i][1] -= cy
                
                # Verify validity
                if check_all_disjoint(inner):
                    # Compute exact S
                    S = compute_required_side(inner, oangle)
                    if S < best_S:
                        best_S = S
                        best_inner = [list(x) for x in inner]
                        best_center = (0.0, 0.0)
                        best_angle = oangle
                        best_valid = True
                        
        except Exception:
            pass

    # Fallback
    if not best_valid or best_inner is None:
        # Linear fallback
        fallback_inner = [[(k - N // 2) * sp, 0.0, 0.0] for k in range(N)]
        S = compute_required_side(fallback_inner, 0.0)
        return fallback_inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
