import numpy as np
import time
import itertools

def _hex_vertices(cx, cy, d, a):
    """
    Compute vertices of a regular hexagon.
    cx, cy: center
    d: side length
    a: angle in degrees
    """
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    """
    Compute the minimum side length of a regular hexagon with the given angle
    that encloses all unit hexagons at the given positions.
    Assumes inner hexagons are aligned with angle 0 relative to their local frame,
    but here we pass absolute positions.
    
    Actually, for the rigid cluster strategy, we will rotate the positions themselves.
    This function assumes the outer hexagon has angle 'angle_deg' and centers at (0,0) 
    for the sake of finding 's', but we need to handle translation too.
    
    Better approach for this function: 
    Given a set of vertices (absolute coords), find min s for outer hex at (0,0) with angle angle_deg.
    But we want to optimize translation as well.
    
    Let's define a function that takes the cluster vertices (relative to cluster center) 
    and the outer hexagon parameters (center, angle), and returns min s.
    
    Actually, the outer hexagon center is a free variable too.
    However, for a convex body enclosing another convex body, the minimal enclosing hexagon 
    for a fixed orientation is determined by the support functions.
    
    Let's simplify:
    1. Fix the cluster shape (12 hexes in honeycomb).
    2. Rotate the cluster by theta.
    3. The cluster is a set of points V.
    4. We want to find min s such that there exists a translation t such that V + t is inside 
       Hex(0, 0, s, phi).
       
    Note: The outer hexagon's position (x,y) and angle (phi) are variables.
    Due to symmetry, we can fix the outer hexagon's center to (0,0) and let the cluster 
    translate? No, the cluster must be centered? 
    Actually, the problem asks for (inner, outer, s). 
    inner: (12, 3) array (x, y, angle).
    outer: (3,) array (x, y, angle).
    
    If we fix the outer hexagon to be centered at (0,0) with angle 0, we can just move the cluster.
    Is it optimal to align the outer hexagon? 
    Usually, yes, for symmetric clusters. But the cluster might not be symmetric.
    The honeycomb cluster of 12 is not perfectly symmetric (it's a hexagon of radius 2 minus corners? 
    No, 1+6+5=12. A full hex of radius 2 has 19. 
    12 hexes can be arranged in a 3-4-3-2 or similar compact shape.
    
    Let's stick to the Rigid Cluster idea:
    1. Generate a specific 12-hexagon layout.
    2. Add epsilon to spacing to ensure separation > 1e-6.
    3. Optimize: 
       - Outer hexagon angle (0 to 60 degrees, symmetry)
       - Cluster rotation (0 to 360)
       - Cluster translation (x, y) -- wait, if we fix outer at (0,0), translation of cluster is just centering.
       
    Actually, for a fixed orientation of both, the minimal s is determined by the projection of the 
    cluster's vertices onto the outer hexagon's normals.
    If we allow the cluster to translate, the minimal s for fixed orientations is:
    s = max_k ( (max_proj_k - min_proj_k) ) / (2 * cos(30)) ? No.
    
    Let's derive:
    Outer hex H with center c_out, side s, angle phi.
    Inner cluster C with vertices V (relative to cluster center c_in).
    Absolute vertices: V_abs = R(theta) V + t.
    Constraint: V_abs inside H.
    
    H is intersection of 6 half-planes.
    Normal n_i at angle phi + 60i + 30.
    Distance from center c_out to edge i is s * cos(30) = s * sqrt(3)/2.
    So, for all v in V_abs:
    dot(v - c_out, n_i) <= s * sqrt(3)/2  for all i=0..5
    AND
    dot(v - c_out, -n_i) <= s * sqrt(3)/2  for all i=0..5 (since normals cover all directions, we just need 6 normals if we define them correctly as outward normals of edges).
    
    Actually, a regular hexagon centered at c with side s and angle phi has edges with outward normals 
    n_i = (cos(phi + 60i + 30), sin(phi + 60i + 30)) for i=0..5.
    The support function in direction n_i is s * cos(30) = s * sqrt(3)/2.
    So constraint: max_{v in V_abs} dot(v - c_out, n_i) <= s * sqrt(3)/2.
    Also min_{v in V_abs} dot(v - c_out, n_i) >= -s * sqrt(3)/2.
    
    Let P_i = max_{v in V} dot(R(theta)v, n_i) and Q_i = min_{v in V} dot(R(theta)v, n_i).
    Let T = t - c_out.
    dot(R(theta)v + T, n_i) = dot(R(theta)v, n_i) + dot(T, n_i).
    
    Max constraint: max_v (dot(R(theta)v, n_i) + dot(T, n_i)) <= s * sqrt(3)/2
    => P_i + dot(T, n_i) <= s * sqrt(3)/2
    
    Min constraint: min_v (dot(R(theta)v, n_i) + dot(T, n_i)) >= -s * sqrt(3)/2
    => Q_i + dot(T, n_i) >= -s * sqrt(3)/2
    
    We want to minimize s.
    For a fixed theta and phi, we can choose T to minimize the required s.
    Let u_i = dot(T, n_i).
    We need:
    u_i <= s*sqrt(3)/2 - P_i
    u_i >= -s*sqrt(3)/2 - Q_i
    
    So, -s*sqrt(3)/2 - Q_i <= u_i <= s*sqrt(3)/2 - P_i
    This implies:
    -s*sqrt(3)/2 - Q_i <= s*sqrt(3)/2 - P_i
    => P_i - Q_i <= s * sqrt(3)
    => s >= (P_i - Q_i) / sqrt(3)
    
    So for fixed theta, phi, the minimum s is:
    s_min = max_i ( (P_i - Q_i) / sqrt(3) )
    
    And the optimal T is determined by setting u_i such that the interval is centered?
    Actually, we just need to find *a* T that satisfies the constraints.
    If we set T=0 (cluster center at outer center), does it work?
    If the cluster is centered at 0, then P_i and Q_i are roughly symmetric.
    The condition s >= (P_i - Q_i)/sqrt(3) is necessary.
    Is it sufficient?
    We need to find u_i = dot(T, n_i) such that:
    -s*sqrt(3)/2 - Q_i <= u_i <= s*sqrt(3)/2 - P_i
    Let L_i = -s*sqrt(3)/2 - Q_i and U_i = s*sqrt(3)/2 - P_i.
    We need to find T such that dot(T, n_i) is in [L_i, U_i] for all i.
    Since n_i are 6 vectors in 2D, this is a feasibility problem in 2D.
    However, if we assume the cluster is "centered" (centroid at 0), then P_i ≈ -Q_i.
    Then L_i ≈ -s*sqrt(3)/2 + P_i and U_i ≈ s*sqrt(3)/2 - P_i.
    Midpoint is 0. So u_i = 0 is a good candidate.
    If u_i = 0 works for all i, then T=0 is optimal.
    0 <= s*sqrt(3)/2 - P_i  => s >= 2*P_i/sqrt(3)
    0 >= -s*sqrt(3)/2 - Q_i => s >= -2*Q_i/sqrt(3)
    
    If the cluster is not perfectly symmetric, T=0 might not be optimal.
    But for a hexagonal lattice cluster, it is likely close to optimal.
    Let's refine: We can optimize T as well.
    But T is 2D. s is 1D. theta, phi are 1D each. Total 5D search.
    
    Strategy:
    1. Define cluster vertices V (12 hexes, 72 vertices total).
    2. Search over theta (cluster rotation) and phi (outer rotation).
    3. For each (theta, phi), compute P_i, Q_i.
    4. Calculate s_candidate = max_i ( (P_i - Q_i) / sqrt(3) ).
    5. Check if T=0 is feasible:
       For all i:
       -s*sqrt(3)/2 - Q_i <= 0 <= s*sqrt(3)/2 - P_i
       i.e., Q_i <= s*sqrt(3)/2 and P_i <= s*sqrt(3)/2.
       Since s = max (P-Q)/sqrt(3), this is not automatically true.
       Example: P=10, Q=-10. s=20/sqrt(3)=11.5.
       s*sqrt(3)/2 = 10.
       10 <= 10 (OK). -10 <= 10 (OK).
       
       If P=11, Q=-9. Width=20. s=11.5.
       s*sqrt(3)/2 = 10.
       P=11 > 10. T=0 fails.
       We need to shift T.
       u_i must be <= 10 - 11 = -1.
       u_i must be >= -10 - (-9) = -1.
       So u_i = -1.
       
       So, for each i, we have an interval for u_i = dot(T, n_i).
       We need to find T such that dot(T, n_i) in [L_i, U_i].
       This is a 2D linear programming problem (or just check feasibility).
       If feasible, the s is valid.
       
       To make it simple:
       We will optimize theta and phi using a coarse grid or Nelder-Mead.
       For each (theta, phi), we compute the required s and check feasibility of T.
       If feasible, we record s.
       Then we can refine T? No, we just need to return *a* valid solution.
       If T=0 is not feasible, we can try to find T.
       
       Actually, finding T is easy:
       We have 6 constraints on 2 variables (T_x, T_y).
       n_i = (cos(alpha_i), sin(alpha_i)).
       L_i <= T_x cos(alpha_i) + T_y sin(alpha_i) <= U_i.
       
       We can use scipy.optimize.linprog to find T?
       Or just use a small local search for T?
       Or, simpler:
       Since the cluster is convex and symmetric-ish, T=0 is often very close to optimal.
       If T=0 fails, we can adjust T in the direction of the failing constraint.
       
       Let's implement a function `_get_min_s_and_T(V, theta, phi)` that returns s and T.
       
       1. Rotate V by theta.
       2. Compute P_i, Q_i for 6 normals of outer hex (angle phi).
       3. s = max(P_i - Q_i) / sqrt(3).
       4. Check if T=0 works.
          If yes, return s, 0.
          If no, solve for T.
          
       Solving for T:
       We want to minimize s? No, s is fixed by the width.
       We just need *any* T that fits.
       Actually, if s is determined by width, T is just for centering.
       The minimal s is indeed max width / sqrt(3).
       Wait, is it?
       The width of the cluster in direction n is W_n = P_n - Q_n.
       The width of the outer hex in direction n is s * sqrt(3).
       So s >= W_n / sqrt(3).
       So s_min = max_n (W_n / sqrt(3)).
       And we need to fit the cluster inside.
       This is always possible if s >= s_min?
       Yes, because we can translate the cluster to center it in the hexagon.
       The hexagon has width s*sqrt(3) in every normal direction.
       The cluster has width W_n <= s*sqrt(3).
       So there exists a translation T such that the cluster fits.
       
       So the algorithm is:
       1. For given theta, phi:
          a. Compute W_n for 6 directions.
          b. s = max(W_n) / sqrt(3).
          c. Find T such that cluster fits.
             T is the center of the "gap" in each direction?
             For direction n_i, the cluster occupies [Q_i + u_i, P_i + u_i].
             The hexagon occupies [-s*sqrt(3)/2 + u_i_center?, ...] 
             Wait, the hexagon is fixed at (0,0) in this local frame?
             No, the hexagon is at c_out. The cluster is at c_in.
             Relative T = c_in - c_out.
             Constraint: Cluster + T inside Hex(0,0,s,phi).
             Hex(0,0,s,phi) is defined by:
             dot(x, n_i) <= s*sqrt(3)/2
             dot(x, -n_i) <= s*sqrt(3)/2  (which is dot(x, n_i) >= -s*sqrt(3)/2)
             
             So we need:
             Q_i + dot(T, n_i) >= -s*sqrt(3)/2
             P_i + dot(T, n_i) <= s*sqrt(3)/2
             
             => -s*sqrt(3)/2 - Q_i <= dot(T, n_i) <= s*sqrt(3)/2 - P_i
             
             Let L_i = -s*sqrt(3)/2 - Q_i
             Let U_i = s*sqrt(3)/2 - P_i
             
             We need to find T such that dot(T, n_i) in [L_i, U_i] for all i.
             
             Since n_i are 6 vectors, this is a 2D intersection of 6 strips.
             We can find T by taking the average of the constraints?
             Or use a simple iterative projection (Poisson)?
             Or, since it's 2D, we can just use scipy.optimize.minimize to find T that minimizes 
             sum of violations, or just pick T=0 and if it fails, adjust.
             
             Actually, T=0 works if L_i <= 0 <= U_i for all i.
             i.e., Q_i <= s*sqrt(3)/2 and P_i <= s*sqrt(3)/2.
             Since s = max(P-Q)/sqrt(3), let k be the max width direction.
             W_k = P_k - Q_k = s*sqrt(3).
             Then U_k = s*sqrt(3)/2 - P_k.
             L_k = -s*sqrt(3)/2 - Q_k.
             U_k - L_k = s*sqrt(3) - (P_k - Q_k) = 0.
             So for the critical direction, dot(T, n_k) must be exactly L_k = U_k.
             So T is uniquely determined by the critical direction?
             Not necessarily, if multiple directions are critical.
             
             Let's just use `scipy.optimize.linprog` to find a feasible T.
             Minimize 0 subject to:
             n_i^T T >= L_i
             -n_i^T T >= -U_i  => n_i^T T <= U_i
             
             If feasible, return T.
             If not (shouldn't happen if s is correct), increase s slightly.
             
       2. Optimize theta and phi to minimize s.
       
       This reduces the problem to 2D optimization (theta, phi) + linear programming for T.
       This is MUCH faster and more robust than 25D Nelder-Mead.
    """
    pass

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580
    
    # 1. Define the 12 hexagons in a honeycomb pattern
    # Hex grid coordinates (axial)
    # Center (0,0)
    # Ring 1: 6 hexes
    # Ring 2: 5 hexes (to make 12 total? 1+6+5=12)
    # A full ring 2 has 12 hexes. 1+6+12=19.
    # We need a compact shape.
    # Let's use a "lozenge" or "parallelogram" of 2x6? No, 12.
    # 3x4 grid?
    # Axial coords for a 3x4 block:
    # (0,0), (1,0), (2,0), (3,0)
    # (0,1), (1,1), (2,1), (3,1)
    # (0,2), (1,2), (2,2), (3,2)
    # (0,3)... no, 12 hexes.
    
    # Let's use the 1+6+5 arrangement.
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (partial): 
    # (2,0), (1,1), (0,2), (-1,2), (-2,1) ? 
    # Let's check distances.
    # Axial to Cartesian:
    # x = size * (q + r/2)
    # y = size * (r * sqrt(3)/2)
    
    size = 1.0
    eps = 1e-5 # Small epsilon to ensure separation
    
    # Define axial coordinates
    axial = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    
    # Convert to Cartesian
    centers = []
    for q, r in axial:
        x = size * (q + r/2.0)
        y = size * (r * np.sqrt(3)/2.0)
        centers.append([x, y])
        
    centers = np.array(centers)
    
    # Apply epsilon scaling from centroid to ensure strict separation
    centroid = np.mean(centers, axis=0)
    centers = centroid + (centers - centroid) * (1.0 + eps)
    
    # The vertices of the cluster
    # Each hex has 6 vertices.
    # We need the absolute vertices of all 12 hexes.
    # But for the optimization, we only need the set of all vertices V.
    # V will be rotated by theta.
    
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    
    # All vertices relative to cluster center (centroid)
    V_rel = []
    for c in centers:
        # Vertices are c + unit_hex_verts
        verts = c[np.newaxis, :] + unit_hex_verts
        V_rel.append(verts)
        
    V_rel = np.vstack(V_rel) # (72, 2)
    
    # Optimization over theta (cluster rotation) and phi (outer rotation)
    # Due to symmetry, we can restrict theta and phi to [0, 30] or [0, 60]?
    # Hexagon has 6-fold symmetry. Cluster has 1-fold (asymmetric).
    # So theta in [0, 360), phi in [0, 60).
    
    def _evaluate(theta, phi):
        """
        Returns s for given theta, phi.
        """
        # Rotate V_rel by theta
        theta_rad = np.deg2rad(theta)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        V_rot = V_rel @ R.T
        
        # Normals for outer hexagon with angle phi
        phi_rad = np.deg2rad(phi)
        normals = []
        for i in range(6):
            n_angle = phi_rad + i * np.pi/3 + np.pi/6
            normals.append(np.array([np.cos(n_angle), np.sin(n_angle)]))
        normals = np.array(normals) # (6, 2)
        
        # Projections
        projs = V_rot @ normals.T # (72, 6)
        P = np.max(projs, axis=0) # (6,)
        Q = np.min(projs, axis=0) # (6,)
        
        widths = P - Q
        s = np.max(widths) / np.sqrt(3)
        
        # Check feasibility of T
        # L_i = -s*sqrt(3)/2 - Q_i
        # U_i = s*sqrt(3)/2 - P_i
        L = -s * np.sqrt(3)/2 - Q
        U = s * np.sqrt(3)/2 - P
        
        # Find T such that dot(T, n_i) in [L_i, U_i]
        # Use linprog
        from scipy.optimize import linprog
        
        # Minimize 0
        c = np.zeros(2)
        
        # Inequalities:
        # n_i^T T >= L_i  =>  -n_i^T T <= -L_i
        # -n_i^T T >= -U_i =>  n_i^T T <= U_i
        
        A_ub = []
        b_ub = []
        
        for i in range(6):
            # -n_i^T T <= -L_i
            A_ub.append(-normals[i])
            b_ub.append(-L[i])
            
            # n_i^T T <= U_i
            A_ub.append(normals[i])
            b_ub.append(U[i])
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
            if res.status == 0:
                T = res.x
                return s, T
            else:
                # Infeasible, increase s slightly
                return s * 1.01, np.zeros(2)
        except Exception:
            return s * 1.01, np.zeros(2)

    best_s = float('inf')
    best_theta = 0
    best_phi = 0
    best_T = np.zeros(2)
    
    # Coarse grid search
    thetas = np.linspace(0, 360, 36, endpoint=False)
    phis = np.linspace(0, 60, 12, endpoint=False)
    
    for theta in thetas:
        if time.time() > deadline: break
        for phi in phis:
            if time.time() > deadline: break
            s, T = _evaluate(theta, phi)
            if s < best_s:
                best_s = s
                best_theta = theta
                best_phi = phi
                best_T = T
                
    # Refine around best with Nelder-Mead
    def _objective(x):
        theta, phi = x
        s, _ = _evaluate(theta, phi)
        return s
        
    x0 = np.array([best_theta, best_phi])
    try:
        res = minimize(_objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.fun < best_s:
            best_s = res.fun
            best_theta, best_phi = res.x
            _, best_T = _evaluate(best_theta, best_phi)
    except Exception:
        pass
        
    # Construct final answer
    # Rotate centers by best_theta
    theta_rad = np.deg2rad(best_theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    final_centers = centers @ R.T + best_T
    
    # Inner hexagons: (x, y, angle)
    # Angle is 0 relative to local frame, but since we rotated the cluster, 
    # the absolute angle is best_theta.
    inner = np.zeros((12, 3))
    inner[:, 0] = final_centers[:, 0]
    inner[:, 1] = final_centers[:, 1]
    inner[:, 2] = best_theta
    
    # Outer hexagon
    outer = np.array([0, 0, best_phi])
    
    # Check validity and return
    # Ensure s is at least sqrt(12)
    if best_s < np.sqrt(12):
        best_s = np.sqrt(12)
        
    return inner, outer, best_s

# EVOLVE-BLOCK-END
