<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def hexagon_radius_in_direction(cx, cy, angle_deg, theta):
        """Maximum projection of hexagon vertices in direction theta."""
        angle = math.radians(angle_deg)
        max_proj = 0.0
        for i in range(6):
            vert_angle = angle + i * math.pi / 3
            # Vertex position relative to center
            vx = math.cos(vert_angle)
            vy = math.sin(vert_angle)
            proj = vx * math.cos(theta) + vy * math.sin(theta)
            if proj > max_proj:
                max_proj = proj
        return max_proj

    def check_disjoint(centers, angles, min_dist=math.sqrt(3) - 0.001):
        """Check if all hexagons are disjoint (center distance >= sqrt(3))."""
        n = len(centers)
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    return False
        return True

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        """Compute minimum outer hexagon side for given configuration."""
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                # For each hexagon, find max projection of its vertices
                cx, cy = centers[i]
                angle = angles[i]
                for vi in range(6):
                    vert_angle = angle + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def optimize_config(centers, angles, max_iters=500, step_size=0.1):
        """Local optimization to minimize outer hexagon side."""
        n = len(centers)
        best_centers = [c.copy() for c in centers]
        best_angles = angles.copy()
        best_side = compute_outer_side(centers, angles, 0, 0, 0)
        
        # Optimize outer hexagon center and angle first
        for outer_iter in range(50):
            if time.time() > deadline - 20:
                break
            # Try perturbing outer center and angle
            improved = False
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    for da in [-math.pi/18, 0, math.pi/18]:
                        side = compute_outer_side(centers, angles, dx, dy, da)
                        if side < best_side - 1e-10:
                            best_side = side
                            best_centers = centers
                            best_angles = angles
                            improved = True
            if not improved:
                break
        
        # Now optimize inner hexagon positions
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(n):
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        new_centers = [c.copy() for c in centers]
                        new_centers[i][0] += dx
                        new_centers[i][1] += dy
                        if check_disjoint(new_centers, angles):
                            side = compute_outer_side(new_centers, angles, 0, 0, 0)
                            if side < best_side - 1e-10:
                                best_side = side
                                best_centers = [c.copy() for c in new_centers]
                                best_angles = angles.copy()
                                centers = new_centers
                                improved = True
            if not improved:
                break
        
        return best_centers, best_angles, best_side

    # Lattice vectors for hexagonal packing
    eps = 0.01
    d = math.sqrt(3) + eps
    
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])
    
    # Ring 0
    centers = [np.array([0.0, 0.0])]
    
    # Ring 1
    ring1_vectors = [
        u,
        v,
        v - u,
        -u,
        -v,
        u - v
    ]
    for vec in ring1_vectors:
        centers.append(vec.copy())
        
    # Ring 2 Candidates
    ring2_candidates = []
    basis = [u, v, v-u, -u, -v, u-v]
    
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2)
                    
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    
    for i in range(5):
        centers.append(ring2_candidates[i].copy())
        
    assert len(centers) == 12
    
    # Convert to list format
    centers_list = [[c[0], c[1]] for c in centers]
    angles_list = [30.0] * 12
    
    # Run local optimization
    opt_centers, opt_angles, opt_side = optimize_config(centers_list, angles_list)
    
    # Use optimized configuration
    inner = [[c[0], c[1], a] for c, a in zip(opt_centers, opt_angles)]
    
    # Compute final outer hexagon
    # Find optimal outer center and angle
    best_outer_side = float('inf')
    best_ocx, best_ocy, best_otheta = 0.0, 0.0, 0.0
    
    # Try a range of outer hexagon angles and centers
    for otheta_deg in np.linspace(0, 60, 31):
        otheta = math.radians(otheta_deg)
        for ocx in [-1, 0, 1]:
            for ocy in [-1, 0, 1]:
                side = compute_outer_side(centers_list, angles_list, ocx, ocy, otheta)
                if side < best_outer_side:
                    best_outer_side = side
                    best_ocx, best_ocy, best_otheta = ocx, ocy, otheta_deg
    
    outer_center = (best_ocx, best_ocy)
    outer_side = best_outer_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0

    def compute_outer_side_vectorized(c_coords, c_angles, ocx, ocy, otheta):
        """
        Compute minimum outer hexagon side for given configuration.
        c_coords: numpy array (N, 2)
        c_angles: numpy array (N,)
        ocx, ocy: outer center
        otheta: outer angle in radians
        """
        vi = np.arange(6)
        thetas = c_angles[:, None] + vi[None, :] * math.pi / 3
        vxs = c_coords[:, 0:1] + np.cos(thetas)
        vys = c_coords[:, 1:2] + np.sin(thetas)
        
        k = np.arange(6)
        normal_angles = otheta + math.pi/6 + k * math.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        vxs_rel = vxs - ocx
        vys_rel = vys - ocy
        
        projs = vxs_rel[:, :, None] * nx[None, None, :] + vys_rel[:, :, None] * ny[None, None, :]
        max_proj = np.max(projs)
        
        return 2.0 * max_proj / math.sqrt(3)

    def objective(params):
        if time.time() > deadline - 5:
            return 1e10
        
        c_coords = params[:24].reshape(12, 2)
        c_angles = params[24:36]
        ocx = params[36]
        ocy = params[37]
        otheta = params[38]
        
        # Vectorized overlap check
        diff = c_coords[:, None, :] - c_coords[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        mask = np.triu(np.ones((12, 12), dtype=bool), k=1)
        
        overlaps = np.maximum(0, math.sqrt(3) - dists)
        penalty = np.sum(overlaps[mask] ** 2)
        
        side = compute_outer_side_vectorized(c_coords, c_angles, ocx, ocy, otheta)
        
        return side + 10.0 * penalty

    # Initial guess: Hexagonal lattice
    eps = 0.01
    d = math.sqrt(3) + eps
    
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])
    
    centers_init = [np.array([0.0, 0.0])]
    
    ring1_vectors = [
        u,
        v,
        v - u,
        -u,
        -v,
        u - v
    ]
    for vec in ring1_vectors:
        centers_init.append(vec.copy())
        
    ring2_candidates = []
    basis = [u, v, v-u, -u, -v, u-v]
    
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2)
                    
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    
    for i in range(5):
        centers_init.append(ring2_candidates[i].copy())
        
    assert len(centers_init) == 12
    
    # Flatten initial params
    init_params = []
    for c in centers_init:
        init_params.extend([c[0], c[1]])
    init_params.extend([0.0] * 12) # Angles
    init_params.extend([0.0, 0.0, 0.0]) # Outer center and angle
    
    best_x = np.array(init_params)
    best_f = objective(best_x)
    
    def callback(xk):
        nonlocal best_x, best_f
        if time.time() > deadline - 10:
            raise TimeoutError
        f_k = objective(xk)
        if f_k < best_f:
            best_f = f_k
            best_x = xk.copy()
            
    try:
        res = minimize(objective, np.array(init_params), method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5},
                       callback=callback)
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
    except TimeoutError:
        pass
    except Exception:
        pass
    
    c_coords = best_x[:24].reshape(12, 2)
    c_angles_rad = best_x[24:36]
    ocx = best_x[36]
    ocy = best_x[37]
    otheta = best_x[38]
    
    c_angles_deg = [math.degrees(a) for a in c_angles_rad]
    
    def refine_outer(c_coords, c_angles_rad):
        best_side = float('inf')
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
        
        for dx in np.linspace(-0.5, 0.5, 11):
            for dy in np.linspace(-0.5, 0.5, 11):
                for da_deg in np.linspace(-15, 15, 31):
                    test_ocx = ocx + dx
                    test_ocy = ocy + dy
                    test_otheta = otheta + math.radians(da_deg)
                    side = compute_outer_side_vectorized(c_coords, c_angles_rad, test_ocx, test_ocy, test_otheta)
                    if side < best_side:
                        best_side = side
                        best_ocx = test_ocx
                        best_ocy = test_ocy
                        best_otheta = test_otheta
                        
        return best_ocx, best_ocy, best_otheta, best_side

    ocx, ocy, otheta, final_side = refine_outer(c_coords, c_angles_rad)
    
    inner = [[c_coords[i, 0], c_coords[i, 1], c_angles_deg[i]] for i in range(12)]
    outer_center = (ocx, ocy)
    outer_side = final_side
    outer_angle = math.degrees(otheta)
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
