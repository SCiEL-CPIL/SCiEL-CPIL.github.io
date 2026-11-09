# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    sqrt3 = math.sqrt(3)
    inradius = sqrt3 / 2.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        return [(cx + side * math.cos(angle_rad + k * math.pi / 3),
                 cy + side * math.sin(angle_rad + k * math.pi / 3)) for k in range(6)]

    def compute_min_side(inner_positions, outer_angle):
        max_side = 0.0
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        normals = [(math.cos(math.radians(30 + k * 60)), math.sin(math.radians(30 + k * 60))) for k in range(6)]

        for ix, iy, iang in inner_positions:
            iv = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in iv:
                rx = vx * cos_a - vy * sin_a
                ry = vx * sin_a + vy * cos_a
                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    req = proj / inradius
                    if req > max_side:
                        max_side = req
        return max_side

    def sat_overlap(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d = math.sqrt(dx*dx + dy*dy)
        if d >= 2.0:
            return 0.0
        if d < 1e-9:
            return 2.0

        v1 = hexagon_vertices(p1[0], p1[1], 1.0, p1[2])
        v2 = hexagon_vertices(p2[0], p2[1], 1.0, p2[2])

        axes = []
        for verts in (v1, v2):
            for i in range(6):
                x1, y1 = verts[i]
                x2, y2 = verts[(i+1) % 6]
                ex, ey = x2-x1, y2-y1
                L = math.sqrt(ex*ex + ey*ey)
                if L > 1e-12:
                    axes.append((-ey/L, ex/L))
        axes.append((dx/d, dy/d))

        min_gap = float('inf')
        for nx, ny in axes:
            p1min = min(vx*nx+vy*ny for vx,vy in v1)
            p1max = max(vx*nx+vy*ny for vx,vy in v1)
            p2min = min(vx*nx+vy*ny for vx,vy in v2)
            p2max = max(vx*nx+vy*ny for vx,vy in v2)
            gap = max(p2min - p1max, p1min - p2max)
            if gap > 0:
                return 0.0
            min_gap = min(min_gap, -gap)
        return min_gap if min_gap < float('inf') else 0.0

    def total_overlap_penalty(positions):
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                pen += sat_overlap(positions[i], positions[j])
        return pen

    def objective(params, outer_angle=0.0):
        positions = []
        for i in range(N):
            positions.append([params[3*i], params[3*i+1], params[3*i+2]])
        side = compute_min_side(positions, outer_angle)
        pen = total_overlap_penalty(positions)
        return side + 1000.0 * pen * pen

    def make_honeycomb():
        positions = []
        positions.append([0.0, 0.0, 0.0])
        ring_dist = sqrt3
        for k in range(6):
            a = math.radians(60*k)
            positions.append([ring_dist*math.cos(a), ring_dist*math.sin(a), 0.0])
        # Place 4 outer hexagons in the gaps
        # Gaps are at 30, 90, 150, 210, 270, 330
        # We pick 4 of them, e.g., 30, 150, 210, 330 (symmetric)
        # Distance for touching:
        # In hex lattice, distance to (1,1) axial is sqrt(3)*sqrt(3) = 3? No.
        # Axial (1,1) is distance sqrt(3)*sqrt(1+1+1) = 3.
        # So r = 3.0 is the lattice distance.
        # But we can push them in if we rotate?
        # Let's start with r = 3.0
        for ad in [30, 150, 210, 330]:
            a = math.radians(ad)
            positions.append([3.0*math.cos(a), 3.0*math.sin(a), 0.0])
        return positions

    def to_params(positions):
        params = []
        for p in positions:
            params.extend([p[0], p[1], p[2]])
        return np.array(params)

    def from_params(params):
        positions = []
        for i in range(N):
            positions.append([params[3*i], params[3*i+1], params[3*i+2]])
        return positions

    def nelder_mead_optimize(init_positions, max_iter=2000, outer_angle=0.0):
        try:
            from scipy.optimize import minimize
            params = to_params(init_positions)
            res = minimize(objective, params, method='Nelder-Mead',
                          args=(outer_angle,),
                          options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-7,
                                   'adaptive': True})
            return from_params(res.x), res.fun
        except Exception:
            return init_positions, objective(to_params(init_positions), outer_angle)

    def differential_evolution_optimize(bounds, max_iter=100, popsize=20, outer_angle=0.0):
        try:
            from scipy.optimize import differential_evolution
            res = differential_evolution(objective, bounds, args=(outer_angle,),
                                         maxiter=max_iter, popsize=popsize,
                                         tol=1e-4, seed=42, polish=True)
            return from_params(res.x), res.fun
        except Exception:
            return None, float('inf')

    # Strategy:
    # 1. Generate honeycomb.
    # 2. Use DE to find global optimum.
    # 3. Refine with NM.
    
    best_positions = None
    best_obj = float('inf')
    
    # Bounds for DE: x, y in [-5, 5], angle in [-30, 30]
    bounds = []
    for _ in range(N):
        bounds.append((-5, 5))
        bounds.append((-5, 5))
        bounds.append((-30, 30))
        
    # Run DE
    if time.time() < deadline - 30:
        de_pos, de_obj = differential_evolution_optimize(bounds, max_iter=50, popsize=15, outer_angle=0.0)
        if de_pos is not None and de_obj < best_obj:
            best_obj = de_obj
            best_positions = de_pos
            
    # Refine with NM from DE result or Honeycomb
    inits = [make_honeycomb()]
    if best_positions:
        inits.append(best_positions)
        
    for init in inits:
        if time.time() > deadline - 20:
            break
        pos, obj = nelder_mead_optimize(init, max_iter=1000, outer_angle=0.0)
        if obj < best_obj:
            best_obj = obj
            best_positions = pos
            
    # Optimize outer angle
    if best_positions is None:
        best_positions = make_honeycomb()
        
    positions = best_positions
    best_side = float('inf')
    best_angle = 0.0
    for ang in range(0, 31, 1):
        s = compute_min_side(positions, float(ang))
        if s < best_side:
            best_side = s
            best_angle = float(ang)
            
    # Final check
    pen = total_overlap_penalty(positions)
    if pen > 1e-4:
        # Fallback
        positions = make_honeycomb()
        best_side = float('inf')
        best_angle = 0.0
        for ang in range(0, 31, 2):
            s = compute_min_side(positions, float(ang))
            if s < best_side:
                best_side = s
                best_angle = float(ang)
                
    return positions, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
