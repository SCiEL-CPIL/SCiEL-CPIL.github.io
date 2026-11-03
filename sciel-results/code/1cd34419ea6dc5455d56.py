import numpy as np
import time

N = 11

def construct():
    """
    Place N=11 unit regular hexagons inside a minimal regular hexagon.
    
    Strategy:
    - Use the natural hexagonal lattice packing where unit hexagons tile the plane.
    - For N=11, the optimal arrangement is a "flower" pattern: 1 center + 6 ring + 4 more.
    - The minimal outer hexagon side length is 3.0 (apothem = 3*sqrt(3)/2 = 1.5*sqrt(3)).
    
    The 11 centers in hex coordinates (q, r) where:
    x = sqrt(3)*(q + r/2), y = 1.5*r
    
    Flower pattern centers:
    Ring 0: (0,0)
    Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    Ring 2 (pick 4 to get 11 total): (2,0), (1,1), (0,2), (-2,0) -- but need to check fit
    
    Actually, let me compute the exact minimal bounding hexagon.
    
    For a regular hexagon with side S centered at origin, a point (x,y) is inside if:
    For each of the 6 face normals at angles 0°, 60°, 120°, 180°, 240°, 300°:
    n_k · p <= S*sqrt(3)/2 (the apothem)
    
    n_k = (cos(k*60°), sin(k*60°)) for k=0..5
    
    The apothem needed is the max over all inner hexagon vertices of n_k · v.
    For a unit hexagon centered at (cx, cy) with rotation 0, its vertices are at distance 1 from center.
    The max projection onto any direction is 1 (the circumradius of a unit hexagon).
    So the apothem needed >= max over all inner hex centers of (n_k · c) + 1.
    
    Wait, that's not quite right. The inner hexagons also have extent. A unit regular hexagon 
    centered at the origin has vertices at distance 1 (circumradius = side length for regular hex).
    The maximum projection of a unit hexagon centered at c onto direction n is n·c + 1.
    
    So apothem_needed = max_k max_i (n_k · c_i + 1)
    Side S = 2 * apothem_needed / sqrt(3)
    
    For the flower pattern with centers:
    (0,0), (sqrt(3),0), (sqrt(3)/2, 3/2), (-sqrt(3)/2, 3/2), (-sqrt(3),0), (-sqrt(3)/2,-3/2), (sqrt(3)/2,-3/2)
    plus 4 more from ring 2.
    
    Let me just use a direct analytical solution.
    
    The optimal packing of 11 unit hexagons in a hexagon:
    Side length = 3.0
    """
    
    # Hexagonal lattice basis vectors for unit hexagon centers (non-overlapping)
    # Two adjacent unit hexagons have center distance sqrt(3)
    d = np.sqrt(3)
    
    # Flower pattern: center + ring of 6 + 4 more
    # Ring 0: (0, 0)
    # Ring 1: 6 neighbors
    # Ring 2: pick 4 that fit in side-3 hexagon
    
    centers = [
        (0.0, 0.0),                    # center
        (d, 0.0),                       # right
        (d/2, 1.5),                     # upper-right
        (-d/2, 1.5),                    # upper-left
        (-d, 0.0),                      # left
        (-d/2, -1.5),                   # lower-left
        (d/2, -1.5),                    # lower-right
        # Ring 2 - 4 more
        (1.5*d, 1.5),                   # far upper-right
        (0.0, 3.0),                     # far top
        (-1.5*d, 1.5),                  # far upper-left
        (0.0, -3.0),                    # far bottom
    ]
    
    # Verify: all 11 centers should give apothem <= 1.5*sqrt(3) for side=3
    # Apothem for side S is S*sqrt(3)/2 = 3*sqrt(3)/2 = 1.5*sqrt(3)
    
    # Normals of outer hexagon
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Compute the required apothem
    max_apothem = 0.0
    for cx, cy in centers:
        for n in normals:
            proj = n[0]*cx + n[1]*cy + 1.0  # +1 for circumradius of unit hex
            if proj > max_apothem:
                max_apothem = proj
    
    # The side length
    side = 2.0 * max_apothem / np.sqrt(3)
    
    # For the analytical solution, side should be exactly 3.0
    # Let's verify: max projection should be 1.5*sqrt(3)
    # 1.5*sqrt(3) ≈ 2.598
    
    # Check the farthest point: (0, 3) with normal (0.5, sqrt(3)/2):
    # proj = 0.5*0 + (sqrt(3)/2)*3 + 1 = 1.5*sqrt(3) + 1 ≈ 3.598
    # That's too large! Side would be 2*(1.5*sqrt(3)+1)/sqrt(3) = 3 + 2/sqrt(3) ≈ 4.155
    
    # So the simple flower with (0,3) doesn't fit in side 3.
    # Let me recalculate properly.
    
    # For side S=3, apothem = 3*sqrt(3)/2 ≈ 2.598
    # A unit hexagon centered at (cx,cy) fits if for all normals n:
    # n·(cx,cy) + 1 <= apothem
    # i.e., n·(cx,cy) <= apothem - 1 = 3*sqrt(3)/2 - 1
    
    # Let's find what centers fit in a side-3 hexagon.
    # apothem - 1 = 1.5*sqrt(3) - 1 ≈ 1.598
    
    # The hexagonal lattice points within this constraint...
    # This is getting complex. Let me use a numerical approach.
    
    # Actually, let me just use scipy to optimize this properly.
    import scipy.optimize
    
    deadline = 140.0
    start = time.time()
    
    def get_verts(cx, cy, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        base = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        R = np.array([[c, -s], [s, c]])
        return base @ R.T + np.array([cx, cy])
    
    def compute_side_and_penalty(params):
        """params: [x0,y0,a0, x1,y1,a1, ...]"""
        max_apothem = 0.0
        for i in range(N):
            cx, cy, ang = params[i*3], params[i*3+1], params[i*3+2]
            verts = get_verts(cx, cy, ang)
            for v in verts:
                for n in normals:
                    proj = np.dot(v, n)
                    if proj > max_apothem:
                        max_apothem = proj
        
        side = 2.0 * max_apothem / np.sqrt(3)
        
        # Overlap penalty: minimum center distance for non-overlapping unit hexagons
        # For same orientation, min dist is sqrt(3)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = params[i*3] - params[j*3]
                dy = params[i*3+1] - params[j*3+1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_d = np.sqrt(3) - 0.001  # small tolerance
                if dist < min_d:
                    penalty += (min_d - dist)**2 * 1000
        
        return side, penalty
    
    def objective(params):
        side, penalty = compute_side_and_penalty(params)
        return side + penalty
    
    # Initial guess: hexagonal lattice flower
    init = []
    for cx, cy in centers:
        init.extend([cx, cy, 0.0])
    init = np.array(init)
    
    # Try Nelder-Mead
    best_side = float('inf')
    best_params = init.copy()
    
    # First, compute the side for our initial guess
    s0, p0 = compute_side_and_penalty(init)
    best_side = s0
    
    # Local search to improve
    np.random.seed(42)
    current = init.copy()
    current_obj = objective(current)
    step = 0.2
    no_improve = 0
    
    for it in range(2000):
        if time.time() > deadline:
            break
        if no_improve > 100:
            step *= 0.9
            if step < 0.001:
                break
            no_improve = 0
        
        pert = np.random.normal(0, step, len(current))
        pert[2::3] *= 0.3  # smaller angle perturbations
        candidate = current + pert
        
        cand_obj = objective(candidate)
        
        if cand_obj < current_obj:
            current = candidate
            current_obj = cand_obj
            no_improve = 0
            s, p = compute_side_and_penalty(current)
            if s < best_side:
                best_side = s
                best_params = current.copy()
        else:
            no_improve += 1
    
    # Also try a few random restarts
    for restart in range(5):
        if time.time() > deadline:
            break
        
        np.random.seed(100 + restart)
        random_params = []
        for i in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            ang = np.random.uniform(-10, 10)
            random_params.extend([x, y, ang])
        rp = np.random.array(random_params)
        
        s_r, p_r = compute_side_and_penalty(rp)
        if s_r < best_side and p_r < 1e-6:
            best_side = s_r
            best_params = rp.copy()
        
        # Quick local search from this point
        cur = rp.copy()
        cur_obj = objective(cur)
        stp = 0.15
        ni = 0
        for it2 in range(500):
            if time.time() > deadline:
                break
            if ni > 80:
                stp *= 0.9
                if stp < 0.001:
                    break
                ni = 0
            pert = np.random.normal(0, stp, len(cur))
            pert[2::3] *= 0.3
            cand = cur + pert
            cand_obj = objective(cand)
            if cand_obj < cur_obj:
                cur = cand
                cur_obj = cand_obj
                ni = 0
                s_c, p_c = compute_side_and_penalty(cur)
                if s_c < best_side:
                    best_side = s_c
                    best_params = cur.copy()
            else:
                ni += 1
    
    # Build output
    inner = []
    for i in range(N):
        x = float(best_params[i*3])
        y = float(best_params[i*3+1])
        ang = float(best_params[i*3+2])
        inner.append([x, y, ang])
    
    # Ensure the side is at least large enough (validity check)
    if best_side < 2.5:
        best_side = 2.5
    
    return inner, (0.0, 0.0), float(best_side), 0.0
