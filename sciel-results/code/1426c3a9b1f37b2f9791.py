<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    base = _honeycomb_init()
    rng = np.random.default_rng(42)
    
    def eval_config(x):
        """Evaluate configuration: x is [x0,y0,a0, x1,y1,a1, ..., xN-1,yN-1,aN-1, side, outer_angle]"""
        pos = x[:3*N].reshape(N, 3)[:, :2]
        ang = x[:3*N].reshape(N, 3)[:, 2]
        side = x[-2]
        oang = x[-1]
        
        outer = _hex_verts(0.0, 0.0, oang) * side
        
        # Check containment
        contain_pen = 0.0
        for i in range(N):
            P = _hex_verts(pos[i, 0], pos[i, 1], ang[i])
            d = _containment_dist(P, outer)
            if d < 0:
                contain_pen += d * d  # Squared penalty for violation
        
        # Check overlaps
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                Pi = _hex_verts(pos[i, 0], pos[i, 1], ang[i])
                Pj = _hex_verts(pos[j, 0], pos[j, 1], ang[j])
                d = _min_dist_convex(Pi, Pj)
                if d < 0:  # Overlapping
                    overlap_pen += d * d
        
        return side + 1000.0 * (contain_pen + overlap_pen)
    
    def is_valid(x):
        pos = x[:3*N].reshape(N, 3)[:, :2]
        ang = x[:3*N].reshape(N, 3)[:, 2]
        side = x[-2]
        oang = x[-1]
        
        outer = _hex_verts(0.0, 0.0, oang) * side
        
        for i in range(N):
            P = _hex_verts(pos[i, 0], pos[i, 1], ang[i])
            if _containment_dist(P, outer) < -1e-6:
                return False
        
        for i in range(N):
            for j in range(i + 1, N):
                Pi = _hex_verts(pos[i, 0], pos[i, 1], ang[i])
                Pj = _hex_verts(pos[j, 0], pos[j, 1], ang[j])
                if _min_dist_convex(Pi, Pj) < -1e-6:
                    return False
        
        return True
    
    # Compute initial side length needed
    def compute_min_side(pos, ang, oang=0.0):
        """Find minimum side length to contain all hexagons."""
        outer_base = _hex_verts(0.0, 0.0, oang)
        best_side = 0.0
        for i in range(N):
            P = _hex_verts(pos[i, 0], pos[i, 1], ang[i])
            # For each edge of outer hex, find required scale
            for k in range(6):
                v1 = outer_base[k]; v2 = outer_base[(k + 1) % 6]
                n = np.array([-(v2[1] - v1[1]), v2[0] - v1[0]])
                nn = np.linalg.norm(n)
                if nn > _TOL:
                    n = n / nn
                else:
                    continue
                # Ensure n points inward
                if np.dot(n, v1) > 0:
                    n = -n
                for p in P:
                    d = np.dot(p - v1, n)
                    # We need: dot(p - s*v1, n) >= 0 => dot(p, n) - s*dot(v1, n) >= 0
                    # => s <= dot(p, n) / dot(v1, n) if dot(v1, n) > 0
                    # But we want the tightest constraint
                    # Actually: p must satisfy dot(p - s*v1, n) >= 0
                    # dot(p, n) >= s * dot(v1, n)
                    # If dot(v1, n) > 0: s <= dot(p, n) / dot(v1, n)
                    # If dot(v1, n) < 0: s >= dot(p, n) / dot(v1, n) (but this is always true for large s)
                    # We want max s that works for all p, k
                    # Actually we want min s that contains all. Let me reconsider.
                    # The half-plane is: dot(x - s*v1, n) >= 0
                    # => dot(x, n) >= s * dot(v1, n)
                    # For this to hold: if dot(v1, n) > 0, then s <= dot(x, n) / dot(v1, n)
                    # We need s to satisfy ALL constraints, so s <= min over all (x,k) of dot(x,n)/dot(v1,n) where dot(v1,n)>0
                    # But also s must be positive.
                    # Hmm, this is getting complex. Let me just do binary search.
                    pass
        # Binary search for minimum side
        lo, hi = 0.5, 10.0
        for _ in range(50):
            mid = (lo + hi) / 2
            outer = _hex_verts(0.0, 0.0, oang) * mid
            ok = True
            for i in range(N):
                P = _hex_verts(pos[i, 0], pos[i, 1], ang[i])
                if _containment_dist(P, outer) < -1e-9:
                    ok = False
                    break
            if ok:
                hi = mid
            else:
                lo = mid
        return hi
    
    # Initial configuration
    x0 = np.zeros(3 * N + 2)
    for i in range(N):
        x0[3*i] = base[i, 0]
        x0[3*i+1] = base[i, 1]
        x0[3*i+2] = 0.0
    x0[-2] = compute_min_side(base, np.zeros(N))
    x0[-1] = 0.0
    
    best_x = x0.copy()
    best_side = x0[-2]
    best_score = eval_config(best_x)
    
    # Simulated annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.995
    step_size = 0.5
    
    n_iter = 0
    max_iter = 50000
    
    while time.time() < deadline - 10.0 and n_iter < max_iter:
        n_iter += 1
        x = best_x.copy()
        
        # Random perturbation
        dim = np.random.randint(3 * N)  # Perturb position or angle
        if dim % 3 == 2:
            # Angle perturbation
            x[dim] += rng.normal(0, step_size * 10)
            x[dim] = ((x[dim] + 180) % 360) - 180
        else:
            # Position perturbation
            x[dim] += rng.normal(0, step_size)
        
        # Occasionally perturb the outer angle
        if rng.random() < 0.05:
            x[-1] += rng.normal(0, 5)
            x[-1] = ((x[-1] + 180) % 360) - 180
        
        new_score = eval_config(x)
        delta = new_score - best_score
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            best_x = x
            best_score = new_score
            if is_valid(best_x):
                # Try to reduce the side length
                pos = best_x[:3*N].reshape(N, 3)[:, :2]
                ang = best_x[:3*N].reshape(N, 3)[:, 2]
                oang = best_x[-1]
                new_side = compute_min_side(pos, ang, oang)
                if new_side < best_x[-2]:
                    best_x[-2] = new_side
                    best_score = eval_config(best_x)
                if best_x[-2] < best_side:
                    best_side = best_x[-2]
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart with larger steps occasionally
            if n_iter % 5000 == 0:
                step_size = max(0.1, step_size * 0.8)
    
    # Final refinement: Nelder-Mead on the best valid configuration
    if is_valid(best_x):
        from scipy.optimize import minimize
        
        def obj(y):
            return y[-2]
        
        def obj_grad(y):
            g = np.zeros_like(y)
            g[-2] = 1.0
            return g
        
        # Just minimize the side with the current positions fixed
        # Actually, let's do a local search on all variables
        try:
            res = minimize(
                lambda y: eval_config(y),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True}
            )
            if is_valid(res.x) and res.x[-2] < best_side:
                best_x = res.x
                best_side = res.x[-2]
        except:
            pass
    
    # Ensure we have a valid result
    if not is_valid(best_x):
        # Fall back to a known valid configuration
        pos = base
        ang = np.zeros(N)
        side = compute_min_side(pos, ang)
        inner = [[float(pos[i, 0]), float(pos[i, 1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), float(side), 0.0
    
    pos = best_x[:3*N].reshape(N, 3)[:, :2]
    ang = best_x[:3*N].reshape(N, 3)[:, 2]
    side = best_x[-2]
    oang = best_x[-1]
    
    inner = [[float(pos[i, 0]), float(pos[i, 1]), float(ang[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(side), float(oang)
=======
def construct():
    global _START_TIME
    _START_TIME = time.time()
    deadline = _START_TIME + 110.0  # Safety margin: 120s limit, 10s buffer
    
    base = _honeycomb_init()
    rng = np.random.default_rng(42)
    
    def eval_config(x):
        """Evaluate configuration: x is [x0,y0,a0, x1,y1,a1, ..., xN-1,yN-1,aN-1, side, outer_angle]"""
        if time.time() > deadline - 5.0:
            return float('inf')
        
        pos = x[:3*N].reshape(N, 3)[:, :2]
        ang = x[:3*N].reshape(N, 3)[:, 2]
        side = x[-2]
        oang = x[-1]
        
        outer = _hex_verts(0.0, 0.0, oang) * side
        
        # Check containment
        contain_pen = 0.0
        for i in range(N):
            P = _hex_verts(pos[i, 0], pos[i, 1], ang[i])
            d = _containment_dist(P, outer)
            if d < 0:
                contain_pen += d * d  # Squared penalty for violation
        
        # Check overlaps
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                Pi = _hex_verts(pos[i, 0], pos[i, 1], ang[i])
                Pj = _hex_verts(pos[j, 0], pos[j, 1], ang[j])
                d = _min_dist_convex(Pi, Pj)
                if d < 0:  # Overlapping
                    overlap_pen += d * d
        
        return side + 1000.0 * (contain_pen + overlap_pen)
    
    def is_valid(x):
        if time.time() > deadline - 5.0:
            return False
        pos = x[:3*N].reshape(N, 3)[:, :2]
        ang = x[:3*N].reshape(N, 3)[:, 2]
        side = x[-2]
        oang = x[-1]
        
        outer = _hex_verts(0.0, 0.0, oang) * side
        
        for i in range(N):
            P = _hex_verts(pos[i, 0], pos[i, 1], ang[i])
            if _containment_dist(P, outer) < -1e-6:
                return False
        
        for i in range(N):
            for j in range(i + 1, N):
                Pi = _hex_verts(pos[i, 0], pos[i, 1], ang[i])
                Pj = _hex_verts(pos[j, 0], pos[j, 1], ang[j])
                if _min_dist_convex(Pi, Pj) < -1e-6:
                    return False
        
        return True
    
    # Compute initial side length needed
    def compute_min_side(pos, ang, oang=0.0):
        """Find minimum side length to contain all hexagons."""
        if time.time() > deadline - 5.0:
            return 10.0
        # Binary search for minimum side
        lo, hi = 0.5, 10.0
        for _ in range(30):  # Reduced from 50
            mid = (lo + hi) / 2
            outer = _hex_verts(0.0, 0.0, oang) * mid
            ok = True
            for i in range(N):
                P = _hex_verts(pos[i, 0], pos[i, 1], ang[i])
                if _containment_dist(P, outer) < -1e-9:
                    ok = False
                    break
            if ok:
                hi = mid
            else:
                lo = mid
        return hi
    
    # Initial configuration
    x0 = np.zeros(3 * N + 2)
    for i in range(N):
        x0[3*i] = base[i, 0]
        x0[3*i+1] = base[i, 1]
        x0[3*i+2] = 0.0
    x0[-2] = compute_min_side(base, np.zeros(N))
    x0[-1] = 0.0
    
    best_x = x0.copy()
    best_side = x0[-2]
    best_score = eval_config(best_x)
    
    # Simulated annealing with time-based termination
    T = 1.0
    T_min = 0.001
    cooling = 0.999  # Slower cooling
    step_size = 0.5
    
    n_iter = 0
    max_iter = 5000  # Reduced from 50000
    
    while time.time() < deadline - 15.0 and n_iter < max_iter:
        n_iter += 1
        x = best_x.copy()
        
        # Random perturbation
        dim = np.random.randint(3 * N)  # Perturb position or angle
        if dim % 3 == 2:
            # Angle perturbation
            x[dim] += rng.normal(0, step_size * 10)
            x[dim] = ((x[dim] + 180) % 360) - 180
        else:
            # Position perturbation
            x[dim] += rng.normal(0, step_size)
        
        # Occasionally perturb the outer angle
        if rng.random() < 0.05:
            x[-1] += rng.normal(0, 5)
            x[-1] = ((x[-1] + 180) % 360) - 180
        
        new_score = eval_config(x)
        if new_score == float('inf'):
            break
            
        delta = new_score - best_score
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            best_x = x
            best_score = new_score
            if is_valid(best_x):
                # Try to reduce the side length
                pos = best_x[:3*N].reshape(N, 3)[:, :2]
                ang = best_x[:3*N].reshape(N, 3)[:, 2]
                oang = best_x[-1]
                new_side = compute_min_side(pos, ang, oang)
                if new_side < best_x[-2]:
                    best_x[-2] = new_side
                    best_score = eval_config(best_x)
                if best_x[-2] < best_side:
                    best_side = best_x[-2]
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart with larger steps occasionally
            if n_iter % 2000 == 0:
                step_size = max(0.1, step_size * 0.8)
    
    # Ensure we have a valid result
    if not is_valid(best_x):
        # Fall back to a known valid configuration
        pos = base
        ang = np.zeros(N)
        side = compute_min_side(pos, ang)
        inner = [[float(pos[i, 0]), float(pos[i, 1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), float(side), 0.0
    
    pos = best_x[:3*N].reshape(N, 3)[:, :2]
    ang = best_x[:3*N].reshape(N, 3)[:, 2]
    side = best_x[-2]
    oang = best_x[-1]
    
    inner = [[float(pos[i, 0]), float(pos[i, 1]), float(ang[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(side), float(oang)
>>>>>>> REPLACE
