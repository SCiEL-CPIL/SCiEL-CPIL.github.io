import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 142.0

tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)
N_TRI = len(tri_indices)

H = np.sqrt(3) / 2

def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    ui = u[tri_idx[:, 0]]; vi = v[tri_idx[:, 0]]; wi = w[tri_idx[:, 0]]
    uj = u[tri_idx[:, 1]]; vj = v[tri_idx[:, 1]]; wj = w[tri_idx[:, 1]]
    uk = u[tri_idx[:, 2]]; vk = v[tri_idx[:, 2]]; wk = w[tri_idx[:, 2]]
    det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    return np.abs(det)

def clamp_bary(bary, eps=1e-10):
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    s = u + v
    bad = s > 1.0 - eps
    if np.any(bad):
        sc = (1.0 - eps) / np.maximum(s, eps)
        u[bad] *= sc[bad]
        v[bad] *= sc[bad]
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    return np.column_stack([u, v])

def hard_min_value(bary):
    areas = calc_areas_bary(bary)
    return float(np.min(areas))

def epigraph_obj(x):
    """x = [u0,v0,...,u10,v10,t]. Maximize t s.t. all areas >= t."""
    bary = x[:2*N].reshape(N, 2)
    t = x[2*N]
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    # Penalty for violating constraints
    viol = np.maximum(0, t - areas)
    penalty = np.sum(viol**2)
    # Also penalize if t is negative
    t_pen = max(0, -t)**2
    return penalty + t_pen - t

def epigraph_obj_grad(x):
    """Gradient of epigraph objective for L-BFGS-B."""
    bary = x[:2*N].reshape(N, 2).copy()
    t = x[2*N]
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    viol = np.maximum(0, t - areas)
    t_pen = max(0, -t)**2
    
    # Gradient w.r.t. t: -1 + 2*sum(viol)
    grad_t = -1.0 + 2.0 * np.sum(viol)
    if t < 0:
        grad_t += 2.0 * (-t)
    
    # Gradient w.r.t. bary: need d(area_k)/d(bary) for violating triangles
    grad_bary = np.zeros((N, 2))
    if np.any(viol > 0):
        # For each violating triangle, compute gradient of area w.r.t. the 3 points
        u = bary[:, 0]
        v = bary[:, 1]
        w = 1.0 - u - v
        
        for k in range(N_TRI):
            if viol[k] <= 0:
                continue
            i, j, kk = tri_idx[k]
            # area = |det|, for interior points det > 0 typically
            # d(area)/d(u_i) = v_j*w_k - v_k*w_j
            # d(area)/d(v_i) = u_k*w_j - u_j*w_k
            # d(area)/d(w_i) = u_j*v_k - u_k*v_j
            # But w_i = 1 - u_i - v_i, so d/d(u_i) includes -1 from w_i
            # Full: area = u_i(v_j w_k - v_k w_j) - v_i(u_j w_k - u_k w_j) + w_i(u_j v_k - u_k v_j)
            # d(area)/du_i = (v_j w_k - v_k w_j) - (u_j w_k - u_k w_j)  [from -v_i term: -1 * ...]
            # Wait, let me redo. w_i = 1 - u_i - v_i
            # area = u_i*A - v_i*B + (1-u_i-v_i)*C where A=v_j w_k - v_k w_j, B=u_j w_k - u_k w_j, C=u_j v_k - u_k v_j
            # d(area)/du_i = A - C
            # d(area)/dv_i = -B - C
            A = v[j]*w[kk] - v[kk]*w[j]
            B = u[j]*w[kk] - u[kk]*w[j]
            C = u[j]*v[kk] - u[kk]*v[j]
            
            coeff = 2.0 * viol[k]
            grad_bary[i, 0] += coeff * (A - C)
            grad_bary[i, 1] += coeff * (-B - C)
            
            # Similarly for j and kk
            A2 = v[i]*w[kk] - v[kk]*w[i]
            B2 = u[i]*w[kk] - u[kk]*w[i]
            C2 = u[i]*v[kk] - u[kk]*v[i]
            grad_bary[j, 0] += coeff * (A2 - C2)
            grad_bary[j, 1] += coeff * (-B2 - C2)
            
            A3 = v[i]*w[j] - v[j]*w[i]
            B3 = u[i]*w[j] - u[j]*w[i]
            C3 = u[i]*v[j] - u[j]*v[i]
            grad_bary[kk, 0] += coeff * (A3 - C3)
            grad_bary[kk, 1] += coeff * (-B3 - C3)
    
    grad = np.zeros(2*N + 1)
    grad[:2*N] = grad_bary.flatten()
    grad[2*N] = grad_t
    return grad

def make_symmetric_init():
    """Generate an init with approximate 3-fold symmetry."""
    # 11 points: 1 center + 10 in two rings of 5? Or 3+3+3+1+1?
    # Try: 1 center, 3 in inner ring, 3 in middle ring, 3 in outer ring, 1 extra
    # Actually for 11: center(1) + ring1(3) + ring2(3) + ring3(3) + 1 = 11
    
    center = np.array([1/3, 1/3])
    
    # Ring at radius r from center, 3 points at 120 degrees
    def ring3(r, angle_offset=0):
        pts = []
        for k in range(3):
            ang = angle_offset + 2*np.pi*k/3
            # In barycentric, center is (1/3,1/3)
            # Moving in direction (cos(ang), sin(ang)) in Cartesian
            # But easier: offset in barycentric
            # Actually let's just place them at specific barycentric coords
            pass
        return pts
    
    # Simpler: use the K=3 lattice (10 pts) + 1 center offset
    # K=3 lattice: (i/3, j/3) for i+j<=3
    pts = []
    for i in range(4):
        for j in range(4-i):
            pts.append([i/3.0, j/3.0])
    # 10 points. Add 11th at center (1/3,1/3) - but that's already there (i=1,j=1)
    # So offset the center slightly
    pts.append([1/3.0 + 0.05, 1/3.0 - 0.03])
    return np.array(pts)

def make_hex_init():
    """Hexagonal packing init."""
    # Try to fit 11 points in hexagonal pattern
    # Row 0 (w=1): 1 point at center
    # Row 1 (w=2/3): 2 points
    # Row 2 (w=1/3): 3 points  
    # Row 3 (w=0): 5 points? That's 11
    
    # Actually let's use a proper hex grid
    s = 0.28  # spacing
    pts = []
    for row in range(5):
        w_val = 1.0 - row * s
        if w_val < 0:
            break
        max_j = int((1 - w_val) / s) + 1
        for j in range(max_j):
            u_val = j * s
            v_val = 1 - w_val - u_val
            if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                pts.append([u_val, v_val])
    
    if len(pts) >= N:
        rng = np.random.default_rng(7)
        idx = rng.choice(len(pts), N, replace=False)
        return np.array(pts)[idx]
    return make_symmetric_init()

def make_lattice_init():
    """K=3 lattice with perturbed center."""
    pts = []
    for i in range(4):
        for j in range(4-i):
            pts.append([i/3.0, j/3.0])
    # 10 points, perturb the center (1/3,1/3) which is index 4 (i=1,j=1)
    # Actually indices: (0,0)=0, (1,0)=1, (2,0)=2, (3,0)=3, (0,1)=4, (1,1)=5, (2,1)=6, (0,2)=7, (1,2)=8, (0,3)=9
    # Center is (1,1) = index 5
    pts[5] = [1/3.0 + 0.04, 1/3.0 + 0.02]
    return np.array(pts)

def make_random_init(seed):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def refine_epigraph(bary_init, t_init=None, time_limit=10.0):
    """Refine using epigraph formulation with SLSQP."""
    start = time.time()
    bary = clamp_bary(bary_init)
    if t_init is None:
        t_init = hard_min_value(bary) * 0.9
    
    x0 = np.concatenate([bary.flatten(), [t_init]])
    
    bounds = [(0, 1)] * (2*N) + [(0, 0.05)]
    
    best_bary = bary.copy()
    best_t = t_init
    
    # Stage 1: SLSQP with epigraph
    try:
        res = minimize(
            epigraph_obj,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-15, 'iprint': 0}
        )
        if res.x is not None:
            b = clamp_bary(res.x[:2*N].reshape(N, 2))
            t = res.x[2*N]
            val = hard_min_value(b)
            if val > best_t:
                best_t = val
                best_bary = b.copy()
    except Exception:
        pass
    
    if time.time() - start > time_limit * 0.5:
        return best_bary, best_t
    
    # Stage 2: Coordinate descent on epigraph
    # For each point, optimize its (u,v) given others fixed
    for iteration in range(3):
        if time.time() - start > time_limit * 0.85:
            break
        improved = False
        for i in range(N):
            # Fix all other points, optimize point i
            others = np.delete(best_bary, i, axis=0)
            
            def sub_obj(x_i):
                bary_sub = np.vstack([best_bary[:i], x_i.reshape(1,2), best_bary[i+1:]])
                bary_sub = clamp_bary(bary_sub)
                areas = calc_areas_bary(bary_sub)
                return -np.min(areas)
            
            x0_i = best_bary[i]
            try:
                res_i = minimize(
                    sub_obj,
                    x0_i,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                new_pt = clamp_bary(res_i.x.reshape(1, 2))[0]
                bary_new = np.vstack([best_bary[:i], new_pt.reshape(1,2), best_bary[i+1:]])
                bary_new = clamp_bary(bary_new)
                val_new = hard_min_value(bary_new)
                if val_new > best_t + 1e-15:
                    best_bary = bary_new
                    best_t = val_new
                    improved = True
            except Exception:
                pass
        
        if not improved:
            break
    
    # Stage 3: Final SLSQP polish
    if time.time() - start < time_limit - 2:
        try:
            x0 = np.concatenate([best_bary.flatten(), [best_t * 0.95]])
            res = minimize(
                epigraph_obj,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-16, 'iprint': 0}
            )
            if res.x is not None:
                b = clamp_bary(res.x[:2*N].reshape(N, 2))
                val = hard_min_value(b)
                if val > best_t:
                    best_t = val
                    best_bary = b.copy()
        except Exception:
            pass
    
    return best_bary, best_t

def refine_hard_min(bary_init, time_limit=8.0):
    """Refine using direct hard-min minimization with multiple methods."""
    start = time.time()
    bary = clamp_bary(bary_init)
    best_val = hard_min_value(bary)
    
    def obj(x):
        b = clamp_bary(x.reshape(N, 2))
        return -hard_min_value(b)
    
    bounds = [(0, 1)] * (2*N)
    
    # Nelder-Mead
    try:
        res = minimize(obj, bary.flatten(), method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-15, 'fatol': 1e-18, 'adaptive': True})
        b = clamp_bary(res.x.reshape(N, 2))
        val = hard_min_value(b)
        if val > best_val:
            best_val = val
            bary = b
    except Exception:
        pass
    
    if time.time() - start > time_limit * 0.6:
        return bary, best_val
    
    # SLSQP
    try:
        res = minimize(obj, bary.flatten(), method='SLSQP', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-16, 'iprint': 0})
        b = clamp_bary(res.x.reshape(N, 2))
        val = hard_min_value(b)
        if val > best_val:
            best_val = val
            bary = b
    except Exception:
        pass
    
    return bary, best_val

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = hard_min_value(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
    
    # Generate diverse initializations
    inits = []
    inits.append(make_symmetric_init())
    inits.append(make_hex_init())
    inits.append(make_lattice_init())
    inits.append(make_random_init(0))
    inits.append(make_random_init(1))
    
    # Refine each init with epigraph method
    for i, init in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 15:
            break
        tl = min(12.0, remaining * 0.25)
        try:
            b, v = refine_epigraph(init, time_limit=tl)
            update(b)
        except Exception:
            pass
    
    # Additional random restarts with epigraph refinement
    rng = np.random.default_rng(99)
    n_restarts = 0
    while time.time() < deadline - 20:
        n_restarts += 1
        if best_bary is not None and n_restarts % 2 == 0:
            # Perturb best
            noise = rng.normal(0, 0.06, best_bary.shape)
            init = clamp_bary(best_bary + noise)
        else:
            init = make_random_init(1000 + n_restarts)
        
        remaining = deadline - time.time()
        tl = min(8.0, remaining * 0.3)
        if tl < 3:
            break
        try:
            b, v = refine_epigraph(init, time_limit=tl)
            update(b)
        except Exception:
            pass
    
    # Final polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            b, v = refine_hard_min(best_bary, time_limit=5.0)
            update(b)
        except Exception:
            pass
    
    if best_bary is None:
        best_bary = make_symmetric_init()
    
    return bary_to_cart(best_bary)
