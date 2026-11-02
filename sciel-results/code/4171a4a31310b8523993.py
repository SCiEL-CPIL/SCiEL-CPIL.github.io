import numpy as np
from scipy.optimize import minimize, linprog
import time
import random

N = 21
IDX_I, IDX_J = np.triu_indices(N, k=1)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def check_time():
        return time.time() > deadline

    def get_rect_dims(t):
        t = float(np.clip(t, 0.15, 0.85))
        return 2.0 * t, 2.0 * (1.0 - t)

    def unpack(state):
        t = state[0]
        w, h = get_rect_dims(t)
        x = state[1:1 + N]
        y = state[1 + N:1 + 2 * N]
        # Note: We don't store radii in the main optimization state for Phase 1
        return w, h, x, y

    def is_valid(w, h, x, y, r, eps=1e-7):
        if np.any(r < 1e-8) or w <= 0 or h <= 0:
            return False
        if np.any(x - r < -eps) or np.any(x + r > w + eps) or \
           np.any(y - r < -eps) or np.any(y + r > h + eps):
            return False
        dx = x[IDX_I] - x[IDX_J]
        dy = y[IDX_I] - y[IDX_J]
        d2 = dx * dx + dy * dy
        md = r[IDX_I] + r[IDX_J]
        if np.any(d2 < (md - eps) ** 2):
            return False
        return True

    def sum_r(r):
        return float(np.sum(r))

    # ---------- Initial Configurations ----------
    def make_hex(t, jitter=0.0):
        w, h = get_rect_dims(t)
        r_est = np.sqrt(0.9069 * w * h / (N * np.pi))
        pos = []
        # Try to fit a hex grid
        for scale in [1.0, 0.95, 0.9, 0.85]:
            r = r_est * scale
            y = r
            row = 0
            pos = []
            while y <= h - r + 1e-9 and len(pos) < N:
                off = r if row % 2 else 0.0
                x = r + off
                while x <= w - r + 1e-9 and len(pos) < N:
                    pos.append((x, y))
                    x += 2.0 * r
                y += np.sqrt(3.0) * r
                row += 1
            if len(pos) >= N:
                break
        
        if len(pos) < N:
            # Fallback to grid
            side = int(np.ceil(np.sqrt(N)))
            for i in range(side):
                for j in range(side):
                    if len(pos) >= N: break
                    pos.append(((i + 0.5) / side * w, (j + 0.5) / side * h))
                    if len(pos) >= N: break

        state = np.zeros(1 + 2 * N) # t, x, y
        state[0] = t
        for i in range(N):
            px, py = pos[i]
            if jitter > 0:
                px += random.uniform(-jitter, jitter)
                py += random.uniform(-jitter, jitter)
            state[1 + i] = np.clip(px, 0.1, w - 0.1)
            state[1 + N + i] = np.clip(py, 0.1, h - 0.1)
        return state

    def make_grid(t, rows, cols, jitter=0.0):
        w, h = get_rect_dims(t)
        state = np.zeros(1 + 2 * N)
        state[0] = t
        for i in range(N):
            rr = i // cols
            cc = i % cols
            px = (cc + 0.5) / cols * w
            py = (rr + 0.5) / rows * h
            if jitter > 0:
                px += random.uniform(-jitter, jitter)
                py += random.uniform(-jitter, jitter)
            state[1 + i] = np.clip(px, 0.1, w - 0.1)
            state[1 + N + i] = np.clip(py, 0.1, h - 0.1)
        return state

    # ---------- Phase 1: Maximize Minimum Distance (Geometry Optimization) ----------
    # Objective: Maximize min( d_ij, d_i_wall )
    # Equivalent to minimizing -log(sum(exp(-beta * dist))) / beta  (Soft Min)
    
    def optimize_geometry(v0, beta=100.0, maxiter=300):
        if check_time():
            return v0
        
        def obj(v):
            t = v[0]
            w, h = get_rect_dims(t)
            x = v[1:1 + N]
            y = v[1 + N:1 + 2 * N]
            
            # Inter-circle distances
            dx = x[IDX_I] - x[IDX_J]
            dy = y[IDX_I] - y[IDX_J]
            d_inter = np.sqrt(dx * dx + dy * dy + 1e-12)
            
            # Wall distances
            d_left = x
            d_right = w - x
            d_bottom = y
            d_top = h - y
            
            all_dists = np.concatenate([d_inter, d_left, d_right, d_bottom, d_top])
            
            # Soft minimum: -1/beta * log(sum(exp(-beta * d)))
            # We want to maximize this, so minimize negative.
            # To avoid overflow, use log-sum-exp trick
            m = np.min(all_dists)
            lse = m + np.log(np.sum(np.exp(-beta * (all_dists - m))))
            return -lse / beta # Minimize negative soft-min -> Maximize soft-min

        def grad(v):
            t = v[0]
            w, h = get_rect_dims(t)
            x = v[1:1 + N]
            y = v[1 + N:1 + 2 * N]
            
            dx = x[IDX_I] - x[IDX_J]
            dy = y[IDX_I] - y[IDX_J]
            d_inter = np.sqrt(dx * dx + dy * dy + 1e-12)
            
            d_left = x
            d_right = w - x
            d_bottom = y
            d_top = h - y
            
            all_dists = np.concatenate([d_inter, d_left, d_right, d_bottom, d_top])
            
            # Weights for the soft-min
            # w_k = exp(-beta * d_k) / sum(exp(-beta * d_j))
            m = np.min(all_dists)
            exp_terms = np.exp(-beta * (all_dists - m))
            sum_exp = np.sum(exp_terms)
            weights = exp_terms / sum_exp
            
            g = np.zeros_like(v)
            
            # Gradient w.r.t x, y for inter-circle distances
            # d(d_ij)/dx_i = (x_i - x_j)/d_ij
            # d(obj)/dx = -1/beta * d(obj_raw)/dx
            # obj_raw = -lse. grad(lse) = sum(w_k * grad(d_k))
            # So grad(obj) = -1/beta * sum(w_k * grad(d_k))
            
            # Inter-circle
            w_inter = weights[:len(d_inter)]
            grad_d_inter_x = w_inter * (dx / d_inter)
            grad_d_inter_y = w_inter * (dy / d_inter)
            
            np.add.at(g[1:1+N], IDX_I, -grad_d_inter_x) # d/dx_i (x_i-x_j) = 1
            np.add.at(g[1:1+N], IDX_J, grad_d_inter_x)  # d/dx_j (x_i-x_j) = -1
            np.add.at(g[1+N:1+2*N], IDX_I, -grad_d_inter_y)
            np.add.at(g[1+N:1+2*N], IDX_J, grad_d_inter_y)
            
            # Wall distances
            w_left = weights[len(d_inter):len(d_inter)+N]
            w_right = weights[len(d_inter)+N:len(d_inter)+2*N]
            w_bottom = weights[len(d_inter)+2*N:len(d_inter)+3*N]
            w_top = weights[len(d_inter)+3*N:]
            
            # d_left = x -> grad x = 1
            # d_right = w-x -> grad x = -1
            # d_bottom = y -> grad y = 1
            # d_top = h-y -> grad y = -1
            
            g[1:1+N] += -w_left + w_right
            g[1+N:1+2*N] += -w_bottom + w_top
            
            # Gradient w.r.t t
            # w = 2t, h = 2(1-t)
            # d_right = 2t - x -> dw/dt = 2
            # d_top = 2(1-t) - y -> dh/dt = -2
            # g_t = -1/beta * ( sum(w_right * 2) + sum(w_top * -2) )
            g[0] += - (2.0 * np.sum(w_right) - 2.0 * np.sum(w_top))
            
            return -g / beta # Remember obj was -lse/beta, so grad is -grad(lse)/beta

        bounds = [(0.15, 0.85)] + [(0.0, 2.0)] * (2 * N)
        try:
            res = minimize(obj, v0, method='L-BFGS-B', jac=grad, bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False})
            if res.success and res.fun < obj(v0):
                return res.x
        except Exception:
            pass
        return v0

    # ---------- Phase 2: Solve Radii via LP ----------
    def solve_radii_lp(w, h, x, y):
        if check_time():
            return None
            
        c = -np.ones(N) # Maximize sum(r) -> Minimize -sum(r)
        A = []
        b = []
        
        # Inter-circle constraints: r_i + r_j <= d_ij
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                row = np.zeros(N)
                row[i] = 1.0
                row[j] = 1.0
                A.append(row)
                b.append(d)
                
        # Wall constraints: r_i <= dist_to_wall
        for i in range(N):
            for bv in (x[i], w - x[i], y[i], h - y[i]):
                if bv > 1e-6:
                    row = np.zeros(N)
                    row[i] = 1.0
                    A.append(row)
                    b.append(bv)
                    
        try:
            res = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                          bounds=[(1e-8, None)] * N, method='highs')
            if res.success:
                return res.x
        except Exception:
            pass
        return None

    # ---------- Main Search Loop ----------
    best_val = -np.inf
    best_state = None

    def consider_geometry(v_geo):
        nonlocal best_val, best_state
        if check_time():
            return
        t = v_geo[0]
        w, h = get_rect_dims(t)
        x = v_geo[1:1 + N]
        y = v_geo[1 + N:1 + 2 * N]
        
        r = solve_radii_lp(w, h, x, y)
        if r is None:
            return
            
        if is_valid(w, h, x, y, r):
            val = sum_r(r)
            if val > best_val:
                best_val = val
                best_state = (t, x.copy(), y.copy(), r.copy())

    # Generate diverse initial geometries
    inits = []
    # Aspect ratios
    t_vals = [0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65]
    
    for t in t_vals:
        inits.append(make_hex(t))
        
    # Add some grid inits
    for t in [0.5, 0.5]:
        inits.append(make_grid(t, 5, 5))
        inits.append(make_grid(t, 4, 6))
        
    # Add random jittered inits
    for _ in range(5):
        t = random.uniform(0.3, 0.7)
        inits.append(make_hex(t, jitter=0.05))

    # Phase 1: Optimize Geometry for each init
    for v0 in inits:
        if check_time():
            break
        
        # Try a few beta values for the soft-min approximation
        # Lower beta = smoother landscape, easier to escape local minima
        # Higher beta = sharper, better local optimization
        for beta in [20.0, 50.0, 100.0]:
            if check_time():
                break
            v_opt = optimize_geometry(v0, beta=beta, maxiter=200)
            consider_geometry(v_opt)

    # Phase 2: Polish the best result found
    if best_state is not None and (deadline - time.time()) > 20.0:
        t_best, x_best, y_best, r_best = best_state
        v_best = np.zeros(1 + 2 * N)
        v_best[0] = t_best
        v_best[1:1 + N] = x_best
        v_best[1 + N:1 + 2 * N] = y_best
        
        # Final high-precision polish
        for beta in [200.0, 500.0]:
            if check_time():
                break
            v_polished = optimize_geometry(v_best, beta=beta, maxiter=500)
            consider_geometry(v_polished)
            if best_state is not None:
                t_best, x_best, y_best, r_best = best_state
                v_best = np.zeros(1 + 2 * N)
                v_best[0] = t_best
                v_best[1:1 + N] = x_best
                v_best[1 + N:1 + 2 * N] = y_best

    # Fallback if nothing worked
    if best_state is None:
        v_fallback = make_grid(0.5, 5, 5)
        consider_geometry(v_fallback)
        
        if best_state is None:
            # Absolute last resort: manual valid config
            t = 0.5
            w, h = get_rect_dims(t)
            x = np.linspace(0.2, w-0.2, N)
            y = np.linspace(0.2, h-0.2, N)
            r = np.full(N, 0.1)
            # Ensure validity by shrinking
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    r[i] = min(r[i], d/2 - 1e-5)
                    r[j] = min(r[j], d/2 - 1e-5)
                r[i] = min(r[i], x[i]-1e-5, w-x[i]-1e-5, y[i]-1e-5, h-y[i]-1e-5)
            best_state = (t, x, y, r)
            best_val = sum_r(r)

    t, x, y, r = best_state
    
    # Safety margin
    r = r * (1 - 1e-7)
    for i in range(N):
        r[i] = min(r[i], x[i], w - x[i], y[i], h - y[i])
        
    out = np.column_stack([x, y, r])
    return out
