import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_value = -np.inf
    best_result = None
    
    def evaluate_solution(x, y, r, W):
        """Check if solution is valid and return sum of radii."""
        H = 2.0 - W
        if W < 0.01 or H < 0.01:
            return -np.inf
        for i in range(N):
            if r[i] < 1e-10:
                return -np.inf
            if x[i] - r[i] < -1e-9 or x[i] + r[i] > W + 1e-9:
                return -np.inf
            if y[i] - r[i] < -1e-9 or y[i] + r[i] > H + 1e-9:
                return -np.inf
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist*min_dist - 1e-9:
                    return -np.inf
        return np.sum(r)
    
    def make_constraints():
        constraints = []
        wi = 3*N
        constraints.append({'type': 'ineq', 'fun': lambda v: v[wi] - 0.1})
        constraints.append({'type': 'ineq', 'fun': lambda v: 1.9 - v[wi]})
        
        for i in range(N):
            xi, yi, ri = i*3, i*3+1, i*3+2
            constraints.append({'type': 'ineq', 'fun': lambda v, ri=ri: v[ri] - 1e-4})
            constraints.append({'type': 'ineq', 'fun': lambda v, xi=xi, ri=ri: v[xi] - v[ri]})
            constraints.append({'type': 'ineq', 'fun': lambda v, xi=xi, ri=ri, wi=wi: v[wi] - v[xi] - v[ri]})
            constraints.append({'type': 'ineq', 'fun': lambda v, yi=yi, ri=ri: v[yi] - v[ri]})
            constraints.append({'type': 'ineq', 'fun': lambda v, yi=yi, ri=ri, wi=wi: 2.0 - v[wi] - v[yi] - v[ri]})
        
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = i*3, i*3+1, i*3+2
                xj, yj, rj = j*3, j*3+1, j*3+2
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda v, xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj: 
                        (v[xi]-v[xj])**2 + (v[yi]-v[yj])**2 - (v[ri]+v[rj])**2
                })
        return constraints
    
    def objective(v):
        r = v[2::3][:N]
        return -np.sum(r)
    
    def optimize_from_initial(x0, y0, r0, W0, maxiter=300, ftol=1e-10):
        v0 = np.zeros(3*N + 1)
        v0[0::3] = x0
        v0[1::3] = y0
        v0[2::3] = r0
        v0[3*N] = W0
        
        bounds = [(0, None)] * N + [(0, None)] * N + [(1e-4, None)] * N + [(0.1, 1.9)]
        
        try:
            result = minimize(
                objective, v0, method='SLSQP',
                bounds=bounds,
                constraints=make_constraints(),
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            if result.success:
                v = result.x
                x = v[0::3][:N]
                y = v[1::3][:N]
                r = v[2::3][:N]
                W = v[3*N]
                val = evaluate_solution(x, y, r, W)
                if val > -np.inf:
                    return x, y, r, W, val
        except Exception:
            pass
        return None
    
    # Phase 1: Grid initializations
    aspect_ratios = [0.6, 0.8, 1.0, 1.2, 1.5]
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 / (1.0 + ar)
        
        for cols in [4, 5, 6]:
            if time.time() > deadline:
                break
            rows = max(1, int(np.ceil(N / cols)))
            if cols * rows < N:
                continue
            
            x = np.array([(i + 0.5) * W / cols for i in range(cols) for j in range(rows)])[:N]
            y = np.array([(j + 0.5) * H / rows for j in range(rows) for i in range(cols)])[:N]
            r = np.full(N, min(W/cols, H/rows) * 0.4)
            
            result = optimize_from_initial(x, y, r, W)
            if result is not None:
                x_opt, y_opt, r_opt, W_opt, val = result
                if val > best_value:
                    best_value = val
                    best_result = (x_opt, y_opt, r_opt, W_opt)
    
    # Phase 2: Random restarts
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        W0 = np.random.uniform(0.6, 1.4)
        H0 = 2.0 - W0
        x0 = np.random.uniform(0, W0, N)
        y0 = np.random.uniform(0, H0, N)
        r0 = np.random.uniform(0.03, 0.12, N)
        for i in range(N):
            x0[i] = np.clip(x0[i], r0[i], W0 - r0[i])
            y0[i] = np.clip(y0[i], r0[i], H0 - r0[i])
        
        result = optimize_from_initial(x0, y0, r0, W0, maxiter=200, ftol=1e-9)
        if result is not None:
            x_opt, y_opt, r_opt, W_opt, val = result
            if val > best_value:
                best_value = val
                best_result = (x_opt, y_opt, r_opt, W_opt)
    
    # Phase 3: Polish
    if best_result is not None and time.time() < deadline - 30:
        x_best, y_best, r_best, W_best = best_result
        v0 = np.zeros(3*N + 1)
        v0[0::3] = x_best
        v0[1::3] = y_best
        v0[2::3] = r_best
        v0[3*N] = W_best
        bounds = [(0, None)] * N + [(0, None)] * N + [(1e-6, None)] * N + [(0.1, 1.9)]
        try:
            result = minimize(
                objective, v0, method='SLSQP',
                bounds=bounds,
                constraints=make_constraints(),
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            if result.success:
                v = result.x
                x = v[0::3][:N]
                y = v[1::3][:N]
                r = v[2::3][:N]
                W = v[3*N]
                val = evaluate_solution(x, y, r, W)
                if val > best_value:
                    best_value = val
                    best_result = (x, y, r, W)
        except Exception:
            pass
    
    if best_result is not None:
        x, y, r, W = best_result
        H = 2.0 - W
        eps = 1e-7
        x = np.clip(x, r + eps, W - r - eps)
        y = np.clip(y, r + eps, H - r - eps)
        r = np.maximum(r, 1e-7)
        return np.column_stack([x, y, r])
    else:
        W = 1.0; H = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append([(i+0.5)*W/side, (j+0.5)*H/side, 0.4*min(W,H)/side])
        return np.array(cells[:N])
