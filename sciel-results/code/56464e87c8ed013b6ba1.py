import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    def is_valid(x, y, r, tol=1e-7):
        if np.any(r < tol):
            return False
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r[i] + r[j] - tol:
                    return False
        return True

    def score(x, y, r):
        return np.sum(r)

    def get_initial_guess(method='grid'):
        if method == 'grid':
            side = int(np.ceil(np.sqrt(N)))
            xs = np.linspace(0.1, 0.9, side)
            ys = np.linspace(0.1, 0.9, side)
            x, y = np.meshgrid(xs, ys)
            x = x.flatten()
            y = y.flatten()
            # Take first N
            x = x[:N]
            y = y[:N]
            r = np.full(N, 0.15) # Start with reasonable radius
            return x, y, r
        elif method == 'hex':
            # Simple hex-like pattern
            x = []
            y = []
            r = []
            # 5 rows
            for row in range(5):
                for col in range(6):
                    if len(x) >= N:
                        break
                    xx = 0.1 + col * 0.15 + (0.075 if row % 2 == 1 else 0)
                    yy = 0.1 + row * 0.18
                    if xx > 0.9:
                        continue
                    x.append(xx)
                    y.append(yy)
                    r.append(0.08)
            # Pad if necessary
            while len(x) < N:
                x.append(0.5)
                y.append(0.5)
                r.append(0.05)
            return np.array(x[:N]), np.array(y[:N]), np.array(r[:N])
        else: # random
            rng = np.random.RandomState(42 + hash(method) % 100)
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)
            r = rng.uniform(0.05, 0.1, N)
            return x, y, r

    best_result = None
    best_score = -np.inf

    # Optimization function using SLSQP
    def optimize_slsqp(x0, y0, r0, time_limit=10.0):
        params = np.concatenate([x0, y0, r0])
        
        def objective(p):
            x, y, r = p[:N], p[N:2*N], p[2*N:3*N]
            return -np.sum(r)
            
        # Constraints
        cons = []
        # Boundary constraints
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*N+i]}) # x >= r
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i] - p[2*N+i]}) # 1-x >= r
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[N+i] - p[2*N+i]}) # y >= r
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[N+i] - p[2*N+i]}) # 1-y >= r
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*N+i] - 1e-6}) # r >= eps

        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: 
                             (p[i]-p[j])**2 + (p[N+i]-p[N+j])**2 - (p[2*N+i]+p[2*N+j])**2})
                             
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N

        t_start = time.time()
        try:
            res = minimize(
                objective, 
                params, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            if time.time() - t_start > time_limit:
                return None
            return res.x
        except Exception:
            return None

    # Optimization function using L-BFGS-B with penalty (faster, less strict)
    def optimize_lbfgsb(x0, y0, r0, time_limit=10.0):
        params = np.concatenate([x0, y0, r0])
        
        def objective(p):
            x, y, r = p[:N], p[N:2*N], p[2*N:3*N]
            obj = -np.sum(r)
            pen = 0.0
            # Boundary
            pen += np.sum(np.maximum(0, r - x)**2)
            pen += np.sum(np.maximum(0, r - (1-x))**2)
            pen += np.sum(np.maximum(0, r - y)**2)
            pen += np.sum(np.maximum(0, r - (1-y))**2)
            # Overlap
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if d < r[i] + r[j]:
                        pen += (r[i] + r[j] - d)**2
            return obj + 100.0 * pen

        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
        t_start = time.time()
        try:
            res = minimize(
                objective, 
                params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 5000}
            )
            if time.time() - t_start > time_limit:
                return None
            return res.x
        except Exception:
            return None

    def repair(p):
        """Shrink radii to ensure validity if small violations exist."""
        x, y, r = p[:N].copy(), p[N:2*N].copy(), p[2*N:3*N].copy()
        for _ in range(50):
            changed = False
            # Boundary repair
            for i in range(N):
                r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i], 0.5)
                if r[i] < 1e-6:
                    r[i] = 1e-6
            # Overlap repair
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    min_d = r[i] + r[j]
                    if d < min_d - 1e-8:
                        shrink = (min_d - d) / 2 + 1e-7
                        r[i] = max(1e-6, r[i] - shrink)
                        r[j] = max(1e-6, r[j] - shrink)
                        changed = True
            if not changed:
                break
        return np.concatenate([x, y, r])

    # Strategies
    strategies = [
        ('grid', 'slsqp'),
        ('hex', 'slsqp'),
        ('random', 'slsqp'),
        ('grid', 'lbfgsb'),
        ('hex', 'lbfgsb'),
        ('random', 'lbfgsb'),
    ]

    for init_type, opt_type in strategies:
        if time.time() > deadline:
            break
            
        x0, y0, r0 = get_initial_guess(init_type)
        
        if opt_type == 'slsqp':
            p = optimize_slsqp(x0, y0, r0, time_limit=15.0)
        else:
            p = optimize_lbfgsb(x0, y0, r0, time_limit=15.0)
            
        if p is None:
            continue
            
        # Repair and validate
        p_repaired = repair(p)
        x, y, r = p_repaired[:N], p_repaired[N:2*N], p_repaired[2*N:3*N]
        
        if is_valid(x, y, r):
            s = score(x, y, r)
            if s > best_score:
                best_score = s
                best_result = np.column_stack((x, y, r))

    if best_result is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        best_result = np.array(cells[:N])
        
    return best_result
