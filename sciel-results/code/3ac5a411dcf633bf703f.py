"""Circle packing in perimeter-4 rectangle: Force-Directed + SLSQP + Expansion Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.05 or h <= 0.05:
            return False
        if np.any(x - r < -tol) or np.any(y - r < -tol):
            return False
        if np.any(x + r > w + tol) or np.any(y + r > h + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False
        return True

    def get_max_radius(x, y, r, w, h, i):
        """Calculate the maximum radius for circle i given current positions of others."""
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        r_max = r_wall
        for j in range(N):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            if r_allowed < r_max:
                r_max = r_allowed
        return max(0.0, r_max)

    def expand_radii(x, y, r, w, h, time_budget):
        """Iteratively expand radii to the maximum possible."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        max_rounds = 20
        for _ in range(max_rounds):
            if time.time() > time_budget:
                break
            improved = False
            for i in range(N):
                # Check if circle i can be expanded
                r_max = get_max_radius(x, y, r, w, h, i)
                if r_max > r[i] + 1e-6:
                    # Binary search for the exact max radius
                    lo = r[i]
                    hi = r_max
                    for _ in range(15):
                        mid = (lo + hi) / 2.0
                        r[i] = mid
                        if check_validity(x, y, r, w, h):
                            lo = mid
                        else:
                            hi = mid
                    r[i] = lo
                    if r[i] > r[i] * (1 + 1e-6): # Check if it actually grew
                        improved = True
            
            if not improved:
                break
        
        return x, y, r

    def force_relax(x, y, r, w, h, max_iter=1000):
        """Force-directed relaxation to resolve overlaps."""
        x = x.copy()
        y = y.copy()
        dt = 0.01
        for _ in range(max_iter):
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Wall repulsion
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 100
                if x[i] > w - r[i]:
                    fx[i] -= (x[i] - (w - r[i])) * 100
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 100
                if y[i] > h - r[i]:
                    fy[i] -= (y[i] - (h - r[i])) * 100

            # Pairwise repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_sum = r[i] + r[j]
                    if d < r_sum:
                        overlap = r_sum - d
                        if d > 1e-6:
                            fx[i] += (dx / d) * overlap * 10
                            fy[i] += (dy / d) * overlap * 10
                            fx[j] -= (dx / d) * overlap * 10
                            fy[j] -= (dy / d) * overlap * 10
                        else:
                            # Random push if coincident
                            fx[i] += np.random.uniform(-1, 1) * overlap * 10
                            fy[i] += np.random.uniform(-1, 1) * overlap * 10
                            fx[j] -= np.random.uniform(-1, 1) * overlap * 10
                            fy[j] -= np.random.uniform(-1, 1) * overlap * 10

            # Update positions
            x += dt * fx
            y += dt * fy
            
            # Clamp
            for i in range(N):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])
                
            # Check convergence
            max_overlap = 0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < r[i] + r[j]:
                        max_overlap = max(max_overlap, r[i] + r[j] - d)
            if max_overlap < 1e-6:
                break

        return x, y

    def slsqp_optimize(x, y, r, w, h, time_budget):
        """SLSQP optimization to maximize sum of radii."""
        def objective(z):
            x_ = z[0::3][:N]
            y_ = z[1::3][:N]
            r_ = z[2::3][:N]
            w_ = z[-1]
            h_ = 2.0 - w_
            
            if w_ < 0.1 or h_ < 0.1 or np.any(r_ < 0):
                return 1e6
                
            # Penalty for overlaps
            penalty = 0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x_[i] - x_[j]
                    dy = y_[i] - y_[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_sum = r_[i] + r_[j]
                    if d < r_sum:
                        penalty += (r_sum - d) ** 2
            
            # Penalty for boundaries
            for i in range(N):
                if x_[i] < r_[i]: penalty += (r_[i] - x_[i]) ** 2
                if x_[i] > w_ - r_[i]: penalty += (x_[i] - (w_ - r_[i])) ** 2
                if y_[i] < r_[i]: penalty += (r_[i] - y_[i]) ** 2
                if y_[i] > h_ - r_[i]: penalty += (y_[i] - (h_ - r_[i])) ** 2
                
            return -np.sum(r_) + 100 * penalty

        z0 = np.zeros(3 * N + 1)
        z0[0::3][:N] = x
        z0[1::3][:N] = y
        z0[2::3][:N] = r
        z0[-1] = w

        bounds = []
        for i in range(N):
            bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
        bounds.append((0.3, 1.7))

        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-12})
            if res.success:
                x_opt = res.x[0::3][:N]
                y_opt = res.x[1::3][:N]
                r_opt = res.x[2::3][:N]
                w_opt = res.x[-1]
                h_opt = 2.0 - w_opt
                if check_validity(x_opt, y_opt, r_opt, w_opt, h_opt):
                    return x_opt, y_opt, r_opt, w_opt, h_opt
        except Exception:
            pass
        return x, y, r, w, h

    # Generate diverse initial guesses
    guesses = []
    side = int(np.ceil(np.sqrt(N)))
    
    # Grid starts
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55]
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            x0[i] = (col + 0.5) / side * w0
            y0[i] = (row + 0.5) / side * h0
            r0[i] = min(w0 / side, h0 / side) * 0.45
        guesses.append((x0, y0, r0, w0, h0))

    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        w0 = rng.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        for i in range(N):
            r0[i] = rng.uniform(0.05, 0.12)
            x0[i] = rng.uniform(r0[i], w0 - r0[i])
            y0[i] = rng.uniform(r0[i], h0 - r0[i])
        guesses.append((x0, y0, r0, w0, h0))

    for i, (x0, y0, r0, w0, h0) in enumerate(guesses):
        if time.time() > deadline - 15:
            break

        # 1. Force Relax
        x_rel, y_rel = force_relax(x0, y0, r0, w0, h0)
        
        # 2. SLSQP Optimize
        x_opt, y_opt, r_opt, w_opt, h_opt = slsqp_optimize(x_rel, y_rel, r0, w0, h0, deadline - 10)
        
        # 3. Expand Radii
        x_exp, y_exp, r_exp = expand_radii(x_opt, y_opt, r_opt, w_opt, h_opt, deadline - 5)
        
        val = np.sum(r_exp)
        if val > best_value:
            best_value = val
            best_result = (x_exp, y_exp, r_exp, w_opt, h_opt)

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result
    
    # Final safety check and shrink
    for _ in range(10):
        if check_validity(x, y, r, w, h):
            break
        r = r * 0.99

    return np.column_stack((x, y, r))
