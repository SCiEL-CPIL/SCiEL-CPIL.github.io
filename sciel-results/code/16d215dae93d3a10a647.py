"""Circle packing in perimeter-4 rectangle: Smooth Barrier Optimization."""
import numpy as np
import time

N = 21

def _lse(vals, k=100.0):
    m = np.max(vals)
    if not np.isfinite(m):
        return m
    return m + (1.0 / k) * np.log(np.sum(np.exp(k * (vals - m))))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def get_initial_guesses():
        """Generates structured initial configurations."""
        guesses = []
        
        # 1. Square Grid (5x5 -> 21 circles)
        side = 5
        w0, h0 = 1.0, 1.0
        z = np.zeros(3 * N + 1)
        for i in range(N):
            r_idx, c_idx = i // side, i % side
            z[3*i] = (c_idx + 0.5) / side * w0
            z[3*i+1] = (r_idx + 0.5) / side * h0
            z[3*i+2] = 0.45 * (w0 / side)
        z[-1] = w0
        guesses.append(z.copy())

        # 2. Hexagonal Lattice (approximate packing)
        # Rows of 5, 5, 5, 4, 2? No, let's try 4-5-4-4-4 or just 5 rows of 4/5.
        # Simpler: 5x5 hex offset.
        w0, h0 = 1.0, 1.0
        z = np.zeros(3 * N + 1)
        dy = 0.16
        dx = 0.18
        for i in range(N):
            row = i // 5
            col = i % 5
            x0 = 0.1 + col * dx
            y0 = 0.1 + row * dy
            if row % 2 != 0:
                x0 += dx * 0.5
            z[3*i] = x0
            z[3*i+1] = y0
            z[3*i+2] = 0.08
        z[-1] = w0
        guesses.append(z.copy())

        # 3. Fibonacci Disk / Sunflower model
        w0, h0 = 1.0, 1.0
        z = np.zeros(3 * N + 1)
        n = N
        golden_angle = np.pi * (3. - np.sqrt(5.))
        for i in range(n):
            r = np.sqrt((i + 0.5) / n) * 0.45
            theta = (i + 0.5) * golden_angle
            z[3*i] = 0.5 + r * np.cos(theta)
            z[3*i+1] = 0.5 + r * np.sin(theta)
            z[3*i+2] = 0.05
        z[-1] = w0
        guesses.append(z.copy())
        
        # 4. Rectangle variations (Aspect Ratio Search)
        aspects = [1.2, 0.8, 1.5, 0.6]
        for a in aspects:
            w0 = a * (1.0 / (1.0 + a)) * 2.0 # Perimeter 4 -> 2(w+h)=4 -> w+h=2
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            side = 5
            for i in range(N):
                r_idx, c_idx = i // side, i % side
                z[3*i] = (c_idx + 0.5) / side * w0
                z[3*i+1] = (r_idx + 0.5) / side * h0
                z[3*i+2] = 0.45 * min(w0, h0) / side
            z[-1] = w0
            guesses.append(z.copy())

        return guesses

    def compute_obj_grad(z, K=100.0):
        """
        Computes objective and gradient using LSE barriers for constraints.
        Variables: x[0..N-1], y[0..N-1], r[0..N-1], w
        Perimeter constraint is handled by w+h=2 -> h=2-w.
        Objective: Maximize sum(r) => Minimize -sum(r)
        Penalties:
        1. Pair overlap: max(0, r_i + r_j - d_ij)
        2. Boundary: max(0, r_i - x_i), max(0, r_i - y_i), max(0, r_i + x_i - w), max(0, r_i + y_i - h)
        """
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        
        if w <= 0.0 or h <= 0.0:
            return 1e6, np.zeros_like(z)

        neg_sum_r = -np.sum(r)
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        grad_r = -np.ones(N)
        grad_w = 0.0

        iu = np.triu_indices(N, k=1)
        pairs = iu
        
        # 1. Pairwise Constraints
        dx = x[pairs[0]] - x[pairs[1]]
        dy = y[pairs[0]] - y[pairs[1]]
        d = np.sqrt(dx*dx + dy*dy + 1e-12)
        
        # Violation amount for each pair
        r_sum = r[pairs[0]] + r[pairs[1]]
        viol_pairs = np.maximum(0.0, r_sum - d)
        
        # LSE barrier
        if np.max(viol_pairs) > 0:
            # Gradient of LSE: exp(k(v-m)) / sum(exp(k(v-m))) * k
            # We use the analytic derivative of the LSE sum
            # L = (1/k) log(sum exp(k * viol))
            # dL/dviol = exp(k*viol) / sum(exp(k*viol))
            
            exp_v = np.exp(K * (viol_pairs - np.max(viol_pairs)))
            sum_exp = np.sum(exp_v)
            weights = exp_v / sum_exp
            
            # Gradient w.r.t x
            for idx in range(len(pairs[0])):
                vi = pairs[0][idx]
                vj = pairs[1][idx]
                # Force is proportional to weight and direction (dx/d, dy/d)
                # If overlap, d<0 for distance derivative? No, d is distance.
                # d(d)/dx_i = dx/d.
                # d(viol)/dx_i = -dx/d.
                # dL/dx_i = w * d(viol)/dx_i = -w * dx/d
                
                if d[idx] > 1e-9:
                    fx = weights[idx] * K * (dx[idx] / d[idx])
                    fy = weights[idx] * K * (dy[idx] / d[idx])
                    
                    # x_i moves in +fx direction to increase d? 
                    # dx = x_i - x_j. 
                    # If dx>0, x_i is right of x_j.
                    # To increase d, x_i should move right (+), x_j left (-).
                    # dx/d = 1 (for x_i), -1 (for x_j).
                    # So grad x_i gets -weight * (dx/d) ?
                    # Wait, LSE of viol = max(0, r_i+r_j - d).
                    # dL/dx_i = w * d(viol)/dx_i = w * (0 - 1) * d(d)/dx_i = -w * (dx/d).
                    # Yes.
                    grad_x[vi] -= weights[idx] * K * (dx[idx] / d[idx])
                    grad_x[vj] += weights[idx] * K * (dx[idx] / d[idx])
                    grad_y[vi] -= weights[idx] * K * (dy[idx] / d[idx])
                    grad_y[vj] += weights[idx] * K * (dy[idx] / d[idx])
                    
                    # d(viol)/dr_i = 1
                    grad_r[vi] -= weights[idx] * K
                    grad_r[vj] -= weights[idx] * K
        
        # 2. Boundary Constraints
        # Left: max(0, r - x)
        viol_l = np.maximum(0.0, r - x)
        if np.max(viol_l) > 0:
            exp_v = np.exp(K * (viol_l - np.max(viol_l)))
            sum_exp = np.sum(exp_v)
            weights = exp_v / sum_exp
            # d(viol)/dx = -1, d(viol)/dr = 1
            grad_x += weights * K
            grad_r -= weights * K
            
        # Right: max(0, r + x - w)
        viol_r = np.maximum(0.0, r + x - w)
        if np.max(viol_r) > 0:
            exp_v = np.exp(K * (viol_r - np.max(viol_r)))
            sum_exp = np.sum(exp_v)
            weights = exp_v / sum_exp
            # d(viol)/dx = 1, d(viol)/dr = 1, d(viol)/dw = -1
            grad_x -= weights * K
            grad_r -= weights * K
            grad_w += weights * K
            
        # Bottom: max(0, r - y)
        viol_b = np.maximum(0.0, r - y)
        if np.max(viol_b) > 0:
            exp_v = np.exp(K * (viol_b - np.max(viol_b)))
            sum_exp = np.sum(exp_v)
            weights = exp_v / sum_exp
            grad_y += weights * K
            grad_r -= weights * K
            
        # Top: max(0, r + y - h) = max(0, r + y - (2-w)) = max(0, r + y + w - 2)
        viol_t = np.maximum(0.0, r + y + w - 2.0)
        if np.max(viol_t) > 0:
            exp_v = np.exp(K * (viol_t - np.max(viol_t)))
            sum_exp = np.sum(exp_v)
            weights = exp_v / sum_exp
            # d(viol)/dy = 1, d(viol)/dr = 1, d(viol)/dw = 1
            grad_y -= weights * K
            grad_r -= weights * K
            grad_w += weights * K
            
        # 3. LSE Sum Value
        lse_val = 0.0
        if np.max(viol_pairs) > 0:
            lse_val += (1.0/K) * np.log(np.sum(np.exp(K * viol_pairs)))
        if np.max(viol_l) > 0: lse_val += (1.0/K) * np.log(np.sum(np.exp(K * viol_l)))
        if np.max(viol_r) > 0: lse_val += (1.0/K) * np.log(np.sum(np.exp(K * viol_r)))
        if np.max(viol_b) > 0: lse_val += (1.0/K) * np.log(np.sum(np.exp(K * viol_b)))
        if np.max(viol_t) > 0: lse_val += (1.0/K) * np.log(np.sum(np.exp(K * viol_t)))

        obj = neg_sum_r + lse_val
        
        # Assemble Gradient
        grad = np.zeros_like(z)
        grad[0::3][:N] = grad_x
        grad[1::3][:N] = grad_y
        grad[2::3][:N] = grad_r
        grad[-1] = grad_w
        
        return obj, grad

    def check_validity(z, tol=1e-7):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        
        if w < 1e-3 or h < 1e-3:
            return False, 0.0
            
        # Boundaries
        if np.any(x - r < -tol) or np.any(y - r < -tol):
            return False, 0.0
        if np.any(x + r > w + tol) or np.any(y + r > h + tol):
            return False, 0.0
            
        # Overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx*dx + dy*dy
                r_sum = r[i] + r[j]
                if d_sq < r_sum*r_sum - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    def gradient_descent(z0, time_limit, K=100.0):
        z = z0.copy()
        best_val = -np.inf
        best_z = z.copy()
        last_time = time.time()
        lr = 0.001
        iu = np.triu_indices(N, k=1)
        
        while time.time() < time_limit:
            obj, grad = compute_obj_grad(z, K)
            
            # Check validity for bookkeeping
            valid, val = check_validity(z)
            if valid and val > best_val:
                best_val = val
                best_z = z.copy()
                
            # Adaptive Learning Rate
            g_norm = np.linalg.norm(grad)
            if g_norm < 1e-8:
                lr *= 0.5
            elif g_norm > 10.0:
                lr *= 0.8
            else:
                lr = min(lr * 1.05, 0.01)
                
            # Update
            z = z - lr * grad
            
            # Simple Projection (soft)
            # Ensure w in [0.2, 1.8]
            z[-1] = np.clip(z[-1], 0.2, 1.8)
            
            # Ensure Radii positive
            z[2::3][:N] = np.maximum(z[2::3][:N], 1e-6)
            
            # If K is getting very large or constraints tight, adjust K slightly?
            # Keep K fixed for stability in this loop, maybe increase slowly?
            if obj < 0.1 and time.time() < time_limit - 5.0:
                # Progressing well, maybe reduce K to smooth landscape?
                # Or increase to harden? 
                # Let's keep K constant to avoid oscillation.
                pass
            
            # Periodically reset lr if stuck
            if time.time() - last_time > 2.0:
                last_time = time.time()
                if g_norm < 1e-5:
                    lr = 0.002
        
        return best_z, best_val

    def polish_lagrange(z, time_limit):
        """
        Refines a valid configuration by adjusting radii and positions 
        based on active constraints (Lagrange Multipliers intuition).
        """
        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        h = 2.0 - w
        
        # Build active constraint graph or just iterate
        for _ in range(50):
            if time.time() > time_limit:
                break
            changed = False
            for i in range(N):
                # Try to increase r[i]
                old_r = r[i]
                # Max possible r based on neighbors
                max_r = min(x[i], w-x[i], y[i], h-y[i])
                
                # Check neighbors
                for j in range(N):
                    if i == j: continue
                    d = np.hypot(x[i]-x[j], y[i]-y[j])
                    max_r = min(max_r, d - r[j])
                
                max_r = max(max_r, 1e-6)
                
                # Binary search for best r that fits without moving?
                # Actually, if we just grow r, it might violate.
                # This polish assumes positions are fixed and tries to maximize r.
                # But positions might not be optimal for max r.
                
                # Let's do a local search: try moving i and growing r.
                # For simplicity, let's just do the simple grow loop.
                
                lo, hi = r[i], max_r
                if hi > lo:
                    for _ in range(10):
                        mid = (lo + hi) / 2.0
                        r_tmp = r.copy()
                        r_tmp[i] = mid
                        # Quick validity check for i
                        if (x[i] >= mid and x[i] + mid <= w and y[i] >= mid and y[i] + mid <= h):
                             ok = True
                             for j in range(N):
                                 if i==j: continue
                                 d = np.hypot(x[i]-x[j], y[i]-y[j])
                                 if d < r_tmp[i] + r[j] - 1e-9:
                                     ok = False
                                     break
                             if ok:
                                 lo = mid
                                 changed = True
                             else:
                                 hi = mid
                        else:
                             hi = lo
                    r[i] = lo
            
            if not changed:
                break
                
        valid, val = check_validity(np.hstack([x, y, r, [w]]))
        if valid:
             return np.hstack([x, y, r, [w]])
        return None

    guesses = get_initial_guesses()
    best_z = None
    best_score = -np.inf
    
    # Allocate time per guess
    time_per_guess = 15.0
    
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 10.0:
            break
            
        t_start = time.time()
        t_limit = t_start + time_per_guess
        
        # Run Gradient Descent
        z_opt, val = gradient_descent(z0, t_limit)
        
        # Polish
        if time.time() < deadline - 5.0:
            z_polished = polish_lagrange(z_opt, time.time() + 2.0)
            if z_polished is not None:
                valid, v = check_validity(z_polished)
                if valid and v > val:
                    z_opt = z_polished
                    val = v
                    
        if val > best_score:
            best_score = val
            best_z = z_opt
            
    if best_z is None:
        # Fallback
        z = np.zeros(3*N+1)
        for i in range(N):
            z[3*i] = 0.5
            z[3*i+1] = 0.5
            z[3*i+2] = 0.01
        z[-1] = 1.0
        return np.column_stack([z[0::3][:N], z[1::3][:N], z[2::3][:N]])
        
    x = best_z[0::3][:N]
    y = best_z[1::3][:N]
    r = best_z[2::3][:N]
    
    return np.column_stack([x, y, r])

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(res.shape)
