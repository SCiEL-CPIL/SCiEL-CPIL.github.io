import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    best_value = -np.inf
    best_solution = None
    
    # Define the cost function
    def cost_and_grad(z):
        w = z[0]
        h = 2.0 - w
        if w < 0.1 or h < 0.1:
            return 1e10, np.zeros_like(z)
        
        x = z[1::3]
        y = z[2::3]
        r = z[3::3]
        
        cost = -np.sum(r)
        grad = np.zeros_like(z)
        
        Lambda = 50.0 # Increased penalty weight for better constraint enforcement
        
        # Boundary and Overlap constraints
        # We use quadratic penalties for smooth gradients
        
        # 1. Boundary constraints
        # x_i >= r_i, x_i <= w - r_i
        # y_i >= r_i, y_i <= h - r_i
        
        for i in range(N):
            xi, yi, ri = x[i], y[i], r[i]
            
            # Left
            viol = ri - xi
            if viol > 0:
                cost += Lambda * viol**2
                grad[1 + 3*i] -= 2 * Lambda * viol
                grad[3 + 3*i] += 2 * Lambda * viol
                
            # Right
            viol = xi + ri - w
            if viol > 0:
                cost += Lambda * viol**2
                grad[1 + 3*i] += 2 * Lambda * viol
                grad[3 + 3*i] += 2 * Lambda * viol
                grad[0] -= 2 * Lambda * viol
                
            # Bottom
            viol = ri - yi
            if viol > 0:
                cost += Lambda * viol**2
                grad[2 + 3*i] -= 2 * Lambda * viol
                grad[3 + 3*i] += 2 * Lambda * viol
                
            # Top
            viol = yi + ri - h
            if viol > 0:
                cost += Lambda * viol**2
                grad[2 + 3*i] += 2 * Lambda * viol
                grad[3 + 3*i] += 2 * Lambda * viol
                grad[0] += 2 * Lambda * viol
                
            # Objective gradient for radius
            grad[3 + 3*i] -= 1.0

        # 2. Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + 1e-12)
                r_sum = r[i] + r[j]
                
                viol = r_sum - dist
                if viol > 0:
                    cost += Lambda * viol**2
                    
                    # Gradients
                    factor = 2 * Lambda * viol / dist
                    
                    grad[1 + 3*i] -= factor * dx
                    grad[2 + 3*i] -= factor * dy
                    grad[1 + 3*j] += factor * dx
                    grad[2 + 3*j] += factor * dy
                    
                    grad[3 + 3*i] += 2 * Lambda * viol
                    grad[3 + 3*j] += 2 * Lambda * viol

        return cost, grad

    # Helper to create initial guess
    def make_initial(w, h, pattern='hex'):
        z = np.zeros(1 + 3*N)
        z[0] = w
        
        if pattern == 'hex':
            # Hexagonal packing initialization
            # Approximate radius
            r_est = 0.5 / np.sqrt(N) * 0.9
            spacing = 2 * r_est
            xs = []
            ys = []
            row = 0
            while len(xs) < N:
                offset = 0 if row % 2 == 0 else spacing / 2
                for col in range(10):
                    x = spacing / 2 + col * spacing + offset
                    y = spacing / 2 + row * spacing * np.sqrt(3) / 2
                    if x < w and y < h:
                        xs.append(x)
                        ys.append(y)
                        if len(xs) == N:
                            break
                row += 1
            if len(xs) < N:
                # Fill remaining with random
                for _ in range(N - len(xs)):
                    xs.append(np.random.uniform(r_est, w - r_est))
                    ys.append(np.random.uniform(r_est, h - r_est))
            
            for i in range(N):
                z[1 + 3*i] = xs[i]
                z[2 + 3*i] = ys[i]
                z[3 + 3*i] = r_est
        else:
            # Grid
            side = int(np.ceil(np.sqrt(N)))
            r_est = min(w, h) / side * 0.45
            for i in range(N):
                row = i // side
                col = i % side
                z[1 + 3*i] = (col + 0.5) * w / side
                z[2 + 3*i] = (row + 0.5) * h / side
                z[3 + 3*i] = r_est
                
        return z

    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.4, 0.6, 1.5, 0.5]
    
    for ar in aspect_ratios:
        w = 2.0 * ar / (1 + ar)
        h = 2.0 - w
        
        # Try different initial patterns
        for pattern in ['hex', 'grid']:
            np.random.seed(int(ar * 100) + (0 if pattern == 'hex' else 1))
            z0 = make_initial(w, h, pattern)
            
            try:
                res = minimize(cost_and_grad, z0, method='L-BFGS-B', jac=True, 
                               options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
                
                z_opt = res.x
                w_opt = z_opt[0]
                h_opt = 2.0 - w_opt
                if w_opt < 0.1 or h_opt < 0.1:
                    continue
                    
                x = z_opt[1::3]
                y = z_opt[2::3]
                r = z_opt[3::3]
                
                # Validate
                if np.any(r <= 0):
                    continue
                
                valid = True
                for i in range(N):
                    if x[i] < r[i] - 1e-6 or x[i] > w_opt - r[i] + 1e-6 or \
                       y[i] < r[i] - 1e-6 or y[i] > h_opt - r[i] + 1e-6:
                        valid = False
                        break
                
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if d < r[i] + r[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    val = np.sum(r)
                    if val > best_value:
                        best_value = val
                        best_solution = np.column_stack((x, y, r))
                        
            except Exception:
                pass

    if best_solution is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_solution
