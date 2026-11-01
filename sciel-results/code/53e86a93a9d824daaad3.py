"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

def _valid(arr):
    """Check if arrangement is valid: perimeter <= 4, circles inside, disjoint interiors."""
    if arr is None or len(arr) != N:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r <= 0):
        return False
    # Bounding box
    xmin, xmax = np.min(x - r), np.max(x + r)
    ymin, ymax = np.min(y - r), np.max(y + r)
    w = xmax - xmin
    h = ymax - ymin
    if w <= 0 or h <= 0:
        return False
    if 2 * (w + h) > 4.0 + 1e-9:
        return False
    # Check disjoint interiors: distance between centers >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - 1e-9:
                return False
    return True

def _score(arr):
    """Compute sum of radii if valid, else -inf."""
    if not _valid(arr):
        return -np.inf
    return np.sum(arr[:, 2])

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    best_arr = None
    best_score = -np.inf

    # Initial grid-based arrangement
    side = int(np.ceil(np.sqrt(N)))  # 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4])
    arr0 = np.array(cells)

    if _valid(arr0):
        best_arr = arr0.copy()
        best_score = _score(arr0)

    # Define optimization variables: (x, y, r) for each circle
    # We'll optimize positions and radii jointly
    # Objective: maximize sum(r) subject to constraints
    
    def _neg_sum_r(z):
        # z is flattened array of length 3*N
        return -np.sum(z[2::3])

    def _constraints(z):
        n = len(z) // 3
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        cons = []
        
        # Radius positivity
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2 + 3*i] - 1e-8})
        
        # Perimeter constraint: 2*(w+h) <= 4 => w+h <= 2
        # w = max(x+r) - min(x-r), h = max(y+r) - min(y-r)
        # This is non-smooth. We'll use a different approach:
        # Instead, we'll parameterize the rectangle and ensure circles fit inside.
        # Let's use a fixed aspect ratio approach or optimize w,h as well.
        
        # Actually, let's use a different formulation:
        # Variables: x_i, y_i, r_i for i=0..N-1
        # Constraints:
        # 1. r_i >= r_min
        # 2. For all i<j: (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
        # 3. Perimeter: 2*(max(x_i+r_i) - min(x_i-r_i) + max(y_i+y_i) - min(y_i-r_i)) <= 4
        
        # The non-smooth perimeter constraint is problematic for SLSQP.
        # Alternative: use a soft penalty or reformulate.
        
        # Let's try a different approach: fix the rectangle and optimize circles inside it,
        # then try different rectangle aspect ratios.
        
        return cons
    
    # Better approach: for a given rectangle (w, h) with 2*(w+h)<=4,
    # optimize circle positions and radii to maximize sum(r)
    # subject to circles being inside the rectangle and non-overlapping.
    
    def _optimize_in_rect(w, h, arr_init, deadline):
        """Optimize circles inside a rectangle of size w x h."""
        # Variables: x_i, y_i, r_i
        # Constraints:
        # - x_i - r_i >= 0, x_i + r_i <= w
        # - y_i - r_i >= 0, y_i + r_i <= h
        # - (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
        # - r_i > 0
        
        def _neg_sum_r(z):
            return -np.sum(z[2::3])
        
        def _constraints(z):
            n = len(z) // 3
            x = z[0::3]
            y = z[1::3]
            r = z[2::3]
            
            cons = []
            
            # Rectangle boundaries
            for i in range(n):
                cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i] - z[3*i+2]})  # x_i - r_i >= 0
                cons.append({'type': 'ineq', 'fun': lambda z, i=i: w - z[3*i] - z[3*i+2]})  # x_i + r_i <= w
                cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i+1] - z[3*i+2]})  # y_i - r_i >= 0
                cons.append({'type': 'ineq', 'fun': lambda z, i=i: h - z[3*i+1] - z[3*i+2]})  # y_i + r_i <= h
            
            # Non-overlapping
            for i in range(n):
                for j in range(i+1, n):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda z, i=i, j=j: (z[3*i]-z[3*j])**2 + (z[3*i+1]-z[3*j+1])**2 - (z[3*i+2]+z[3*j+2])**2
                    })
            
            return cons
        
        # Initial guess: distribute circles in the rectangle
        n = N
        side = int(np.ceil(np.sqrt(n)))
        z0 = np.zeros(3*n)
        for i in range(n):
            row = i // side
            col = i % side
            z0[3*i] = (col + 0.5) * w / side
            z0[3*i+1] = (row + 0.5) * h / side
            z0[3*i+2] = 0.5 * min(w, h) / side * 0.9
        
        # Scale initial radii
        max_r = min(w, h) / (2 * side)
        for i in range(n):
            z0[3*i+2] = max_r * 0.9
        
        res = minimize(
            _neg_sum_r,
            z0,
            method='SLSQP',
            constraints=_constraints(z0),
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        if res.success:
            arr = res.x.reshape(n, 3)
            # Translate to make bounding box start at origin
            xmin = np.min(arr[:, 0] - arr[:, 2])
            ymin = np.min(arr[:, 1] - arr[:, 2])
            arr[:, 0] -= xmin
            arr[:, 1] -= ymin
            return arr
        return None
    
    # Try different aspect ratios
    # Perimeter 2*(w+h) <= 4 => w+h <= 2
    # Let w = t, h = 2-t, t in (0, 2)
    aspect_ratios = np.linspace(0.5, 1.5, 11)  # w/h ratio
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        # w/h = ar, w+h = 2 => w = 2*ar/(1+ar), h = 2/(1+ar)
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        arr_res = _optimize_in_rect(w, h, arr0, deadline)
        if arr_res is not None:
            sc = _score(arr_res)
            if sc > best_score:
                best_score = sc
                best_arr = arr_res.copy()
    
    # Final polish: if we have a valid arrangement, try to improve it
    if best_arr is not None and time.time() < deadline - 5:
        # Try to optimize the current best arrangement
        def _neg_sum_r_final(z):
            return -np.sum(z[2::3])
        
        def _constraints_final(z):
            n = len(z) // 3
            x = z[0::3]
            y = z[1::3]
            r = z[2::3]
            
            cons = []
            
            # Perimeter constraint using smooth approximation
            # w = max(x+r) - min(x-r)
            # Use log-sum-exp for smooth max/min
            alpha = 50  # sharpness
            
            def smooth_max(a):
                return (1/alpha) * np.log(np.sum(np.exp(alpha * a)))
            
            def smooth_min(a):
                return -(1/alpha) * np.log(np.sum(np.exp(-alpha * a)))
            
            w = smooth_max(x + r) - smooth_min(x - r)
            h = smooth_max(y + r) - smooth_min(y - r)
            
            # Perimeter constraint: 2*(w+h) <= 4
            cons.append({'type': 'ineq', 'fun': lambda: 4 - 2*(w + h)})
            
            # Non-overlapping
            for i in range(n):
                for j in range(i+1, n):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda z, i=i, j=j: (z[3*i]-z[3*j])**2 + (z[3*i+1]-z[3*j+1])**2 - (z[3*i+2]+z[3*j+2])**2
                    })
            
            # Radius positivity
            for i in range(n):
                cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i+2] - 1e-8})
            
            return cons
        
        z0 = best_arr.flatten()
        res = minimize(
            _neg_sum_r_final,
            z0,
            method='SLSQP',
            constraints=_constraints_final(z0),
            options={'maxiter': 50, 'ftol': 1e-10}
        )
        
        if res.success:
            arr = res.x.reshape(N, 3)
            sc = _score(arr)
            if sc > best_score:
                best_score = sc
                best_arr = arr.copy()
    
    if best_arr is not None:
        return best_arr
    
    # Fallback: return the initial grid
    return arr0
# EVOLVE-BLOCK-END
