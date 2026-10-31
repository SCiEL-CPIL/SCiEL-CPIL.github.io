import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial.distance import cdist

N = 26

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Helper: Solve LP to maximize radii for fixed positions
    def solve_radii_lp(x, y):
        """
        Maximize sum(r_i)
        Subject to:
        r_i + r_j <= d_ij  (for all i < j)
        r_i <= x_i         (left)
        r_i <= 1 - x_i     (right)
        r_i <= y_i         (bottom)
        r_i <= 1 - y_i     (top)
        r_i >= 0
        """
        c = -np.ones(N) # Minimize -sum(r)
        
        A_ub = []
        b_ub = []
        
        # Pairwise constraints: r_i + r_j <= d_ij
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                row = np.zeros(N)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(d)
        
        # Boundary constraints: r_i <= dist_to_wall
        for i in range(N):
            # Left: x_i
            row = np.zeros(N); row[i] = 1.0
            A_ub.append(row); b_ub.append(x[i])
            # Right: 1 - x_i
            A_ub.append(row); b_ub.append(1.0 - x[i])
            # Bottom: y_i
            A_ub.append(row); b_ub.append(y[i])
            # Top: 1 - y_i
            A_ub.append(row); b_ub.append(1.0 - y[i])
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(0, None)] * N
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.x
            else:
                return np.zeros(N) # Fallback
        except Exception:
            return np.zeros(N)

    def is_valid(x, y, r, tol=1e-9):
        # Check bounds
        if np.any(x < r - tol) or np.any(x > 1 - r + tol): return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol): return False
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j] - tol:
                    return False
        return True

    def get_score(x, y, r):
        if not is_valid(x, y, r, tol=1e-8):
            return -np.inf
        return np.sum(r)

    best_score = -np.inf
    best_x = None
    best_y = None
    best_r = None

    # Initializations
    inits = []
    
    # 1. Hexagonal Grid
    # Approximate hexagonal packing
    # Rows of circles.
    # Let's try a 5x5 grid with some jitter or specific hex layout.
    # A simple hex grid:
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # ...
    # Actually, for N=26, a 5x5 grid (25) + 1 is hard to make hex.
    # Let's use a 5x5 grid and one in the center?
    # Or just 4 rows of 7? 4*7=28. Remove 2.
    # Or 5 rows of 6? 5*6=30. Remove 4.
    
    # Let's generate a few random and structured starts.
    
    # Hexagonal-ish start
    # Radius approx 0.1
    # Spacing approx 0.2
    # 5 columns, 5 rows = 25.
    # x = 0.1, 0.3, 0.5, 0.7, 0.9
    # y = 0.1, 0.3, 0.5, 0.7, 0.9
    # This is square grid.
    
    # Let's use random starts and one square grid.
    rng = np.random.default_rng(42)
    
    # Square Grid Start
    side = int(np.ceil(np.sqrt(N))) # 5
    x_grid = np.zeros(N)
    y_grid = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N: break
            x_grid[idx] = (i + 0.5) / side
            y_grid[idx] = (j + 0.5) / side
            idx += 1
    inits.append((x_grid, y_grid))
    
    # Random Starts
    for _ in range(20):
        x_r = rng.uniform(0.2, 0.8, N)
        y_r = rng.uniform(0.2, 0.8, N)
        inits.append((x_r, y_r))
        
    # Corner Biased Starts (circles often want to be in corners)
    for _ in range(5):
        x_c = rng.uniform(0.1, 0.9, N)
        y_c = rng.uniform(0.1, 0.9, N)
        # Force some to corners?
        # Just random for now, SLSQP will move them.
        inits.append((x_c, y_c))

    # SLSQP Objective and Constraints
    def objective(v):
        # v = [x1, y1, r1, x2, y2, r2, ...]
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        return -np.sum(r)
    
    def constraints(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        cons = []
        
        # Overlap constraints: dist - r_i - r_j >= 0
        # We can return a single array for all inequality constraints in SLSQP
        # But SLSQP expects a dict or a function. 
        # Using a function that returns a list of dicts is slow.
        # Better to use the 'ineq' type in a list of dicts? 
        # Or use a single function for all?
        # scipy.optimize.minimize with method='SLSQP' accepts:
        # constraints: [
        #   {'type': 'ineq', 'fun': fun1},
        #   ...
        # ]
        # This is slow if we have 400 constraints.
        
        # Alternative: Use a penalty method with L-BFGS-B but with a much stronger penalty and careful tuning?
        # Or use the 'COBYLA' method which handles constraints but is derivative-free and slow.
        
        # Let's stick to L-BFGS-B with a very well-tuned penalty, as SLSQP with 400 constraints might be too slow or unstable in Python.
        # However, the LP post-processing is the key improvement.
        
        pass # Not using SLSQP directly due to constraint overhead.

    # Revised Strategy:
    # 1. Use L-BFGS-B to optimize positions x, y with a penalty for overlaps, BUT initialize radii to be small.
    # 2. After each L-BFGS-B step (or at the end), run the LP to get max radii.
    # 3. Actually, we can optimize x, y only, and calculate r via LP inside the objective?
    #    No, LP is not differentiable.
    
    # Best Strategy:
    # 1. Multi-start L-BFGS-B on (x, y, r) with penalty.
    # 2. Post-process each result with LP.
    # 3. Keep the best.
    # 4. Additionally, try a "Position Only" optimization where we fix r from LP and move x,y?
    #    If we fix r, the constraints are hard. We can use SLSQP on x,y only with hard constraints:
    #    dist_ij >= r_i + r_j.
    #    This is 325 constraints. Still a lot.
    
    # Let's go with the robust L-BFGS-B + LP post-processing, but improve the penalty formulation and initializations.
    
    def loss_and_grad(v, lam_ov=50.0, lam_box=50.0, delta=1e-4):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        loss = -np.sum(r)
        grad = np.zeros_like(v)
        grad[2::3] -= 1.0
        
        # Overlap
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx*dx + dy*dy
                d = np.sqrt(d_sq + delta*delta)
                
                diff = r[i] + r[j] - d
                if diff > 0:
                    s = np.sqrt(diff*diff + delta*delta)
                    p = 0.5 * (diff + s)
                    p_sq = p * p
                    loss += lam_ov * p_sq
                    
                    dp_ddiff = 0.5 * (1.0 + diff / s)
                    d_p_sq = 2.0 * p * dp_ddiff
                    lam_d_p_sq = lam_ov * d_p_sq
                    
                    grad[2*i] += lam_d_p_sq
                    grad[2*j] += lam_d_p_sq
                    
                    dx_dd = dx / d
                    dy_dd = dy / d
                    
                    grad[i] += lam_d_p_sq * (-dx_dd)
                    grad[i+1] += lam_d_p_sq * (-dy_dd)
                    grad[j] += lam_d_p_sq * (dx_dd)
                    grad[j+1] += lam_d_p_sq * (dy_dd)
        
        # Boundaries
        for i in range(N):
            # Left
            diff_l = r[i] - x[i]
            if diff_l > 0:
                s = np.sqrt(diff_l*diff_l + delta*delta)
                p = 0.5 * (diff_l + s)
                p_sq = p * p
                loss += lam_box * p_sq
                dp_ddiff = 0.5 * (1.0 + diff_l / s)
                d_p_sq = 2.0 * p * dp_ddiff
                lam_d_p_sq = lam_box * d_p_sq
                grad[2*i] += lam_d_p_sq
                grad[i] -= lam_d_p_sq
            # Right
            diff_r = r[i] + x[i] - 1.0
            if diff_r > 0:
                s = np.sqrt(diff_r*diff_r + delta*delta)
                p = 0.5 * (diff_r + s)
                p_sq = p * p
                loss += lam_box * p_sq
                dp_ddiff = 0.5 * (1.0 + diff_r / s)
                d_p_sq = 2.0 * p * dp_ddiff
                lam_d_p_sq = lam_box * d_p_sq
                grad[2*i] += lam_d_p_sq
                grad[i] += lam_d_p_sq
            # Bottom
            diff_b = r[i] - y[i]
            if diff_b > 0:
                s = np.sqrt(diff_b*diff_b + delta*delta)
                p = 0.5 * (diff_b + s)
                p_sq = p * p
                loss += lam_box * p_sq
                dp_ddiff = 0.5 * (1.0 + diff_b / s)
                d_p_sq = 2.0 * p * dp_ddiff
                lam_d_p_sq = lam_box * d_p_sq
                grad[2*i] += lam_d_p_sq
                grad[i+1] -= lam_d_p_sq
            # Top
            diff_t = r[i] + y[i] - 1.0
            if diff_t > 0:
                s = np.sqrt(diff_t*diff_t + delta*delta)
                p = 0.5 * (diff_t + s)
                p_sq = p * p
                loss += lam_box * p_sq
                dp_ddiff = 0.5 * (1.0 + diff_t / s)
                d_p_sq = 2.0 * p * dp_ddiff
                lam_d_p_sq = lam_box * d_p_sq
                grad[2*i] += lam_d_p_sq
                grad[i+1] += lam_d_p_sq
        
        return loss, grad

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    for x0, y0 in inits:
        if time.time() - start_time > time_limit:
            break
            
        # Initialize radii small to avoid immediate huge penalties
        r0 = np.full(N, 0.05)
        
        v0 = np.zeros(3*N)
        v0[0::3] = x0
        v0[1::3] = y0
        v0[2::3] = r0
        
        # Try a few lambda values
        for lam in [10.0, 50.0, 100.0, 500.0]:
            if time.time() - start_time > time_limit:
                break
                
            def obj(v):
                l, g = loss_and_grad(v, lam_ov=lam, lam_box=lam)
                return l
            
            def grad(v):
                l, g = loss_and_grad(v, lam_ov=lam, lam_box=lam)
                return g
                
            try:
                res = minimize(obj, v0, jac=grad, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-7, 'maxls': 20})
            except Exception:
                continue
                
            v_res = res.x
            x_res = v_res[0::3]
            y_res = v_res[1::3]
            r_res = v_res[2::3]
            
            # Post-process: Solve LP for max radii given these positions
            r_lp = solve_radii_lp(x_res, y_res)
            
            # Check if LP result is valid (it should be, but numerical errors can happen)
            if is_valid(x_res, y_res, r_lp):
                score = np.sum(r_lp)
                if score > best_score:
                    best_score = score
                    best_x = x_res.copy()
                    best_y = y_res.copy()
                    best_r = r_lp.copy()
                    
    # Final Polish: If we have a good result, try to optimize positions again 
    # using the LP radii as a guide? 
    # Or just run one more SLSQP on positions with hard constraints from best_r?
    # Let's try a final SLSQP on positions x,y with hard constraints using best_r as lower bounds for radii?
    # No, radii can change.
    
    # Let's just return the best found.
    
    if best_x is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        x = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)][:N])
        y = np.array([(j + 0.5) / side for _ in range(side) for j in range(side)][:N])
        r = solve_radii_lp(x, y)
        return np.column_stack((x, y, r))
        
    result = np.column_stack((best_x, best_y, best_r))
    return result

