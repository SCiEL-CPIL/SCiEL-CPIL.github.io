import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer for final validation and return
    
    # Pre-compute indices for pairwise distances for vectorization
    i_indices, j_indices = np.triu_indices(N, k=1)
    num_pairs = len(i_indices)
    
    def get_state(params):
        """
        params: [W, x1..xN, y1..yN, r1..rN]
        Returns W, H, x, y, r
        """
        W = params[0]
        H = 2.0 - W
        x = params[1:1+N]
        y = params[1+N:1+2*N]
        r = params[1+2*N:1+3*N]
        return W, H, x, y, r

    def compute_cost_and_grad(params):
        """
        Computes the cost and its gradient using softplus penalties.
        Cost = -sum(r) + weight * (softplus(boundary_viol) + softplus(overlap_viol))
        """
        W, H, x, y, r = get_state(params)
        
        if W <= 0 or H <= 0:
            return 1e10, np.zeros_like(params)
        
        # Softplus: log(1 + exp(x)) approximated smoothly
        beta = 10.0  # Larger = closer to max(0,x)
        
        # 1. Boundary violations (smooth)
        # Left: x < r -> viol = r - x
        # Right: x > W-r -> viol = x - (W-r)
        # Bottom: y < r -> viol = r - y
        # Top: y > H-r -> viol = y - (H-r)
        
        viol_left = r - x
        viol_right = x - (W - r)
        viol_bottom = r - y
        viol_top = y - (H - r)
        
        # Softplus of each violation
        def softplus(v):
            return np.where(v > 0, v + np.log1p(np.exp(-v)), np.log1p(np.exp(v)))
        
        sp_left = softplus(viol_left)
        sp_right = softplus(viol_right)
        sp_bottom = softplus(viol_bottom)
        sp_top = softplus(viol_top)
        
        boundary_pen = np.sum(sp_left) + np.sum(sp_right) + np.sum(sp_bottom) + np.sum(sp_top)
        
        # 2. Overlap penalties (smooth)
        dx = x[i_indices] - x[j_indices]
        dy = y[i_indices] - y[j_indices]
        dists_sq = dx**2 + dy**2
        dists = np.sqrt(dists_sq + 1e-12)  # Avoid division by zero
        
        min_dists = r[i_indices] + r[j_indices]
        
        # Overlap violation: min_dist - dist
        overlap_viol = min_dists - dists
        sp_overlap = softplus(overlap_viol)
        
        overlap_pen = np.sum(sp_overlap)
        
        # 3. Objective
        weight = 50.0
        total_pen = weight * (boundary_pen + overlap_pen)
        
        cost = -np.sum(r) + total_pen
        
        # ---- Gradient computation ----
        grad = np.zeros_like(params)
        
        # d/dW from boundary terms
        # viol_right = x - W + r, d(viol_right)/dW = -1
        # sp'(v) = sigmoid(v) = 1/(1+exp(-v))
        def sigmoid(v):
            return np.where(v > 0, 1.0/(1.0+np.exp(-v)), np.exp(v)/(1.0+np.exp(v)))
        
        sig_right = sigmoid(viol_right)
        grad[0] = -weight * np.sum(sig_right)  # d(boundary_pen)/dW
        
        # d/dx_i
        for i in range(N):
            # Left boundary
            vl = viol_left[i]
            sig_l = sigmoid(vl)
            # Right boundary
            vr = viol_right[i]
            sig_r = sigmoid(vr)
            # Overlap contributions
            overlap_grad_x = np.zeros(N)
            for k in range(num_pairs):
                ii, jj = i_indices[k], j_indices[k]
                if ii == i or jj == i:
                    other = jj if ii == i else ii
                    dxi = x[i] - x[other]
                    dyi = y[i] - y[other]
                    di = dists[k]
                    if di > 1e-10:
                        ov = overlap_viol[k]
                        sig_ov = sigmoid(ov)
                        # d(overlap_viol)/dx_i = -d(dist)/dx_i = -(dx_i/dx_i) = -1 for the x component
                        # Actually: overlap_viol = r_i + r_j - dist
                        # d(overlap_viol)/dx_i = -d(dist)/dx_i = -(x_i - x_j)/dist = -dx/di
                        overlap_grad_x[i] += -sig_ov * (-dxi / di)
            
            grad[1+i] = weight * (sig_l * 1 + sig_r * 1 + overlap_grad_x[i])
            # Wait: d(sp_left)/dx_i = sig(vl) * d(vl)/dx_i = sig_l * (-1)
            # d(sp_right)/dx_i = sig(vr) * d(vr)/dx_i = sig_r * (1)
            # So: grad[1+i] = -weight*sig_l + weight*sig_r + weight*overlap_grad_x[i]
            # Let me redo:
            
        # Let me recompute gradient more carefully
        grad = np.zeros_like(params)
        
        # d(cost)/dW
        # cost = -sum(r) + weight*(boundary_pen + overlap_pen)
        # boundary_pen depends on W through viol_right = x - (W-r) = x - W + r
        # d(viol_right)/dW = -1
        # d(sp)/dW = sigmoid(viol_right) * (-1)
        # So d(boundary_pen)/dW = -sum(sigmoid(viol_right))
        sig_right = sigmoid(viol_right)
        grad[0] = -weight * np.sum(sig_right)
        
        # d(cost)/dx_i for each circle i
        # Boundary:
        #   sp_left: d/dx_i = sigmoid(viol_left_i) * d(viol_left_i)/dx_i = sig_l_i * (-1)
        #   sp_right: d/dx_i = sigmoid(viol_right_i) * d(viol_right_i)/dx_i = sig_r_i * (1)
        # Overlap:
        #   For each pair (i,j) containing i:
        #   overlap_viol = r_i + r_j - dist_ij
        #   d(overlap_viol)/dx_i = -d(dist_ij)/dx_i = -(x_i - x_j)/dist_ij
        #   d(sp)/dx_i = sigmoid(overlap_viol) * (-(x_i - x_j)/dist_ij)
        
        sig_left = sigmoid(viol_left)
        sig_right = sigmoid(viol_right)
        
        for i in range(N):
            # Boundary contribution to dx_i
            gx_i = -weight * sig_left[i] + weight * sig_right[i]
            
            # Overlap contribution
            for k in range(num_pairs):
                ii, jj = i_indices[k], j_indices[k]
                if ii == i:
                    other = jj
                    dxi = x[i] - x[other]
                    di = dists[k]
                    if di > 1e-10:
                        ov = overlap_viol[k]
                        sig_ov = sigmoid(ov)
                        gx_i += weight * sig_ov * (-dxi / di)
                elif jj == i:
                    other = ii
                    dxi = x[i] - x[other]
                    di = dists[k]
                    if di > 1e-10:
                        ov = overlap_viol[k]
                        sig_ov = sigmoid(ov)
                        gx_i += weight * sig_ov * (-dxi / di)
            
            grad[1+i] = gx_i
        
        # d(cost)/dy_i
        sig_bottom = sigmoid(viol_bottom)
        sig_top = sigmoid(viol_top)
        
        for i in range(N):
            gy_i = -weight * sig_bottom[i] + weight * sig_top[i]
            
            for k in range(num_pairs):
                ii, jj = i_indices[k], j_indices[k]
                if ii == i:
                    other = jj
                    dyi = y[i] - y[other]
                    di = dists[k]
                    if di > 1e-10:
                        ov = overlap_viol[k]
                        sig_ov = sigmoid(ov)
                        gy_i += weight * sig_ov * (-dyi / di)
                elif jj == i:
                    other = ii
                    dyi = y[i] - y[other]
                    di = dists[k]
                    if di > 1e-10:
                        ov = overlap_viol[k]
                        sig_ov = sigmoid(ov)
                        gy_i += weight * sig_ov * (-dyi / di)
            
            grad[1+N+i] = gy_i
        
        # d(cost)/dr_i
        # -sum(r): d/dr_i = -1
        # Boundary:
        #   viol_left = r - x: d/dr_i = 1
        #   viol_right = x - W + r: d/dr_i = 1
        #   viol_bottom = r - y: d/dr_i = 1
        #   viol_top = y - H + r: d/dr_i = 1
        # Overlap:
        #   For pairs containing i: overlap_viol = r_i + r_j - dist
        #   d(overlap_viol)/dr_i = 1
        #   d(sp)/dr_i = sigmoid(overlap_viol) * 1
        
        for i in range(N):
            gr_i = -1.0  # from -sum(r)
            
            # Boundary
            gr_i += weight * sigmoid(viol_left[i])    # d(sp_left)/dr_i
            gr_i += weight * sigmoid(viol_right[i])   # d(sp_right)/dr_i
            gr_i += weight * sigmoid(viol_bottom[i])  # d(sp_bottom)/dr_i
            gr_i += weight * sigmoid(viol_top[i])     # d(sp_top)/dr_i
            
            # Overlap
            for k in range(num_pairs):
                ii, jj = i_indices[k], j_indices[k]
                if ii == i or jj == i:
                    ov = overlap_viol[k]
                    gr_i += weight * sigmoid(ov)
            
            grad[1+2*N+i] = gr_i
        
        return cost, grad

    def compute_cost(params):
        """Wrapper for cost only."""
        cost, _ = compute_cost_and_grad(params)
        return cost

    def is_valid(params, tol=1e-6):
        """
        Strict validity check for final result.
        """
        W, H, x, y, r = get_state(params)
        if W <= 0 or H <= 0:
            return False
        if np.any(r < tol):
            return False
        
        # Check boundaries
        if np.any(x < r - tol) or np.any(x > W - r + tol) or np.any(y < r - tol) or np.any(y > H - r + tol):
            return False
            
        # Check overlaps
        dx = x[i_indices] - x[j_indices]
        dy = y[i_indices] - y[j_indices]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = r[i_indices] + r[j_indices]
        
        if np.any(dists < min_dists - tol):
            return False
            
        return True

    def polish(params, time_limit=20.0):
        """
        Local optimization using L-BFGS-B with analytic gradient.
        """
        start_p = time.time()
        
        def objective(params):
            if time.time() - start_p > time_limit:
                raise TimeoutError()
            cost, _ = compute_cost_and_grad(params)
            return cost
        
        def gradient(params):
            if time.time() - start_p > time_limit:
                raise TimeoutError()
            _, grad = compute_cost_and_grad(params)
            return grad
        
        # Define bounds
        bounds = [(0.4, 1.6)]
        bounds += [(0, 1.6)] * N  # x
        bounds += [(0, 1.6)] * N  # y
        bounds += [(0.001, 1.0)] * N  # r
        
        try:
            res = minimize(objective, params, jac=gradient, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000})
            
            if time.time() - start_p > time_limit:
                return params
            
            if res.fun < compute_cost(params):
                if is_valid(res.x):
                    return res.x
                else:
                    return params
            else:
                return params
        except (TimeoutError, Exception):
            return params

    def run_sa(initial_params, time_budget):
        """
        Run Simulated Annealing with adaptive step size.
        """
        start_t = time.time()
        T = 0.15
        T_min = 1e-6
        cooling_rate = 0.998
        
        best_params = initial_params.copy()
        best_cost = compute_cost(best_params)
        current_params = initial_params.copy()
        current_cost = best_cost
        
        n_params = len(current_params)
        iterations = 0
        
        while time.time() - start_t < time_budget:
            iterations += 1
            
            # Adaptive step size based on T
            step_size = 0.03 * T + 0.002
            new_params = current_params + np.random.randn(n_params) * step_size
            
            # Keep W within reasonable bounds
            new_params[0] = np.clip(new_params[0], 0.4, 1.6)
            # Keep radii positive
            new_params[1+2*N:] = np.maximum(0.001, new_params[1+2*N:])
            
            new_cost = compute_cost(new_params)
            
            # Acceptance criterion
            delta = new_cost - current_cost
            if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
                current_params = new_params
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_params = current_params.copy()
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best_params

    # Initializations
    inits = []
    
    # Init 1: Hexagonal lattice in 1.4 x 0.6 rectangle
    W1 = 1.4
    H1 = 0.6
    # Hexagonal packing: rows offset by half spacing
    # For H=0.6, we can fit ~2 rows of circles with radius ~0.15
    # Let's try: 2 rows, 11 and 10 circles
    x1 = np.zeros(N)
    y1 = np.zeros(N)
    r1 = np.full(N, 0.13)
    
    # Row 0: 11 circles
    spacing_x = W1 / 11
    for i in range(11):
        x1[i] = (i + 0.5) * spacing_x
        y1[i] = H1 / 3  # Lower row
        r1[i] = 0.12
    
    # Row 1: 10 circles, offset
    spacing_x2 = W1 / 10
    for i in range(10):
        x1[11+i] = (i + 0.5) * spacing_x2
        y1[11+i] = 2 * H1 / 3  # Upper row
        r1[11+i] = 0.12
    
    inits.append(np.concatenate(([W1], x1, y1, r1)))
    
    # Init 2: Grid 7x3 (aspect ratio 1.4/0.6)
    W2 = 1.4
    H2 = 0.6
    x2 = np.zeros(N)
    y2 = np.zeros(N)
    r2 = np.full(N, 0.09)
    for i in range(N):
        col = i % 7
        row = i // 7
        x2[i] = (col + 0.5) * W2 / 7
        y2[i] = (row + 0.5) * H2 / 3
    inits.append(np.concatenate(([W2], x2, y2, r2)))
    
    # Init 3: Square grid 5x5 (aspect ratio 1.0/1.0)
    W3 = 1.0
    H3 = 1.0
    x3 = np.zeros(N)
    y3 = np.zeros(N)
    r3 = np.full(N, 0.10)
    for i in range(N):
        col = i % 5
        row = i // 5
        x3[i] = (col + 0.5) * W3 / 5
        y3[i] = (row + 0.5) * H3 / 5
    inits.append(np.concatenate(([W3], x3, y3, r3)))
    
    # Init 4: Hexagonal in 1.2 x 0.8
    W4 = 1.2
    H4 = 0.8
    x4 = np.zeros(N)
    y4 = np.zeros(N)
    r4 = np.full(N, 0.11)
    
    # 3 rows: 7, 7, 7
    for row in range(3):
        for col in range(7):
            i = row * 7 + col
            x4[i] = (col + 0.5) * W4 / 7
            y4[i] = (row + 0.5) * H4 / 3
            if row % 2 == 1:
                x4[i] += W4 / 14  # Offset
    inits.append(np.concatenate(([W4], x4, y4, r4)))
    
    # Add perturbed inits
    np.random.seed(42)
    for _ in range(4):
        base = inits[np.random.randint(len(inits))]
        noise = np.random.randn(len(base)) * 0.03
        new_init = base + noise
        new_init[0] = np.clip(new_init[0], 0.5, 1.5)
        new_init[1+2*N:] = np.maximum(0.01, new_init[1+2*N:])
        inits.append(new_init)
        
    best_overall_params = None
    best_overall_val = -np.inf
    
    # Time management
    num_restarts = len(inits)
    sa_time_per_restart = 8.0
    
    for i in range(num_restarts):
        if time.time() > deadline - 40.0:
            break
            
        current_init = inits[i]
        
        # Run SA
        sa_best = run_sa(current_init, sa_time_per_restart)
        
        # Quick polish on SA result
        if time.time() < deadline - 30.0:
            polished = polish(sa_best, time_limit=5.0)
            if is_valid(polished):
                sa_best = polished
        
        if is_valid(sa_best):
            val = np.sum(sa_best[1+2*N:])
            if val > best_overall_val:
                best_overall_val = val
                best_overall_params = sa_best.copy()
        else:
            if best_overall_params is None:
                best_overall_params = sa_best.copy()
                best_overall_val = -np.inf
                
    # Final polish phase with more time
    if best_overall_params is not None:
        polish_start = time.time()
        polish_budget = min(30.0, deadline - time.time() - 5.0)
        
        if polish_budget > 5.0:
            polished_params = polish(best_overall_params, time_limit=polish_budget)
            
            if is_valid(polished_params):
                val = np.sum(polished_params[1+2*N:])
                if val > best_overall_val:
                    best_overall_val = val
                    best_overall_params = polished_params
            
            # Second polish attempt with tighter settings if time allows
            if time.time() < deadline - 10.0:
                # Try a second polish from slightly perturbed best
                perturbed = best_overall_params + np.random.randn(len(best_overall_params)) * 0.005
                perturbed[0] = np.clip(perturbed[0], 0.4, 1.6)
                perturbed[1+2*N:] = np.maximum(0.001, perturbed[1+2*N:])
                polished2 = polish(perturbed, time_limit=min(15.0, deadline - time.time() - 5.0))
                if is_valid(polished2):
                    val2 = np.sum(polished2[1+2*N:])
                    if val2 > best_overall_val:
                        best_overall_val = val2
                        best_overall_params = polished2

    # Final Fallback
    if best_overall_params is None or not is_valid(best_overall_params):
        W = 1.4
        H = 0.6
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, 0.08)
        for i in range(N):
            col = i % 7
            row = i // 7
            x[i] = (col + 0.5) * W / 7
            y[i] = (row + 0.5) * H / 3
        best_overall_params = np.concatenate(([W], x, y, r))
        if not is_valid(best_overall_params):
            r = np.full(N, 0.05)
            best_overall_params[1+2*N:] = r
             
    # Final validation and formatting
    W, H, x, y, r = get_state(best_overall_params)
    
    # Safety net: shrink if invalid
    if not is_valid(best_overall_params, tol=1e-7):
        r = r * 0.9
        best_overall_params[1+2*N:] = r
        W, H, x, y, r = get_state(best_overall_params)
        
    result = np.column_stack((x, y, r))
    
    return result

if __name__ == "__main__":
    res = construct()
    print(res)
