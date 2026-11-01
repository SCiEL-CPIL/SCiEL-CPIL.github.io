import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 32

def phyllotaxis_initialization(n):
    """Generate phyllotaxis-based initial positions for n circles."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    points = []
    for i in range(n):
        r = 0.45 * np.sqrt((i + 0.5) / n)
        theta = i * golden_angle
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        points.append([x, y, 0.05])
    return np.array(points)

def hex_initialization(n):
    """Generate hexagonal lattice-based initial positions for n circles."""
    points = []
    spacing = 0.15
    row = 0
    while len(points) < n:
        for col in range(8):
            x = 0.1 + col * spacing + (0.075 if row % 2 else 0.0)
            y = 0.1 + row * spacing * 0.866
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append([x, y, 0.05])
        row += 1
    return np.array(points[:n])

def coordinate_descent_polish(x, y, r, max_iter=200, tol=1e-10):
    """Refine packing using coordinate descent on positions and radii."""
    n = len(r)
    for _ in range(max_iter):
        improved = False
        
        # Try adjusting each radius slightly
        for i in range(n):
            for delta in [0.001, 0.005, 0.01, -0.001, -0.005, -0.01]:
                r_new = r.copy()
                r_new[i] = max(1e-4, r_new[i] + delta)
                if _is_valid(x, y, r_new):
                    if r_new[i] > r[i]:
                        r = r_new
                        improved = True
        
        # Try adjusting each position
        for i in range(n):
            for dx in [-0.01, -0.005, 0, 0.005, 0.01]:
                for dy in [-0.01, -0.005, 0, 0.005, 0.01]:
                    if dx == 0 and dy == 0:
                        continue
                    x_new = x.copy()
                    y_new = y.copy()
                    x_new[i] = x[i] + dx
                    y_new[i] = y[i] + dy
                    if _is_valid(x_new, y_new, r):
                        # Check if this improves the minimum clearance
                        if _min_clearance(x_new, y_new, r) > _min_clearance(x, y, r) - tol:
                            x = x_new
                            y = y_new
                            improved = True
        
        if not improved:
            break
    
    return x, y, r

def _is_valid(x, y, r, tol=1e-9):
    """Check if packing is valid (all circles inside unit square and non-overlapping)."""
    n = len(r)
    for i in range(n):
        if x[i] < r[i] - tol or x[i] > 1 - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > 1 - r[i] + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _min_clearance(x, y, r):
    """Compute minimum clearance in the packing."""
    n = len(r)
    min_c = np.inf
    for i in range(n):
        min_c = min(min_c, x[i] - r[i], 1 - x[i] - r[i], y[i] - r[i], 1 - y[i] - r[i])
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            min_c = min(min_c, dist - r[i] - r[j])
    return min_c

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # Initial solutions
    # 1. Grid 6x6 (using 32)
    side = 6
    grid = []
    for i in range(side):
        for j in range(side):
            grid.append([ (i+0.5)/side, (j+0.5)/side, 0.5/side ])
    grid = np.array(grid[:N])
    
    # 2. Phyllotaxis
    phyl = phyllotaxis_initialization(N)
    
    # 3. Hexagonal
    hex_init = hex_initialization(N)
    
    best_result = None
    best_score = -np.inf

    # Define objective function
    def objective(params, beta=100.0):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Radii must be positive
        # If r < 0, penalty
        r_clipped = np.maximum(r, 1e-6)
        
        # Calculate clearances
        # Boundary clearances
        # min(x, y, 1-x, 1-y) - r
        # We can approximate min with softmin too, or just use the raw min for gradients?
        # Raw min is not differentiable.
        # Let's use softmin for boundary clearances as well.
        
        # Vectorized distance calculations
        # This is O(N^2). N=32 is small.
        
        # 1. Boundary constraints
        # x_i - r_i, 1-x_i - r_i, y_i - r_i, 1-y_i - r_i
        # We want min of these 4 for each i.
        # Let's compute all pairwise circle-circle clearances and boundary clearances.
        
        # Circle-Circle
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        
        # Clearances c_ij = dist_ij - r_i - r_j
        r_sum = r_clipped[:, None] + r_clipped[None, :]
        c_cc = dist - r_sum
        
        # Boundary Clearances
        # For each i, we have 4 values.
        # c_bx1 = x - r
        # c_bx2 = 1 - x - r
        # c_by1 = y - r
        # c_by2 = 1 - y - r
        c_b1 = x - r_clipped
        c_b2 = 1 - x - r_clipped
        c_b3 = y - r_clipped
        c_b4 = 1 - y - r_clipped
        
        # Combine all clearances into a single vector for softmin
        # Flattened c_cc (excluding diagonal)
        c_cc_flat = c_cc[~np.eye(N, dtype=bool)]
        
        c_all = np.concatenate([c_cc_flat, c_b1, c_b2, c_b3, c_b4])
        
        # Softmin
        # To avoid overflow/underflow, subtract max
        # softmin(c) = -1/beta * log(sum(exp(-beta * c)))
        # Let M = min(c). Then c - M >= 0.
        # exp(-beta * (c - M)) = exp(-beta * c) * exp(beta * M)
        # sum(exp(-beta * c)) = exp(-beta * M) * sum(exp(-beta * (c - M)))
        # log(sum(...)) = -beta * M + log(sum(exp(-beta * (c - M))))
        # softmin = -1/beta * (-beta * M + log(...)) = M - 1/beta * log(sum(exp(-beta * (c - M))))
        
        M = np.min(c_all)
        # If M is very large positive, exp(-beta*(c-M)) is fine.
        # If M is very negative, c-M is small.
        
        # Numerical stability
        # We want to maximize the minimum clearance.
        # If beta is high, softmin approximates min.
        
        # Calculate gradient of softmin w.r.t c
        # d(softmin)/d(c_k) = exp(-beta * (c_k - M)) / sum(exp(-beta * (c_j - M)))
        
        exp_vals = np.exp(-beta * (c_all - M))
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        softmin_val = M - (1.0 / beta) * np.log(sum_exp)
        
        # Objective: -sum(r) - lambda * softmin_val
        # We want to maximize sum(r) and maximize softmin_val (min clearance).
        # So minimize -sum(r) - lambda * softmin_val
        # lambda should be large enough to enforce constraints.
        # Let's use a penalty approach or just a large weight.
        # Actually, if we just minimize -sum(r), the circles will expand until they touch.
        # The softmin term ensures they don't overlap.
        
        # Let's use a fixed lambda.
        lam = 10.0
        
        obj = -np.sum(r_clipped) - lam * softmin_val
        
        # Gradient
        grad = np.zeros(3*N)
        
        # d(obj)/d(r) = -1 - lam * d(softmin)/d(r)
        # d(softmin)/d(c_k) = weights[k]
        # d(c_k)/d(r_i) depends on which clearance c_k is.
        
        # This gradient calculation is tricky with flattened arrays.
        # Let's compute gradients for c_all first, then backprop.
        
        # d(softmin)/d(c_all) = weights
        # c_all = [c_cc_flat, c_b1, c_b2, c_b3, c_b4]
        
        # 1. Backprop to c_cc_flat
        # c_cc_flat corresponds to upper triangle of c_cc (or all off-diag)
        # Let's reconstruct the matrix of weights for c_cc
        w_cc = np.zeros((N, N))
        idx = 0
        for i in range(N):
            for j in range(N):
                if i != j:
                    w_cc[i, j] = weights[idx]
                    idx += 1
        
        # c_cc[i, j] = dist[i, j] - r_i - r_j
        # d(c_cc[i, j])/d(x_i) = (x_i - x_j) / dist[i, j]
        # d(c_cc[i, j])/d(r_i) = -1
        
        # Gradient of softmin w.r.t x_i:
        # sum_j w_cc[i, j] * (x_i - x_j)/dist[i, j]
        
        gx = np.zeros(N)
        gy = np.zeros(N)
        gr = np.zeros(N)
        
        # Accumulate contributions from circle-circle
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dist[i, j]
                if d < 1e-10:
                    continue
                wx = w_cc[i, j] * (x[i] - x[j]) / d
                wy = w_cc[i, j] * (y[i] - y[j]) / d
                wr = -w_cc[i, j]
                
                gx[i] += wx
                gy[i] += wy
                gr[i] += wr
                
                # Note: c_cc[j, i] is also in the set.
                # w_cc[j, i] * (x_j - x_i)/d = - w_cc[j, i] * (x_i - x_j)/d
                # So we must also add the contribution from the pair (j, i) to x_i.
                # Wait, the loop covers all i, j.
                # For a fixed i, we sum over all j != i.
                # The term w_cc[i, j] contributes to grad_x[i].
                # The term w_cc[j, i] contributes to grad_x[j].
                # When we iterate j for a fixed i, we are adding to gx[i].
                # The term from c_cc[j, i] to gx[i] would be when the loop is at index j? No.
                # The loop structure is:
                # for i:
                #   for j:
                #     update gx[i] based on c_cc[i, j]
                # This only accounts for the gradient of c_cc[i, j] w.r.t x_i.
                # But x_i also appears in c_cc[k, i] for all k != i.
                # So we need to add contributions from c_cc[k, i] to gx[i].
                # In the loop above, when we are at index k, and j=i, we update gx[k].
                # We do NOT update gx[i] from c_cc[k, i] in that step.
                # We need to accumulate correctly.
                
        # Let's redo the gradient accumulation more carefully using vectorization or double loops that are correct.
        
        gx = np.zeros(N)
        gy = np.zeros(N)
        gr = np.zeros(N)
        
        # Contributions from c_cc[i, j] to x_i, y_i, r_i
        # and c_cc[j, i] to x_j, y_j, r_j
        
        # We can iterate over pairs (i, j) with i < j
        for i in range(N):
            for j in range(i+1, N):
                d = dist[i, j]
                if d < 1e-10:
                    continue
                
                w_ij = w_cc[i, j]
                w_ji = w_cc[j, i]
                
                # c_cc[i, j] = d - r_i - r_j
                # grad_x_i from w_ij: w_ij * (x_i - x_j)/d
                # grad_x_j from w_ij: 0 (x_j not in c_cc[i, j] explicitly in numerator? No, d depends on x_j)
                # d(c_cc[i, j])/d(x_j) = (x_j - x_i)/d = -(x_i - x_j)/d
                # So w_ij contributes -w_ij * (x_i - x_j)/d to gx[j].
                
                # c_cc[j, i] = d - r_j - r_i
                # grad_x_i from w_ji: w_ji * (x_j - x_i)/d = -w_ji * (x_i - x_j)/d
                # grad_x_j from w_ji: w_ji * (x_j - x_i)/d = -w_ji * (x_i - x_j)/d ? No.
                # d(c_cc[j, i])/d(x_j) = (x_j - x_i)/d.
                
                # Let's sum them up.
                # Total grad_x_i = w_ij * (x_i - x_j)/d + w_ji * (x_j - x_i)/d
                #             = (w_ij - w_ji) * (x_i - x_j)/d
                
                # Total grad_y_i = (w_ij - w_ji) * (y_i - y_j)/d
                
                # Total grad_r_i = -w_ij - w_ji
                # Total grad_r_j = -w_ij - w_ji
                
                diff_x = x[i] - x[j]
                diff_y = y[i] - y[j]
                
                coef = (w_ij - w_ji) / d
                
                gx[i] += coef * diff_x
                gy[i] += coef * diff_y
                gr[i] -= (w_ij + w_ji)
                
                gx[j] -= coef * diff_x  # (w_ji - w_ij) * (x_j - x_i)/d = (w_ji - w_ij) * (-diff_x)/d = (w_ij - w_ji) * diff_x / d ?
                # Let's check:
                # grad_x_j from w_ij: w_ij * (x_j - x_i)/d = -w_ij * diff_x / d
                # grad_x_j from w_ji: w_ji * (x_j - x_i)/d = -w_ji * diff_x / d
                # Sum = -(w_ij + w_ji) * diff_x / d ? No.
                # d(c_cc[i, j])/d(x_j) = (x_j - x_i)/d
                # d(c_cc[j, i])/d(x_j) = (x_j - x_i)/d
                # So grad_x_j = w_ij * (x_j - x_i)/d + w_ji * (x_j - x_i)/d
                #             = (w_ij + w_ji) * (x_j - x_i)/d
                #             = -(w_ij + w_ji) * diff_x / d
                
                # My previous derivation for gx[i] was:
                # grad_x_i from c_cc[i, j]: w_ij * (x_i - x_j)/d
                # grad_x_i from c_cc[j, i]: w_ji * (x_j - x_i)/d
                # Sum = (w_ij - w_ji) * diff_x / d. Correct.
                
                # So:
                # gx[i] += (w_ij - w_ji) * diff_x / d
                # gx[j] -= (w_ij + w_ji) * diff_x / d  <-- Wait.
                # (x_j - x_i)/d = -diff_x/d.
                # So gx[j] += (w_ij + w_ji) * (-diff_x/d) = -(w_ij + w_ji) * diff_x / d.
                
                # Let's re-verify gx[j].
                # Term 1: w_ij * d(c_cc[i, j])/d(x_j) = w_ij * (x_j - x_i)/d
                # Term 2: w_ji * d(c_cc[j, i])/d(x_j) = w_ji * (x_j - x_i)/d
                # Sum = (w_ij + w_ji) * (x_j - x_i)/d = -(w_ij + w_ji) * diff_x / d.
                
                # So:
                gx[j] += -(w_ij + w_ji) * diff_x / d
                gy[j] += -(w_ij + w_ji) * diff_y / d
                gr[j] -= (w_ij + w_ji)
                
        # 2. Backprop to boundary clearances
        # c_b1 = x - r. d/dx = 1, d/dr = -1
        # c_b2 = 1 - x - r. d/dx = -1, d/dr = -1
        # c_b3 = y - r. d/dy = 1, d/dr = -1
        # c_b4 = 1 - y - r. d/dy = -1, d/dr = -1
        
        # Indices in c_all:
        # c_cc_flat: N*(N-1) elements
        # c_b1: N
        # c_b2: N
        # c_b3: N
        # c_b4: N
        
        offset_cc = N * (N - 1)
        offset_b1 = offset_cc
        offset_b2 = offset_b1 + N
        offset_b3 = offset_b2 + N
        offset_b4 = offset_b3 + N
        
        w_b1 = weights[offset_b1 : offset_b1 + N]
        w_b2 = weights[offset_b2 : offset_b2 + N]
        w_b3 = weights[offset_b3 : offset_b3 + N]
        w_b4 = weights[offset_b4 : offset_b4 + N]
        
        # gx += w_b1 - w_b2
        # gy += w_b3 - w_b4
        # gr -= w_b1 + w_b2 + w_b3 + w_b4
        
        gx += w_b1 - w_b2
        gy += w_b3 - w_b4
        gr -= (w_b1 + w_b2 + w_b3 + w_b4)
        
        # Total gradient of objective J = -sum(r) - lam * softmin
        # dJ/dr = -1 - lam * gr
        # dJ/dx = -lam * gx
        # dJ/dy = -lam * gy
        
        grad[:N] = -lam * gx
        grad[N:2*N] = -lam * gy
        grad[2*N:3*N] = -1.0 - lam * gr
        
        return obj, grad

    # Optimization loop
    num_restarts = 8
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            x0 = grid[:, 0].copy()
            y0 = grid[:, 1].copy()
            r0 = grid[:, 2].copy()
        elif i == 1:
            # Phyllotaxis
            x0 = phyl[:, 0].copy()
            y0 = phyl[:, 1].copy()
            r0 = phyl[:, 2].copy()
        elif i == 2:
            # Hexagonal
            x0 = hex_init[:, 0].copy()
            y0 = hex_init[:, 1].copy()
            r0 = hex_init[:, 2].copy()
        elif i == 3:
            # Perturbed phyllotaxis
            x0 = phyl[:, 0] + np.random.normal(0, 0.02, N)
            y0 = phyl[:, 1] + np.random.normal(0, 0.02, N)
            r0 = phyl[:, 2].copy()
        elif i == 4:
            # Perturbed hex
            x0 = hex_init[:, 0] + np.random.normal(0, 0.02, N)
            y0 = hex_init[:, 1] + np.random.normal(0, 0.02, N)
            r0 = hex_init[:, 2].copy()
        else:
            # Random
            x0 = np.random.uniform(0.1, 0.9, N)
            y0 = np.random.uniform(0.1, 0.9, N)
            r0 = np.full(N, 0.08)
            
        # Clip initial values to be valid
        r0 = np.maximum(r0, 1e-4)
        
        x0 = np.clip(x0, r0, 1-r0)
        y0 = np.clip(y0, r0, 1-r0)
        
        params0 = np.concatenate([x0, y0, r0])
        
        # Bounds
        # x, y in [0, 1]
        # r in [1e-4, 0.5]
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(1e-4, 0.5)] * N
        
        try:
            res = minimize(objective, params0, method='L-BFGS-B', jac=True, bounds=bounds,
                           options={'maxiter': 800, 'maxfun': 8000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.success or res.fun < best_score:
                # Check validity
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:3*N]
                
                # Verify disjointness
                valid = True
                for i in range(N):
                    if x_opt[i] < r_opt[i] - 1e-5 or x_opt[i] > 1 - r_opt[i] + 1e-5:
                        valid = False
                        break
                    if y_opt[i] < r_opt[i] - 1e-5 or y_opt[i] > 1 - r_opt[i] + 1e-5:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                            if dist < r_opt[i] + r_opt[j] - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    # Apply coordinate descent polish
                    x_pol, y_pol, r_pol = coordinate_descent_polish(x_opt, y_opt, r_opt, max_iter=100)
                    
                    score = np.sum(r_pol)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([x_pol, y_pol, r_pol])
                        
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to seed
        return np.array([[ (i+0.5)/6, (j+0.5)/6, 0.5/6 - 1e-3] for i in range(6) for j in range(6)][:N])
        
    return best_result
