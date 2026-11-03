import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_score = -np.inf
    best_result = None
    
    # Objective function: Maximize sum of radii
    # We minimize -sum(radii) + penalty for constraints
    # Constraints:
    # 1. Circles inside [0,1]^2: x_i >= r_i, y_i >= r_i, x_i <= 1-r_i, y_i <= 1-r_i
    # 2. Non-overlap: dist(i,j) >= r_i + r_j
    
    def penalty_and_grad(z):
        # z shape: (3*N,) -> [x0, y0, r0, x1, y1, r1, ...]
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Softplus penalty for boundary violations
        # Violation if x < r -> x - r < 0. We want to penalize x - r < 0.
        # Let v = r - x. If v > 0, penalty.
        # Use softplus(v) = log(1 + exp(v)) for smoothness, but scale it.
        # Actually, standard softplus is for x > 0. 
        # Let's define violation distance d. If d > 0, it's a violation.
        # Boundary: d_left = r - x. Penalty = softplus(d_left)
        # Overlap: d_overlap = r_i + r_j - dist_ij. Penalty = softplus(d_overlap)
        
        # To avoid overflow in exp, use np.logaddexp(0, v)
        
        # Boundary Penalties
        # x < r => r - x > 0
        v_x_left = r - x
        v_x_right = r - (1 - x) # x > 1-r => 1-x < r => r - (1-x) > 0
        v_y_bottom = r - y
        v_y_top = r - (1 - y)
        
        # Sum of softplus for boundaries
        pen_b = np.sum(np.logaddexp(0, v_x_left) + np.logaddexp(0, v_x_right) + 
                       np.logaddexp(0, v_y_bottom) + np.logaddexp(0, v_y_top))
        
        # Overlap Penalties
        # dist_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
        # violation = ri + rj - dist_ij
        # We compute this for all pairs i < j
        
        # Efficient calculation of pairwise distances
        # x, y, r are vectors of length N
        # Create matrices
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        # dist is NxN matrix. Diagonal is 0.
        # We only care about i < j.
        # Create a mask for upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        dist_upper = dist[mask]
        
        # ri + rj for pairs
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        r_sum_upper = r_sum[mask]
        
        v_overlap = r_sum_upper - dist_upper
        
        pen_o = np.sum(np.logaddexp(0, v_overlap))
        
        total_penalty = pen_b + pen_o
        objective = -np.sum(r) + total_penalty
        
        # Gradients
        # Gradient of -sum(r) is -1 for all r components.
        grad = np.zeros_like(z)
        grad[2::3] -= 1.0
        
        # Gradient of softplus(v) is sigmoid(v) = 1 / (1 + exp(-v))
        # For v > 0, sigmoid(v) ~ 1. For v < 0, sigmoid(v) ~ 0.
        
        # Boundary Grads
        # v_x_left = r - x. d/dx = -sigmoid(v), d/dr = sigmoid(v)
        s_xl = 1 / (1 + np.exp(-v_x_left))
        grad[0::3] -= s_xl
        grad[2::3] += s_xl
        
        # v_x_right = r - 1 + x. d/dx = sigmoid(v), d/dr = sigmoid(v)
        s_xr = 1 / (1 + np.exp(-v_x_right))
        grad[0::3] += s_xr
        grad[2::3] += s_xr
        
        # v_y_bottom = r - y. d/dy = -sigmoid(v), d/dr = sigmoid(v)
        s_yb = 1 / (1 + np.exp(-v_y_bottom))
        grad[1::3] -= s_yb
        grad[2::3] += s_yb
        
        # v_y_top = r - 1 + y. d/dy = sigmoid(v), d/dr = sigmoid(v)
        s_yt = 1 / (1 + np.exp(-v_y_top))
        grad[1::3] += s_yt
        grad[2::3] += s_yt
        
        # Overlap Grads
        # v_ij = r_i + r_j - dist_ij
        # d/dx_i = (x_i - x_j) / dist_ij * sigmoid(v_ij)
        # d/r_i = sigmoid(v_ij)
        
        # We need to scatter the gradients back from the upper triangle vector to the full matrix structure
        # or just loop. N=26, so N^2/2 = 325 pairs. Looping is fine.
        
        # To be faster, we can compute the gradient matrix G_dist where G_dist[i,j] is contribution to x_i
        # But looping is simpler and less error-prone for N=26.
        
        s_ov = 1 / (1 + np.exp(-v_overlap))
        
        # Re-iterate pairs to add gradients
        # This is O(N^2) which is 676 ops, very fast.
        for i in range(N):
            for j in range(i+1, N):
                # Get index in upper triangle array
                # This is tricky to map back without storing indices.
                # Let's just recompute dist and v for this pair? No, we have the values.
                # Better: construct the gradient contributions in a vectorized way if possible, 
                # or just accept the loop cost if we pre-calculate indices.
                
                # Let's use a helper to get the index in the flattened upper triangle
                # Actually, it's easier to just compute the gradient terms directly in the loop
                # using the original x,y,r.
                dx_ij = x[i] - x[j]
                dy_ij = y[i] - y[j]
                d_ij = np.sqrt(dx_ij**2 + dy_ij**2)
                if d_ij < 1e-10:
                    d_ij = 1e-10
                    dx_ij = 1e-10
                    dy_ij = 0
                
                v_ij = r[i] + r[j] - d_ij
                s_ij = 1 / (1 + np.exp(-v_ij))
                
                # Gradient for x_i: (x_i - x_j) / d_ij * s_ij
                # Gradient for x_j: (x_j - x_i) / d_ij * s_ij = - (x_i - x_j) / d_ij * s_ij
                # Gradient for r_i: s_ij
                # Gradient for r_j: s_ij
                
                gx_i = (dx_ij / d_ij) * s_ij
                gy_i = (dy_ij / d_ij) * s_ij
                gx_j = -gx_i
                gy_j = -gy_i
                
                # Add to grad array
                # x_i is at index 3*i
                grad[3*i] += gx_i
                grad[3*i + 1] += gy_i
                grad[3*i + 2] += s_ij
                
                # x_j is at index 3*j
                grad[3*j] += gx_j
                grad[3*j + 1] += gy_j
                grad[3*j + 2] += s_ij
        
        return objective, grad

    def check_feasibility(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Check boundaries
        if np.any(x < r - 1e-9) or np.any(x > 1 - r + 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(y > 1 - r + 1e-9):
            return False
            
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True

    def run_optimization(initial_z):
        # Use L-BFGS-B
        # Bounds: x,y in [0,1], r in [0, 0.5] (max radius for one circle in unit square is 0.5)
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
        bounds = list(zip([0]*N + [0]*N + [0]*N, [1]*N + [1]*N + [0.5]*N))
        
        res = minimize(penalty_and_grad, initial_z, method='L-BFGS-B', jac=True, 
                       bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
        return res.x, res.fun

    # Multi-start strategy
    starts = []
    
    # 1. Grid start
    side = int(np.ceil(np.sqrt(N)))
    for i in range(side):
        for j in range(side):
            if len(starts) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                starts.append([x, y, r])
    # Pad with small circles if needed
    while len(starts) < N:
        starts.append([0.5, 0.5, 0.01])
    starts = np.array(starts).flatten()
    
    # 2. Random starts
    np.random.seed(42)
    for _ in range(5):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.02, 0.08, N)
        z = np.empty(3*N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        starts_list = np.array([x, y, r]).T.flatten()
        # Add to starts list (we will run optimization on each)
        # We'll store them in a list of arrays
        pass # We'll just iterate and run

    # Let's structure the loop properly
    candidate_starts = []
    
    # Add grid start
    grid_z = np.array(starts)
    candidate_starts.append(grid_z)
    
    # Add random starts
    for _ in range(10):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.02, 0.08, N)
        z = np.empty(3*N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        candidate_starts.append(z)

    # Add physics-based start
    # Simple repulsion relaxation
    x = np.random.uniform(0.2, 0.8, N)
    y = np.random.uniform(0.2, 0.8, N)
    r = np.full(N, 0.05)
    for _ in range(500):
        # Calculate forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        for i in range(N):
            # Boundary force
            if x[i] < r[i]: fx[i] += (r[i] - x[i]) * 10
            if x[i] > 1 - r[i]: fx[i] -= (x[i] - (1 - r[i])) * 10
            if y[i] < r[i]: fy[i] += (r[i] - y[i]) * 10
            if y[i] > 1 - r[i]: fy[i] -= (y[i] - (1 - r[i])) * 10
            
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    if dist < 1e-6:
                        dx = np.random.uniform(-1, 1)
                        dy = np.random.uniform(-1, 1)
                        dist = np.sqrt(dx*dx + dy*dy)
                    force = (min_dist - dist) / dist
                    fx[i] += dx * force * 10
                    fy[i] += dy * force * 10
                    fx[j] -= dx * force * 10
                    fy[j] -= dy * force * 10
        
        # Update positions
        x += fx * 0.01
        y += fy * 0.01
        # Clamp to boundaries
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        
    phys_z = np.empty(3*N)
    phys_z[0::3] = x
    phys_z[1::3] = y
    phys_z[2::3] = r
    candidate_starts.append(phys_z)

    # Optimize each candidate
    for z_init in candidate_starts:
        if time.time() > deadline:
            break
            
        try:
            z_opt, _ = run_optimization(z_init)
            
            # Check feasibility
            if check_feasibility(z_opt):
                score = np.sum(z_opt[2::3])
                if score > best_score:
                    best_score = score
                    best_result = z_opt
        except Exception as e:
            continue

    if best_result is None:
        # Fallback to grid if optimization failed
        return np.array(starts).reshape(N, 3)
        
    return best_result.reshape(N, 3)

# EVOLVE-BLOCK-START
# The code above is the complete implementation.
# We need to wrap it in the required structure if the prompt implies the block is the only part to change,
# but the prompt says "Rewrite the program from scratch... Output the COMPLETE program".
# The seed program has EVOLVE-BLOCK markers. I will provide the full script.
# EVOLVE-BLOCK-END
