import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s safety margin
    
    best_score = -1
    best_result = None
    
    # Initialize with a grid
    side = int(np.ceil(np.sqrt(N)))
    # Use a slightly perturbed grid to break symmetry
    rng = np.random.default_rng(42)
    
    for restart in range(50): # Try multiple restarts
        if time.time() > deadline:
            break
            
        # Initialization
        # Grid positions
        xs = np.linspace(0.1, 0.9, side)
        ys = np.linspace(0.1, 0.9, side)
        px, py = np.meshgrid(xs, ys)
        p = np.column_stack((px.ravel(), py.ravel()))[:N]
        
        # Add small random perturbation
        p += rng.uniform(-0.01, 0.01, p.shape)
        
        # Initial radii: estimate based on grid spacing
        # For a grid of side x side in 0.8x0.8 area
        # Spacing is approx 0.8/side
        # Radius approx 0.4/side
        init_r = 0.4 / side
        r = np.full(N, init_r)
        
        # Optimization
        # We use a simple gradient ascent with adaptive step size
        # Objective: Maximize sum(r) - penalty(overlaps) - penalty(boundary)
        
        alpha = 10.0 # Overlap penalty weight
        beta = 10.0  # Boundary penalty weight
        lr = 0.01
        
        for step in range(2000):
            if time.time() > deadline:
                break
                
            # Compute distances
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dist, np.inf)
            
            # Compute overlaps
            # overlap_ij = max(0, r_i + r_j - dist_ij)
            overlap_sum = np.zeros((N, N))
            for i in range(N):
                for j in range(i+1, N):
                    val = r[i] + r[j] - dist[i, j]
                    if val > 0:
                        overlap_sum[i, j] = val
                        overlap_sum[j, i] = val
            
            # Boundary distances
            # dist to boundary = min(x, 1-x, y, 1-y)
            dist_bound = np.minimum(np.minimum(p[:, 0], 1 - p[:, 0]), np.minimum(p[:, 1], 1 - p[:, 1]))
            
            # Penalties
            overlap_pen = np.sum(overlap_sum ** 2)
            bound_pen = np.sum(np.maximum(0, r - dist_bound) ** 2)
            
            score = np.sum(r) - alpha * overlap_pen - beta * bound_pen
            
            if score > best_score:
                best_score = score
                # Check validity for best_result update
                # Valid if no overlaps and inside square
                valid = True
                for i in range(N):
                    if r[i] > dist_bound[i]:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            if r[i] + r[j] > dist[i, j] + 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if valid:
                        best_result = np.column_stack((p, r)).copy()
            
            # Compute gradients
            grad_p = np.zeros_like(p)
            grad_r = np.zeros_like(r)
            
            # Overlap gradients
            for i in range(N):
                for j in range(i+1, N):
                    val = overlap_sum[i, j]
                    if val > 0:
                        # d/dp_i (r_i + r_j - dist)^2 = 2 val * (p_i - p_j) / dist
                        # d/dr_i (r_i + r_j - dist)^2 = 2 val
                        if dist[i, j] > 1e-9:
                            grad_p[i] -= 2 * alpha * val * (p[i] - p[j]) / dist[i, j]
                            grad_p[j] -= 2 * alpha * val * (p[j] - p[i]) / dist[i, j]
                        grad_r[i] -= 2 * alpha * val
                        grad_r[j] -= 2 * alpha * val
            
            # Boundary gradients
            for i in range(N):
                val = r[i] - dist_bound[i]
                if val > 0:
                    grad_r[i] -= 2 * beta * val
                    # Gradient of dist_bound
                    # If x < 0.5, dist=x, grad=1. If x>0.5, dist=1-x, grad=-1.
                    # Similar for y.
                    # We need the gradient of the min function.
                    # Let's just push away from boundary.
                    if p[i, 0] < 0.5:
                        grad_p[i, 0] -= 2 * beta * val
                    else:
                        grad_p[i, 0] += 2 * beta * val
                    if p[i, 1] < 0.5:
                        grad_p[i, 1] -= 2 * beta * val
                    else:
                        grad_p[i, 1] += 2 * beta * val
            
            # Objective gradient: grad(sum r) = 1
            grad_r += 1.0
            
            # Update
            p += lr * grad_p
            r += lr * grad_r
            
            # Clip r to be positive
            r = np.maximum(r, 1e-6)
            
            # Clip p to be inside [0,1]
            p = np.clip(p, 0, 1)
            
            # Adaptive learning rate? Simple decay
            lr *= 0.999
        
        # After optimization, do a final cleanup to ensure strict validity
        if best_result is not None:
             # Try to scale up radii slightly if there is slack?
             # Or just return best valid found.
             pass

    if best_result is not None:
        return best_result
    
    # Fallback: return the grid seed
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
