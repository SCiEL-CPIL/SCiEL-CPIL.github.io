import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _min_area_bary(points_bary):
    """
    Compute the minimum area of any triangle formed by 3 points.
    points_bary: (N, 2) array of (u, v) coordinates. w = 1 - u - v.
    Returns the minimum area.
    """
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    # We need to calculate the determinant for all triples (i, j, k)
    # det = u_i(v_j w_k - w_j v_k) - v_i(u_j w_k - w_j u_k) + w_i(u_j v_k - v_j u_k)
    
    # Vectorized approach:
    # Create arrays of indices
    i, j, k = np.indices((N, N, N))
    # Filter to only i < j < k
    mask = (i < j) & (j < k)
    i_f = i[mask]
    j_f = j[mask]
    k_f = k[mask]
    
    # Extract coordinates for these triples
    u_i = u[i_f]
    v_i = v[i_f]
    w_i = w[i_f]
    
    u_j = u[j_f]
    v_j = v[j_f]
    w_j = w[j_f]
    
    u_k = u[k_f]
    v_k = v[k_f]
    w_k = w[k_f]
    
    # Calculate determinants
    det = (u_i * (v_j * w_k - w_j * v_k) 
         - v_i * (u_j * w_k - w_j * u_k) 
         + w_i * (u_j * v_k - v_j * u_k))
         
    areas = np.abs(det)
    return np.min(areas)

def _project_to_simplex(points_bary):
    """
    Project points to the valid barycentric simplex.
    If u < 0, v < 0, or u+v > 1, adjust them.
    """
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    
    # Clip small negatives
    u = np.maximum(u, 0)
    v = np.maximum(v, 0)
    
    # If u + v > 1, scale down
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
        
    return np.column_stack([u, v])

def _objective(params):
    """
    Objective function for minimization (negative of min area).
    """
    points_bary = params.reshape(N, 2)
    points_bary = _project_to_simplex(points_bary)
    min_a = _min_area_bary(points_bary)
    return -min_a

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s for final polish and safety
    
    # Initial random points
    rng = np.random.RandomState(42)
    # Generate random barycentric coordinates
    # Dirichlet distribution with alpha=1 gives uniform distribution in simplex
    points_bary = rng.dirichlet([1, 1, 1], size=N)[:, :2]
    
    # Calculate initial area
    current_area = _min_area_bary(points_bary)
    best_area = current_area
    best_points_bary = points_bary.copy()
    
    # Simulated Annealing Parameters
    T = 1e-4  # Initial temperature (area scale is roughly 1/100 to 1/10)
    T_min = 1e-10
    alpha = 0.9999 # Cooling rate
    step_size = 0.1 # Initial step size
    
    # SA Loop
    while time.time() < deadline:
        # Generate a neighbor
        # Perturb a random subset of points
        n_perturb = rng.randint(1, 4)
        indices = rng.choice(N, size=n_perturb, replace=False)
        
        new_points_bary = points_bary.copy()
        noise = rng.normal(0, step_size, size=(n_perturb, 2))
        new_points_bary[indices] += noise
        new_points_bary = _project_to_simplex(new_points_bary)
        
        new_area = _min_area_bary(new_points_bary)
        
        # Metropolis criterion
        if new_area > current_area:
            # Accept if better
            current_area = new_area
            points_bary = new_points_bary
            if current_area > best_area:
                best_area = current_area
                best_points_bary = points_bary.copy()
        else:
            # Accept if worse with probability exp((new - current) / T)
            delta = new_area - current_area
            prob = np.exp(delta / T)
            if rng.random() < prob:
                current_area = new_area
                points_bary = new_points_bary
        
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            # Reheat if stuck? Or just keep going with small steps.
            # Let's try a small reheat if we haven't improved in a while
            # But for simplicity, just keep cooling to 0.
        
        # Adjust step size slightly to keep exploration active
        # step_size *= 0.9995 # Optional: shrink step size over time

    # Final Polish using Nelder-Mead on the best point found
    # We convert to Cartesian for the optimizer, or stick to Barycentric?
    # Barycentric is better because the domain is a simplex, not a triangle in xy.
    # However, Nelder-Mead in 22 dimensions (2 per point) is fine.
    
    def polish_objective(params):
        pb = params.reshape(N, 2)
        pb = _project_to_simplex(pb)
        return -_min_area_bary(pb)
    
    x0 = best_points_bary.flatten()
    
    try:
        res = minimize(
            polish_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        if res.fun < -best_area: # res.fun is -area, so if -area < -best_area, area > best_area
            pb_opt = res.x.reshape(N, 2)
            pb_opt = _project_to_simplex(pb_opt)
            best_area = -res.fun
            best_points_bary = pb_opt
    except Exception:
        pass

    # Convert best barycentric points to Cartesian
    u = best_points_bary[:, 0]
    v = best_points_bary[:, 1]
    w = 1.0 - u - v
    
    h = np.sqrt(3) / 2
    x = v * 1.0 + w * 0.5
    y = w * h
    
    points_cart = np.column_stack([x, y])
    
    # Final safety clamp to triangle
    eps = 1e-9
    for i in range(N):
        x, y = points_cart[i]
        y = max(eps, y)
        if y > np.sqrt(3) * x:
            y = np.sqrt(3) * x - eps
        if y > np.sqrt(3) * (1 - x):
            y = np.sqrt(3) * (1 - x) - eps
        x = max(eps, min(1 - eps, x))
        points_cart[i] = [x, y]
        
    return points_cart

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print("Points:")
    print(pts)
    # Calculate score for debugging
    # Convert back to barycentric to check area
    x = pts[:, 0]
    y = pts[:, 1]
    h = np.sqrt(3) / 2
    w = y / h
    v = x - w * 0.5
    u = 1 - w - v
    pb = np.column_stack([u, v])
    print("Min Area:", _min_area_bary(pb))
