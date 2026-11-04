"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radii(centers):
    """
    Computes the maximum valid radius for each circle given the centers.
    r_i = min( dist_to_boundary_i, min_{j != i} dist(i, j) / 2 )
    """
    n = len(centers)
    radii = np.zeros(n)
    
    # Distance to boundaries
    dist_to_box = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                             np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Distance to other circles
    # Using cdist is efficient for N=26
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(centers, centers)
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Minimum distance to any other circle
    min_neighbor_dist = np.min(dist_matrix, axis=1)
    
    # Radius is limited by both boundary and neighbors
    radii = np.minimum(dist_to_box, min_neighbor_dist / 2.0)
    
    return radii

def _objective_and_grad(centers_flat):
    """
    Objective: -sum(radii)
    Gradient: computed analytically for the active constraints.
    """
    centers = centers_flat.reshape(-1, 2)
    n = len(centers)
    radii = _compute_radii(centers)
    objective = -np.sum(radii)
    
    grad = np.zeros_like(centers_flat)
    
    # Distance to boundaries
    dist_to_box = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                             np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
                             
    # Distance to other circles
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(centers, centers)
    np.fill_diagonal(dist_matrix, np.inf)
    
    min_neighbor_dist = np.min(dist_matrix, axis=1)
    nearest_neighbor_idx = np.argmin(dist_matrix, axis=1)
    
    for i in range(n):
        r_bound = dist_to_box[i]
        r_neigh = min_neighbor_dist[i] / 2.0
        
        if r_bound <= r_neigh:
            # Limited by boundary
            # Check which boundary is active
            x, y = centers[i]
            if x <= 1.0 - x:
                grad[2*i] += 1.0
            else:
                grad[2*i] += -1.0
                
            if y <= 1.0 - y:
                grad[2*i+1] += 1.0
            else:
                grad[2*i+1] += -1.0
        else:
            # Limited by neighbor j
            j = nearest_neighbor_idx[i]
            d = min_neighbor_dist[i]
            if d > 1e-9:
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                inv_d = 1.0 / d
                
                # d(r_i)/d(c_i) = 0.5 * (c_i - c_j) / d
                grad[2*i] += 0.5 * dx * inv_d
                grad[2*i+1] += 0.5 * dy * inv_d
                
                # d(r_i)/d(c_j) = -0.5 * (c_i - c_j) / d
                # Note: r_j might also be affected by i, but we only compute grad of sum(r_k) wrt c_i.
                # The term -grad[2*j] ... is not added here because we are computing grad of sum(r_k) wrt c_i.
                # Wait, objective is -sum(r_k).
                # d(-sum(r_k))/d(c_i) = - sum_k d(r_k)/d(c_i).
                # Only r_i and potentially r_j depend on c_i?
                # r_i depends on c_i.
                # r_j depends on c_j and c_i (if i is nearest to j).
                # This analytic gradient is incomplete if we don't account for r_j's dependence on c_i.
                # However, for a "polish" step, L-BFGS-B with finite difference check or just robustness
                # might be enough, or we can use a simpler gradient that only considers r_i.
                # Actually, let's use a more robust approach for the polish:
                # Just use the gradient of r_i. The optimizer will handle the rest via line search.
                # But if the gradient is wrong, it might diverge.
                # Let's stick to the gradient of the individual radius for the circle being moved,
                # assuming other radii don't change significantly in the small step.
                # Or better: use numerical gradient if analytic is tricky?
                # No, let's refine the analytic gradient.
                # d(r_i)/d(c_i) = 0.5 * u_ij
                # d(r_j)/d(c_i) = ?
                # If j's nearest neighbor is i, then r_j = d_ij/2.
                # d(r_j)/d(c_i) = -0.5 * u_ij.
                # So total grad wrt c_i for (r_i + r_j) is 0.5 u_ij - 0.5 u_ij = 0?
                # That would imply moving i doesn't change sum if both are limited by each other?
                # That's true for the pair, but moving i might affect other neighbors.
                # This analytic gradient is complex.
                # Let's use a simpler strategy:
                # The SA will do the heavy lifting. The polish will use a "relaxation" approach
                # or a simpler gradient.
                # Actually, for the polish, we can just use the gradient of the objective function
                # computed via finite differences if the analytic one is unreliable?
                # Or just skip the analytic gradient and use a simple local search in the polish?
                # Let's use a "best fit" local search for the polish instead of L-BFGS.
                # It's more robust and doesn't require gradients.
                pass
        
    # Since the analytic gradient is tricky for the sum of radii due to mutual dependencies,
    # we will use a local search for the final polish instead of L-BFGS.
    # The SA will provide a good starting point.
    return objective, grad

def _eval_packing(centers):
    """
    Evaluate the packing quality.
    Returns the sum of radii.
    """
    radii = _compute_radii(centers)
    return np.sum(radii)

def _grid_init():
    """
    Initialize centers in a grid pattern.
    """
    centers = np.zeros((N, 2))
    side = int(np.ceil(np.sqrt(N)))
    for i in range(N):
        gx = i % side
        gy = i // side
        offset_x = 0.5 / side if gy % 2 == 1 else 0.0
        x = (gx + 0.5) / side + offset_x
        y = (gy + 0.5) / side
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        centers[i] = [x, y]
    return centers

def _random_init():
    """
    Initialize centers randomly.
    """
    return np.random.uniform(0.1, 0.9, (N, 2))

def _best_fit_local_search(centers, max_iter=1000, step_size=0.01):
    """
    A simple local search that moves one circle at a time to the best position
    that increases the sum of radii.
    """
    best_score = _eval_packing(centers)
    current_centers = centers.copy()
    
    for _ in range(max_iter):
        improved = False
        # Try moving each circle
        for i in range(N):
            # Try moving in 8 directions
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_centers = current_centers.copy()
                    new_x = np.clip(new_centers[i, 0] + dx, 0.0, 1.0)
                    new_y = np.clip(new_centers[i, 1] + dy, 0.0, 1.0)
                    new_centers[i, 0] = new_x
                    new_centers[i, 1] = new_y
                    
                    new_score = _eval_packing(new_centers)
                    if new_score > best_score:
                        best_score = new_score
                        current_centers = new_centers
                        improved = True
        if not improved:
            break
            
    return current_centers, best_score

def _sa_packing(init_centers, start_time, deadline):
    """
    Simulated Annealing for circle packing.
    """
    current_centers = init_centers.copy()
    current_score = _eval_packing(current_centers)
    
    best_centers = current_centers.copy()
    best_score = current_score
    
    T = 0.05  # Initial temperature
    cooling_rate = 0.995
    min_T = 1e-5
    
    while T > min_T:
        if time.time() > deadline - 5:
            break
            
        # Generate a new solution by moving a random circle
        i = np.random.randint(N)
        angle = np.random.uniform(0, 2 * np.pi)
        # Step size proportional to temperature? Or fixed?
        # Let's use a fixed step size that decreases slightly
        step = 0.01 + 0.01 * (T / start_T)
        
        new_centers = current_centers.copy()
        new_x = np.clip(new_centers[i, 0] + step * np.cos(angle), 0.0, 1.0)
        new_y = np.clip(new_centers[i, 1] + step * np.sin(angle), 0.0, 1.0)
        new_centers[i, 0] = new_x
        new_centers[i, 1] = new_y
        
        new_score = _eval_packing(new_centers)
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(delta/T) if worse
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_centers = new_centers
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_centers = current_centers.copy()
        
        T *= cooling_rate
        
    return best_centers, best_score

def _gradient_polish(centers):
    """
    Final polish using L-BFGS-B with a smoothed objective or just a simple local search.
    Given the complexity of the gradient, we will use a refined local search.
    """
    # Use a smaller step size local search
    polished_centers, polished_score = _best_fit_local_search(centers, max_iter=500, step_size=0.001)
    return polished_centers, polished_score

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    np.random.seed(42)
    
    best_centers = None
    best_score = -np.inf
    
    # Multi-start with SA
    inits = [_grid_init()]
    for _ in range(15):
        inits.append(_random_init())
        
    for trial, init_centers in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        sa_centers, sa_score = _sa_packing(init_centers, start_time, deadline)
        
        if sa_score > best_score:
            best_score = sa_score
            best_centers = sa_centers.copy()
            
    if best_centers is None:
        best_centers = _grid_init()
        best_score = _eval_packing(best_centers)
        
    # Final polish
    if time.time() < deadline - 5:
        polished_centers, polished_score = _gradient_polish(best_centers)
        if polished_score > best_score:
            best_centers = polished_centers
            best_score = polished_score
            
    radii = _compute_radii(best_centers)
    
    # Final validation and clipping
    result = np.column_stack([best_centers, radii])
    for i in range(N):
        x, y, ri = result[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
        
    return result
# EVOLVE-BLOCK-END
