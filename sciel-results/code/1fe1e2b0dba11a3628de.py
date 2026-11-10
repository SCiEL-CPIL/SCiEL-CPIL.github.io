import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    dx = centers[:, 0:1] - centers[:, 0:1].T
    dy = centers[:, 1:2] - centers[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _soft_min_value_and_grad(dists, boundary_dists, beta=100.0):
    """
    Compute the soft-minimum of distances and its gradient.
    We want to maximize the minimum distance.
    min(d) ≈ -1/beta * log(sum(exp(-beta*d)))
    So we minimize -soft_min = 1/beta * log(sum(exp(-beta*d)))
    
    Here dists includes both pairwise distances (divided by 2? No, let's keep raw distances for consistency)
    and boundary distances.
    Actually, the radius r is min(min_pair_dist/2, min_boundary_dist).
    Let's define the objective as minimizing F(x) = -min( min_pair/2, min_boundary ).
    Using soft-min:
    S1 = soft_min(pairwise_dists)
    S2 = soft_min(boundary_dists)
    S = soft_min([S1/2, S2])? No, that's nested.
    
    Better:
    Let D be the vector of all critical distances:
    D = [d_ij / 2 for i<j, 1-|c_i| for i]
    We want to maximize min(D).
    Objective to minimize: J = -soft_min(D) = 1/beta * log(sum(exp(-beta * D)))
    
    Gradient of J w.r.t x:
    dJ/dx = 1/beta * (1/sum(exp(-beta*D))) * sum( (-beta * exp(-beta*D_k)) * dD_k/dx )
    dJ/dx = - sum( w_k * dD_k/dx )
    where w_k = exp(-beta*D_k) / sum(exp(-beta*D_j))
    
    dD_k/dx for pair (i,j):
    D_k = 0.5 * ||c_i - c_j||
    dD_k/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
    dD_k/dc_j = 0.5 * (c_j - c_i) / ||c_i - c_j||
    
    dD_k/dx for boundary i:
    D_k = 1 - ||c_i||
    dD_k/dc_i = - c_i / ||c_i||
    """
    if dists.shape[0] == 0:
        return 0.0, np.zeros_like(dists) # Placeholder
    
    # To avoid overflow in exp(-beta * D) if D is large, we can shift, 
    # but beta is usually not too large (10-100) and D is O(0.1-1.0), so -beta*D is O(-10 to -100).
    # exp(-100) is small but safe. exp(+100) would be bad, but we have negative sign.
    
    # Weights
    exp_terms = np.exp(-beta * dists)
    sum_exp = np.sum(exp_terms)
    if sum_exp == 0:
        weights = np.zeros_like(dists)
    else:
        weights = exp_terms / sum_exp
    
    value = (1.0 / beta) * np.log(sum_exp)
    
    return value, weights

def _sqp_objective_and_grad(x, beta=50.0):
    """
    Objective function for L-BFGS-B.
    x is flattened centers (N*2,).
    """
    centers = x.reshape(N, 2)
    
    # Pairwise distances
    dx = centers[:, 0:1] - centers[:, 0:1].T
    dy = centers[:, 1:2] - centers[:, 1:2].T
    dists_sq = dx**2 + dy**2
    np.fill_diagonal(dists_sq, 0)
    dists = np.sqrt(dists_sq)
    
    # We only need upper triangle
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    bound_dists = 1.0 - norms
    
    # Combine into single vector D
    D = np.concatenate([pair_dists, bound_dists])
    
    # Soft min value and weights
    val, weights = _soft_min_value_and_grad(D, None, beta)
    
    # Objective is -val (since we want to maximize min, we minimize -soft_min)
    # Wait, _soft_min_value_and_grad returns 1/beta * log(sum(exp(-beta*D))) which is approx -min(D).
    # So J = val is approx -min(D). We want to minimize J.
    # So objective = val.
    
    # Gradient
    grad = np.zeros_like(x)
    
    # Pairwise part
    w_pair = weights[:len(pair_dists)]
    # d(D_k)/dc_i = 0.5 * (c_i - c_j) / d_ij
    # d(D_k)/dc_j = 0.5 * (c_j - c_i) / d_ij
    # dJ/dc_i = - sum_k w_k * dD_k/dc_i
    
    # We need d_ij (full distance, not halved)
    # pair_dists = d_ij / 2. So d_ij = 2 * pair_dists.
    # But we have the vectors dx, dy.
    
    # Let's compute gradients directly from weights
    # For each pair k with weight w_k:
    # Force on i: -w_k * 0.5 * (c_i - c_j) / d_ij
    # Force on j: -w_k * 0.5 * (c_j - c_i) / d_ij
    
    # Vectorized update:
    # grad_pair_i = -0.5 * sum_k w_k * (c_i - c_j) / d_ij
    # We can build a matrix of forces.
    
    # d_ij matrix
    d_ij = np.sqrt(dists_sq)
    # Avoid division by zero
    d_ij[d_ij < 1e-10] = 1e-10
    
    # Weighted displacement vectors
    # W_k is weight for pair (i,j).
    # We want to sum w_k * (c_i - c_j) / d_ij for each i.
    
    # Create a matrix of unit vectors u_ij = (c_i - c_j) / d_ij
    # u_ij is N x N
    # u_ij[i,j] = (c_i - c_j) / d_ij[i,j]
    
    # We only have upper triangle weights.
    # Let's construct a full weight matrix W_full where W_full[i,j] = w_k if (i,j) is the k-th pair.
    # W_full is symmetric? No, weight is for the pair.
    # dD/dx involves c_i and c_j.
    
    # Let's use the fact that:
    # grad_c_i = - sum_{j!=i} w_{ij} * 0.5 * (c_i - c_j) / d_{ij}
    # Note: w_{ij} = w_{ji} in our symmetric formulation?
    # In our D vector, each pair appears once.
    # Let's map weights back to a matrix.
    
    W_mat = np.zeros((N, N))
    W_mat[i_idx, j_idx] = w_pair
    W_mat[j_idx, i_idx] = w_pair
    
    # Now, grad_c_i = -0.5 * sum_j W_mat[i,j] * (c_i - c_j) / d_ij[i,j]
    
    # (c_i - c_j) / d_ij is the unit vector.
    # Let U[i,j] = (c_i - c_j) / d_ij[i,j]
    # U is antisymmetric: U[j,i] = -U[i,j].
    
    # Compute U
    # U[i,j] = (c_i - c_j) / d_ij
    # We have dx, dy.
    # U_x = dx / d_ij
    # U_y = dy / d_ij
    
    U_x = dx / d_ij
    U_y = dy / d_ij
    
    # grad_x_i = -0.5 * sum_j W_mat[i,j] * U_x[i,j]
    # grad_y_i = -0.5 * sum_j W_mat[i,j] * U_y[i,j]
    
    grad_x = -0.5 * (W_mat * U_x).sum(axis=1)
    grad_y = -0.5 * (W_mat * U_y).sum(axis=1)
    
    # Boundary part
    w_bound = weights[len(pair_dists):]
    # D_k = 1 - ||c_i||
    # dD_k/dc_i = - c_i / ||c_i||
    # dJ/dc_i = - w_k * (- c_i / ||c_i||) = w_k * c_i / ||c_i||
    
    # norms is (N,)
    # c_i / ||c_i|| is unit vector.
    # grad_x_i += w_bound[i] * c_i[0] / norms[i]
    # grad_y_i += w_bound[i] * c_i[1] / norms[i]
    
    # Avoid div by zero
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    
    grad_x += w_bound * (centers[:, 0] / safe_norms)
    grad_y += w_bound * (centers[:, 1] / safe_norms)
    
    # Assemble gradient
    grad[0::2] = grad_x
    grad[1::2] = grad_y
    
    return val, grad

def _physics_relax(centers, iterations=500):
    """
    Simple force-directed relaxation to separate points and keep them inside.
    """
    centers = centers.copy()
    # Target distance based on current best estimate or just a fixed small distance
    # Let's use a repulsive force for close points and attractive to center?
    # No, just repulsive + boundary confinement.
    
    for _ in range(iterations):
        forces = np.zeros_like(centers)
        
        # Repulsive forces between close points
        # O(N^2) but N=50 is small.
        for i in range(N):
            for j in range(i+1, N):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 1e-6:
                    d = np.random.normal(size=2)
                    dist = np.linalg.norm(d)
                
                # Strong repulsion if dist < 0.2
                if dist < 0.2:
                    # Force magnitude
                    f = (0.2 - dist) / dist
                    forces[i] += d * f
                    forces[j] -= d * f
        
        # Boundary force
        norms = np.linalg.norm(centers, axis=1)
        for i in range(N):
            if norms[i] > 0.9:
                # Push inward
                f = (norms[i] - 0.9) / norms[i]
                forces[i] -= centers[i] * f * 2.0 # Stronger
        
        # Update
        step_size = 0.05
        centers += forces * step_size
        
        # Hard clip to unit circle
        norms = np.linalg.norm(centers, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            centers[out_mask] /= norms[out_mask, None]
            
    return centers

def _hexagonal_packing():
    """Generate initial hexagonal lattice packing."""
    spacing = 0.25
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    if len(points) < N:
        return _random_packing()
    points = points[np.argsort(radius, kind="stable")[:N]]
    return points

def _concentric_packing():
    """Generate initial concentric ring packing."""
    centers = []
    centers.append([0.0, 0.0])
    rings = [
        (6, 0.25),
        (12, 0.50),
        (18, 0.75),
        (13, 0.95)
    ]
    for count, r in rings:
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        for a in angles:
            centers.append([r * np.cos(a), r * np.sin(a)])
    centers = np.array(centers[:N])
    return centers

def _spiral_packing():
    """Generate initial Fibonacci spiral packing."""
    centers = []
    ga = math.pi * (3.0 - math.sqrt(5.0))
    c = 0.9 / math.sqrt(N)
    for i in range(N):
        r = c * math.sqrt(i + 0.5)
        theta = ga * i
        centers.append([r * math.cos(theta), r * math.sin(theta)])
    return np.array(centers)

def _random_packing():
    """Generate random initial packing."""
    return np.random.uniform(-0.8, 0.8, size=(N, 2))

def _optimize_lagrangian(centers_init, deadline):
    """
    Optimize using L-BFGS-B with soft-min objective.
    """
    x0 = centers_init.flatten()
    
    # Constraints: points must be inside unit circle (strictly, but L-BFGS-B handles bounds)
    # We can use bounds to keep them inside [-1, 1] roughly, but we need circle constraint.
    # L-BFGS-B doesn't support nonlinear constraints.
    # However, the soft-min objective will penalize points going outside heavily if beta is high.
    # To be safe, we can add a large penalty if any point is outside.
    # Or, we can just rely on the gradient of the boundary term in the soft-min.
    
    # Let's try standard L-BFGS-B first.
    
    def objective(x):
        val, grad = _sqp_objective_and_grad(x, beta=50.0)
        # Add penalty if any point is outside
        centers = x.reshape(N, 2)
        norms = np.linalg.norm(centers, axis=1)
        penalty = np.sum(np.maximum(0, norms - 1.0)**2) * 10.0
        # Gradient of penalty
        grad_pen = np.zeros_like(x)
        out_mask = norms > 1.0
        if np.any(out_mask):
            # d/dx ( (||x||-1)^2 ) = 2(||x||-1) * x/||x||
            # For each i:
            idx = np.where(out_mask)[0]
            for i in idx:
                grad_pen[2*i] += 2 * (norms[i] - 1.0) * centers[i, 0] / norms[i] * 10.0
                grad_pen[2*i+1] += 2 * (norms[i] - 1.0) * centers[i, 1] / norms[i] * 10.0
        return val + penalty, grad + grad_pen

    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000})
        if res.success or res.fun < 1.0:
            return res.x.reshape(N, 2)
    except Exception:
        pass
        
    return centers_init

def construct():
    deadline = time.time() + 110.0
    
    inits = [
        _hexagonal_packing(),
        _concentric_packing(),
        _spiral_packing(),
        _random_packing(),
        _random_packing()
    ]
    
    best_centers = inits[0]
    best_r = 0.0
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # 1. Physics relaxation to get a good starting point
        relaxed = _physics_relax(init, iterations=200)
        
        # 2. L-BFGS-B optimization
        optimized = _optimize_lagrangian(relaxed, deadline - 5)
        
        # 3. Evaluate
        r = _compute_radius(optimized)
        
        if r > best_r:
            best_r = r
            best_centers = optimized
            
    return best_centers

if __name__ == "__main__":
    centers = construct()
    print(_compute_radius(centers))
