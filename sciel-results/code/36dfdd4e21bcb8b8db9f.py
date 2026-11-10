import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Exploit symmetries to reduce dimensionality from 42 to 24.
       - Translation: Fix center of mass to 0 (removed by normalization).
       - Scaling: Normalize d_max to 1 (handled in objective).
       - Rotation: Represent points via Gram matrix (dot products) or 
         directly optimize pairwise distances and reconstruct? 
         Actually, optimizing pairwise distances is tricky due to Euclidean constraints.
         
    Alternative Structural Change:
    Use a "Soft-Min/Max" differentiable objective.
    Objective: Maximize log(sum(exp(-d^2/T))) - log(sum(exp(d^2/T)))
    This approximates d_min - d_max (or similar) and is smooth.
    However, the task asks for a specific structural change:
    "Model the problem as a non-convex quadratic programming task by maximizing 
    the minimum pairwise squared distance subject to the maximum pairwise squared 
    distance being fixed to 1, using a barrier function to enforce the upper bound 
    and solving via BFGS on the unconstrained log-barrier augmented objective."
    
    Let's implement that specific directive.
    
    Variables: x in R^{14x3}.
    Objective: Maximize t
    Subject to: 
      d_ij^2 >= t  for all i<j
      d_ij^2 <= 1  for all i<j (normalized max distance)
      sum(x) = 0 (center of mass)
      
    This is a Quadratic Program (QP) but with 'maximize min' which is not standard QP.
    The directive suggests: "maximizing the minimum pairwise squared distance ... using a barrier function".
    
    Let f(x) = min_{i<j} d_ij^2.
    We want to maximize f(x) subject to max_{i<j} d_ij^2 <= 1.
    
    Since f(x) is non-smooth, we can use a smooth approximation:
    g(x, mu) = (1/mu) * log( sum_{i<j} exp(-mu * d_ij^2) )
    As mu -> infinity, g(x, mu) -> min_{i<j} d_ij^2.
    
    Constraint: max_{i<j} d_ij^2 <= 1.
    Smooth approximation for max:
    h(x, nu) = (1/nu) * log( sum_{i<j} exp(nu * d_ij^2) )
    Constraint: h(x, nu) <= 1 + epsilon? Or use barrier:
    Barrier term: -log(1 - h(x, nu)) if h < 1, else large penalty.
    
    Actually, the directive says: "maximizing the minimum ... subject to the maximum ... being fixed to 1".
    If we fix max to 1, we are effectively scaling the configuration.
    So the problem reduces to: Maximize d_min^2 subject to d_max^2 = 1.
    This is equivalent to maximizing d_min^2 / d_max^2.
    
    Let's use the barrier approach on the ratio directly?
    No, the directive is specific:
    "maximizing the minimum pairwise squared distance subject to the maximum pairwise squared distance being fixed to 1, using a barrier function to enforce the upper bound and solving via BFGS on the unconstrained log-barrier augmented objective."
    
    So:
    Maximize:  g(x, mu)  (smooth min of d^2)
    Subject to: h(x, nu) <= 1 (smooth max of d^2)
    
    Augmented Objective for BFGS (Minimization):
    J(x) = -g(x, mu) - gamma * log(1 - h(x, nu))
    
    We minimize J(x).
    
    Steps:
    1. Generate diverse initial points.
    2. For each, run BFGS minimization of J(x).
    3. Keep the best result.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 620
    
    # Precompute indices for pairwise distances
    # There are 91 pairs.
    idx_i, idx_j = np.triu_indices(n, k=1)
    
    def compute_dists_sq(x):
        """
        x: (n, d)
        Returns: (91,) array of squared distances.
        """
        # Efficient calculation of all pairwise squared distances
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
        norms = np.sum(x**2, axis=1)
        # Gram matrix
        G = x @ x.T
        dists_sq = norms[:, None] + norms[None, :] - 2 * G
        # Extract upper triangle
        return dists_sq[idx_i, idx_j]
    
    def smooth_min(dists_sq, mu):
        """
        Smooth approximation of min(dists_sq).
        mu is the sharpness parameter.
        """
        # log-sum-exp trick for stability
        # min(a) = -lim_{mu->inf} (1/mu) log(sum(exp(-mu * a)))
        # Let's use: -1/mu * log(sum(exp(-mu * dists_sq)))
        # To avoid overflow, shift by min_val
        min_val = np.min(dists_sq)
        exp_terms = np.exp(-mu * (dists_sq - min_val))
        return min_val - (1.0 / mu) * np.log(np.sum(exp_terms))
        
    def smooth_max(dists_sq, nu):
        """
        Smooth approximation of max(dists_sq).
        max(a) = lim_{nu->inf} (1/nu) log(sum(exp(nu * a)))
        """
        max_val = np.max(dists_sq)
        exp_terms = np.exp(nu * (dists_sq - max_val))
        return max_val + (1.0 / nu) * np.log(np.sum(exp_terms))
        
    def objective_and_grad(x_flat):
        """
        Computes the augmented objective J(x) and its gradient.
        J(x) = -g(x, mu) - gamma * log(1 - h(x, nu))
        
        We need gradients of g and h.
        g(x) = min_d d^2 (approx)
        h(x) = max_d d^2 (approx)
        
        Let's derive gradients numerically or analytically?
        Analytical is better for BFGS.
        
        Let D be the vector of squared distances.
        D_k = ||x_i - x_j||^2 for pair k.
        dD_k/dx = 2(x_i - x_j) for coordinate i, -2(x_i - x_j) for coordinate j, 0 otherwise.
        
        g(x) = -1/mu * log(sum_k exp(-mu D_k))
        dg/dx = -1/mu * [ sum_k (-mu exp(-mu D_k) dD_k/dx) / sum_k exp(-mu D_k) ]
             = sum_k [ w_k dD_k/dx ]
        where w_k = exp(-mu D_k) / sum_j exp(-mu D_j).
        Note: sum w_k = 1. w_k are weights concentrated on the minimum distance.
        
        h(x) = 1/nu * log(sum_k exp(nu D_k))
        dh/dx = 1/nu * [ sum_k (nu exp(nu D_k) dD_k/dx) / sum_k exp(nu D_k) ]
             = sum_k [ v_k dD_k/dx ]
        where v_k = exp(nu D_k) / sum_j exp(nu D_j).
        v_k are weights concentrated on the maximum distance.
        
        Barrier term: B(x) = -gamma * log(1 - h(x))
        dB/dx = -gamma * (-1 / (1 - h)) * dh/dx
             = gamma / (1 - h) * dh/dx
        
        Total Gradient:
        dJ/dx = -dg/dx + dB/dx
             = - sum_k w_k dD_k/dx + (gamma / (1 - h)) * sum_k v_k dD_k/dx
        
        Let's implement this.
        """
        x = x_flat.reshape(n, d)
        
        # Compute D
        D = compute_dists_sq(x)
        
        # Parameters
        mu = 50.0  # Sharpness for min
        nu = 50.0  # Sharpness for max
        gamma = 10.0 # Barrier strength
        
        # Weights for min (w)
        # exp(-mu * D)
        # Shift for stability
        min_D = np.min(D)
        exp_neg_mu_D = np.exp(-mu * (D - min_D))
        w = exp_neg_mu_D / np.sum(exp_neg_mu_D)
        
        # Weights for max (v)
        max_D = np.max(D)
        exp_nu_D = np.exp(nu * (D - max_D))
        v = exp_nu_D / np.sum(exp_nu_D)
        
        # Compute g and h
        g_val = min_D - (1.0 / mu) * np.log(np.sum(exp_neg_mu_D))
        h_val = max_D + (1.0 / nu) * np.log(np.sum(exp_nu_D))
        
        # Check barrier validity
        if h_val >= 1.0:
            # If we violate the constraint, return a large penalty
            # and a gradient that pushes back
            # This is a soft constraint. BFGS might struggle if it goes too far.
            # We can add a quadratic penalty if h_val > 1.
            penalty = 1000.0 * (h_val - 1.0)**2
            # Gradient of penalty: 2000 * (h_val - 1) * dh/dx
            # We need dh/dx.
            # Let's compute dD/dx as a matrix of shape (91, n, d)
            # This is memory heavy but n=14, d=3, 91 pairs. 91*14*3 = 3822 floats. Fine.
            
            dD_dx = np.zeros((len(D), n, d))
            for k in range(len(D)):
                i = idx_i[k]
                j = idx_j[k]
                diff = x[i] - x[j]
                dD_dx[k, i, :] = 2.0 * diff
                dD_dx[k, j, :] = -2.0 * diff
            
            # dh/dx = sum_k v_k dD_k/dx
            dh_dx = np.tensordot(v, dD_dx, axes=(0, 0))
            
            grad_penalty = 2000.0 * (h_val - 1.0) * dh_dx
            return penalty + 100.0, grad_penalty.reshape(-1)
            
        # Compute gradient
        # dD_dx
        dD_dx = np.zeros((len(D), n, d))
        for k in range(len(D)):
            i = idx_i[k]
            j = idx_j[k]
            diff = x[i] - x[j]
            dD_dx[k, i, :] = 2.0 * diff
            dD_dx[k, j, :] = -2.0 * diff
            
        # dg/dx = sum w_k dD_k/dx
        dg_dx = np.tensordot(w, dD_dx, axes=(0, 0))
        
        # dh/dx = sum v_k dD_k/dx
        dh_dx = np.tensordot(v, dD_dx, axes=(0, 0))
        
        # Barrier gradient
        barrier_grad_factor = gamma / (1.0 - h_val)
        d_barrier_dx = barrier_grad_factor * dh_dx
        
        # Total gradient of J = -g - log(1-h)
        # dJ = -dg + d_barrier
        grad = -dg_dx + d_barrier_dx
        
        # Objective value
        obj_val = -g_val - gamma * np.log(1.0 - h_val)
        
        return obj_val, grad.reshape(-1)
        
    def run_optimization(x0):
        """
        Runs BFGS optimization from x0.
        """
        x0_flat = x0.flatten()
        
        # BFGS options
        options = {
            'maxiter': 500,
            'gtol': 1e-6
        }
        
        try:
            res = minimize(
                objective_and_grad, 
                x0_flat, 
                method='BFGS', 
                jac=True,
                options=options
            )
            return res.x.reshape(n, d)
        except Exception:
            return x0
    
    def get_score(pts):
        dists = pdist(pts, metric='sqeuclidean')
        if len(dists) == 0:
            return 0.0
        d_min_sq = np.min(dists)
        d_max_sq = np.max(dists)
        if d_max_sq < 1e-12:
            return 0.0
        return d_min_sq / d_max_sq

    # Generate initial populations
    # We need diverse starts.
    # 1. Random Gaussian
    # 2. Icosahedron + 2 random
    # 3. Sphere uniform
    
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts = ico_verts / norms
    
    initial_points = []
    
    # 5 Random Gaussian
    for _ in range(5):
        p = np.random.randn(n, d)
        initial_points.append(p)
        
    # 5 Icosahedron + 2 random on sphere
    for _ in range(5):
        p = ico_verts.copy()
        extra1 = np.random.randn(d)
        extra1 = extra1 / np.linalg.norm(extra1)
        extra2 = np.random.randn(d)
        extra2 = extra2 / np.linalg.norm(extra2)
        p = np.vstack([p, [extra1], [extra2]])
        p += np.random.randn(n, d) * 0.05
        initial_points.append(p)
        
    # 5 Random Uniform Sphere
    for _ in range(5):
        p = np.random.randn(n, d)
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        initial_points.append(p)
        
    best_score = 0.0
    best_points = initial_points[0]
    
    for i, p0 in enumerate(initial_points):
        if time.time() > deadline - 60:
            break
            
        # Run optimization
        p_opt = run_optimization(p0)
        
        # Evaluate
        score = get_score(p_opt)
        
        if score > best_score:
            best_score = score
            best_points = p_opt
            
    return best_points
