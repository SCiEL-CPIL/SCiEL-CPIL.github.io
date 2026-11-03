import numpy as np
import scipy.optimize
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_radius = 0.0
    best_points = None
    
    # Neural Network parameters
    D_latent = 32  # Dimension of latent space
    H1 = 64
    H2 = 128
    
    # Objective function using soft-min
    # x: array of values (distances/2 and boundary distances)
    # eps: temperature for soft-min
    def soft_min(x, eps):
        # x is 1D array
        max_x = np.max(x)
        # Prevent overflow in exp
        # log(sum(exp(-eps * (x - max_x))))
        # = -eps * max_x + log(sum(exp(-eps * (x - max_x))))
        # M = max_x - (1/eps) * log(sum(exp(-eps * (x - max_x))))
        exp_terms = np.exp(-eps * (x - max_x))
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return max_x
        log_sum = np.log(sum_exp)
        return max_x - log_sum / eps

    def forward(z, W1, b1, W2, b2, W3, b3):
        # z: (D_latent,)
        # W1: (D_latent, H1), b1: (H1,)
        # W2: (H1, H2), b2: (H2,)
        # W3: (H2, 2*N), b3: (2*N,)
        
        h1 = np.tanh(z @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        out = h2 @ W3 + b3
        return out.reshape(N, 2)

    def objective(z, W1, b1, W2, b2, W3, b3, eps):
        points = forward(z, W1, b1, W2, b2, W3, b3)
        
        # Calculate pairwise distances
        # To save memory and time, we can compute all pairwise distances
        # N=300, so 300x300 matrix is 90k elements. Manageable.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Extract upper triangle (i < j)
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        
        # Boundary distances
        norms = np.sqrt(np.sum(points**2, axis=1))
        bound_dists = 1.0 - norms
        
        # Combine
        all_vals = np.concatenate([pair_dists / 2.0, bound_dists])
        
        # Soft min
        m = soft_min(all_vals, eps)
        
        # We want to maximize m, so minimize -m
        return -m

    def objective_grad(z, W1, b1, W2, b2, W3, b3, eps):
        # Finite difference gradient
        grad = np.zeros_like(z)
        h = 1e-6
        f0 = objective(z, W1, b1, W2, b2, W3, b3, eps)
        for i in range(len(z)):
            z_pert = z.copy()
            z_pert[i] += h
            f_pert = objective(z_pert, W1, b1, W2, b2, W3, b3, eps)
            grad[i] = (f_pert - f0) / h
        return grad

    # Restart loop
    restarts = 0
    eps_start = 10.0  # Initial soft-min temperature (lower = sharper min)
    
    while time.time() < deadline:
        restarts += 1
        
        # Initialize weights randomly
        # Use small initialization to avoid saturation in tanh
        scale1 = np.sqrt(2.0 / D_latent)
        W1 = np.random.randn(D_latent, H1) * scale1
        b1 = np.zeros(H1)
        
        scale2 = np.sqrt(2.0 / H1)
        W2 = np.random.randn(H1, H2) * scale2
        b2 = np.zeros(H2)
        
        scale3 = np.sqrt(2.0 / H2)
        W3 = np.random.randn(H2, 2 * N) * scale3
        b3 = np.zeros(2 * N)
        
        # Initial latent vector
        z0 = np.random.randn(D_latent) * 0.1
        
        # Annealing: start with high eps (smooth), end with low eps (sharp)
        # For each restart, we do a short annealing schedule
        # But to keep it simple within time budget, we can just use a fixed eps
        # or a few steps of annealing.
        # Let's try a few annealing steps per restart.
        
        current_eps = eps_start
        current_z = z0.copy()
        
        # Perform a few optimization steps with decreasing eps
        # This helps escape local minima in the soft-min landscape
        annealing_steps = 5
        for step in range(annealing_steps):
            # Decrease eps
            current_eps *= 0.5
            
            # Optimize z
            res = scipy.optimize.minimize(
                objective, 
                current_z, 
                args=(W1, b1, W2, b2, W3, b3, current_eps),
                method='L-BFGS-B',
                jac=objective_grad,
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.success:
                current_z = res.x
            
            # Check time
            if time.time() > deadline - 5.0:
                break
        
        # Final evaluation with hard min
        final_points = forward(current_z, W1, b1, W2, b2, W3, b3)
        
        # Calculate true radius
        diff = final_points[:, np.newaxis, :] - final_points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        norms = np.sqrt(np.sum(final_points**2, axis=1))
        bound_dists = 1.0 - norms
        
        r = min(np.min(pair_dists) / 2.0, np.min(bound_dists))
        
        if r > best_radius:
            best_radius = r
            best_points = final_points.copy()
            
        # If time is running out, break
        if time.time() > deadline - 10.0:
            break

    # If no valid points found, return seed-like result
    if best_points is None:
        # Fallback to hex lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98

    return best_points
