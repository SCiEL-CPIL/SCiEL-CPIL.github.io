import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def get_radius(centers):
        """Compute the exact radius for given centers."""
        # Wall distance
        wall_dists = 0.5 - np.abs(centers)
        r_wall = np.min(wall_dists)
        
        # Pairwise distance
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        
        return min(r_wall, r_pair)

    def simulate(centers, num_steps=2000, eta0=0.01, T0=0.01, T_end=1e-6, sigma=0.05):
        """
        Run a simulated annealing simulation with exponential repulsion.
        centers: initial positions (N, 3)
        """
        c = centers.copy()
        best_c = c.copy()
        best_r = get_radius(c)
        
        for step in range(num_steps):
            # Temperature schedule: exponential decay
            T = T0 * (T_end / T0) ** (step / num_steps)
            eta = eta0 * (0.999 ** step) # Decay learning rate
            
            # Compute pairwise distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1e9) # Avoid division by zero
            
            # Force magnitude: (1/sigma) * exp(-d/sigma)
            # Direction: diff / d
            # F_ij = (1/sigma) * exp(-d/sigma) * (diff / d)
            
            safe_dists = np.maximum(dists, 0.001)
            exp_term = np.exp(-safe_dists / sigma)
            inv_d = 1.0 / safe_dists
            
            # Force matrix
            # F[i, j] is force on i due to j
            # Magnitude is (1/sigma) * exp_term
            # Direction is diff * inv_d
            forces = (1.0 / sigma) * exp_term[:, :, np.newaxis] * diff * inv_d[:, :, np.newaxis]
            
            # Sum forces from all other spheres
            F_pairs = np.sum(forces, axis=1)
            
            # Wall forces
            # Distance to walls: 0.5 - |c|
            wall_dists = 0.5 - np.abs(c)
            safe_wall_dists = np.maximum(wall_dists, 0.001)
            
            # Force magnitude: (1/sigma) * exp(-d_wall/sigma)
            wall_exp_term = np.exp(-safe_wall_dists / sigma)
            
            # Force direction is towards center: -sign(c)
            F_walls = (1.0 / sigma) * wall_exp_term[:, np.newaxis] * (-np.sign(c))
            
            # Total force
            F_total = F_pairs + F_walls
            
            # Update positions
            # c = c + eta * F_total + noise
            noise = np.random.normal(0, np.sqrt(2 * T), size=c.shape)
            c = c + eta * F_total + noise
            
            # Clip to box
            c = np.clip(c, -0.5, 0.5)
            
            # Check radius periodically to save time
            if step % 10 == 0:
                r = get_radius(c)
                if r > best_r:
                    best_r = r
                    best_c = c.copy()
                    
        # Final check
        r = get_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
            
        return best_c, best_r

    # Initial Grid
    # 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.45, 0.45, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.linspace(-0.45, 0.45, nz)
    X, Y, Z = np.meshgrid(xs, ys, ys, indexing='ij')
    base_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    best_centers = base_centers.copy()
    best_radius = get_radius(best_centers)
    
    np.random.seed(42)
    
    # Multiple restarts
    num_restarts = 20 # Adjust based on time per simulation
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Perturb the grid
        scale = 0.05 * (1 - i / num_restarts) + 0.01
        perturbed = base_centers + np.random.normal(0, scale, size=base_centers.shape)
        perturbed = np.clip(perturbed, -0.5, 0.5)
        
        # Run simulation
        steps = 1000
        if i > 10:
            steps = 500
            
        c, r = simulate(perturbed, num_steps=steps, eta0=0.02, T0=0.05, T_end=1e-5, sigma=0.05)
        
        if r > best_radius:
            best_radius = r
            best_centers = c.copy()
            
    # Final polish: Run a few more steps on the best solution found
    if time.time() < deadline - 5:
        c, r = simulate(best_centers, num_steps=500, eta0=0.005, T0=0.01, T_end=1e-6, sigma=0.05)
        if r > best_radius:
            best_radius = r
            best_centers = c.copy()

    return best_centers
