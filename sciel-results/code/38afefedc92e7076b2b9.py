import numpy as np
from scipy.spatial.distance import cdist

N = 50


def construct():
    """
    Construct centers for 50 equal spheres in the unit cube [-0.5, 0.5]^3.
    Uses a BCC lattice initialization followed by a custom repulsive force
    optimization to maximize the minimum distance between centers and to the cube boundaries.
    """
    # 1. Initialize on a BCC lattice
    # 3x3x3 BCC cells give 54 points. We need 50.
    # Cell size a = 1/3.
    a = 1.0 / 3.0
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Corner
                points.append([i * a - 0.5, j * a - 0.5, k * a - 0.5])
                # Body center
                points.append([(i + 0.5) * a - 0.5, (j + 0.5) * a - 0.5, (k + 0.5) * a - 0.5])
    
    points = np.array(points)
    # We have 54 points. Randomly select 50 to break symmetry and start optimization.
    # Using a fixed seed for reproducibility
    np.random.seed(42)
    indices = np.random.choice(len(points), N, replace=False)
    centers = points[indices].copy()
    
    # Perturb to ensure points are strictly inside the cube and break symmetry
    centers += np.random.uniform(-0.02, 0.02, centers.shape)
    centers = np.clip(centers, -0.49, 0.49)

    # 2. Optimization: Repulsive Force / Langevin Dynamics
    # We want to maximize the minimum distance.
    # We simulate repulsion between all pairs.
    
    # Parameters
    steps = 20000
    learning_rate = 0.01
    cooling_rate = 0.99995
    
    for step in range(steps):
        # Calculate forces
        # Vectorized force calculation
        # v[i,j] = c_i - c_j
        v = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # d^2 matrix
        d2 = np.sum(v**2, axis=2)
        np.fill_diagonal(d2, np.inf)
        
        # Force is proportional to 1/d^2 (Coulomb-like)
        # F = v / d^3
        # d^3 = (d^2)^(3/2)
        d3 = d2 ** 1.5
        # Avoid division by zero
        d3_safe = np.where(d3 < 1e-10, 1e-10, d3)
        
        force_matrix = v / d3_safe[:, :, np.newaxis]
        
        # Sum over neighbors (axis 1)
        repulsive_forces = np.sum(force_matrix, axis=1)
        
        # Update positions
        centers += learning_rate * repulsive_forces
        
        # Add small noise to escape local minima
        noise = np.random.normal(0, 0.0005, centers.shape)
        centers += noise
        
        # Clip to box
        centers = np.clip(centers, -0.499999, 0.499999)
        
        # Cool down
        learning_rate *= cooling_rate

    # Final polish: A few steps of steepest descent on the smoothed objective?
    # Or just return the result.
    
    # Ensure valid shape
    return centers
