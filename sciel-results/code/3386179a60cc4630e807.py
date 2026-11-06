import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def physics_energy(x, repulsive_power=4, wall_strength=100.0):
    """
    Energy function for physics-based relaxation.
    x: shape (N, 2)
    Returns energy value and gradient.
    """
    x = x.reshape(N, 2)
    
    # 1. Repulsive energy: sum_{i<j} 1/d_ij^repulsive_power
    # We use a smooth approximation or direct calculation.
    # Direct calculation with safety for d=0.
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    # Mask diagonal
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    dists_sq_pair = dists_sq[mask]
    dists_pair = np.sqrt(dists_sq_pair)
    
    # Avoid division by zero
    safe_dists = np.maximum(dists_pair, 1e-6)
    
    # Energy term: sum (1/d^k)
    # Gradient: d/dx_i (1/||x_i - x_j||^k) = -k * (x_i - x_j) / ||x_i - x_j||^(k+2)
    # Note: The gradient of the sum over pairs for point i is the sum over j != i.
    
    energy_rep = np.sum(safe_dists ** (-repulsive_power))
    
    grad = np.zeros_like(x)
    
    # Compute gradients for repulsion
    # We need to accumulate gradients for each pair.
    idx_i, idx_j = np.where(mask)
    
    # dists_pair corresponds to the flattened mask
    # diff_pair = x[idx_i] - x[idx_j]
    diff_pair = x[idx_i] - x[idx_j]
    
    # dE/dx_i = -k * (x_i - x_j) / d^(k+2)
    # dE/dx_j = -k * (x_j - x_i) / d^(k+2) = k * (x_i - x_j) / d^(k+2)
    
    # Calculate factor for each pair
    # factor = k / d^(k+2)
    # We have safe_dists
    factor = repulsive_power * (safe_dists ** (-(repulsive_power + 2)))
    
    # grad_i += -factor * (x_i - x_j)
    # grad_j += factor * (x_i - x_j)
    
    # Vectorized accumulation using np.add.at
    # grad[idx_i] -= factor[:, np.newaxis] * diff_pair
    # grad[idx_j] += factor[:, np.newaxis] * diff_pair
    
    np.add.at(grad, idx_i, -factor[:, np.newaxis] * diff_pair)
    np.add.at(grad, idx_j, factor[:, np.newaxis] * diff_pair)
    
    # 2. Wall potential: sum_i max(0, ||x_i|| - 1)^2
    # This keeps points inside the unit circle.
    # If we use a soft wall, we penalize points outside.
    # To ensure they stay inside, we can use a potential that grows rapidly outside.
    # E_wall = sum (max(0, ||x_i|| - 1)^2)
    # Gradient: if ||x_i|| > 1, dE/dx_i = 2 * (||x_i|| - 1) * (x_i / ||x_i||)
    
    norms = np.linalg.norm(x, axis=1)
    outside = norms > 1.0
    
    energy_wall = 0.0
    if np.any(outside):
        over = norms[outside] - 1.0
        energy_wall = np.sum(over ** 2) * wall_strength
        
        # Gradient contribution
        # grad_i += wall_strength * 2 * over * (x_i / norm_i)
        grad[outside] += wall_strength * 2.0 * over[:, np.newaxis] * (x[outside] / norms[outside, np.newaxis])
        
    total_energy = energy_rep + energy_wall
    
    return total_energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin (10s buffer)
    
    best_x = None
    best_score = -np.inf
    
    # Initializations
    inits = []
    
    # 1. Random init
    def random_init():
        angles = np.random.uniform(0, 2*np.pi, N)
        r = np.sqrt(np.random.uniform(0, 0.95, N)) # Keep well inside
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        return np.column_stack((x, y))
    
    inits.append(random_init())
    
    # 2. Lattice init (Robust version)
    try:
        spacing = 0.25
        k = int(1.0 / spacing) + 2
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    points.append((x, y))
        
        points = np.array(points)
        if len(points) >= N:
            # Shuffle to avoid bias
            np.random.shuffle(points)
            p0 = points[:N].copy()
            # Add small noise
            p0 += np.random.normal(0, 0.01, (N, 2))
            # Clip to unit disk
            norms = np.linalg.norm(p0, axis=1)
            mask = norms > 0.99
            if np.any(mask):
                p0[mask] = p0[mask] * (0.99 / norms[mask])
            inits.append(p0)
    except Exception:
        pass

    # 3. Ring init
    try:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        r = 0.8
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        p_ring = np.column_stack((x, y)) + np.random.normal(0, 0.05, (N, 2))
        norms = np.linalg.norm(p_ring, axis=1)
        mask = norms > 0.99
        if np.any(mask):
            p_ring[mask] = p_ring[mask] * (0.99 / norms[mask])
        inits.append(p_ring)
    except Exception:
        pass

    # 4. More random inits
    for _ in range(5):
        if time.time() > deadline:
            break
        inits.append(random_init())

    if not inits:
        # Fallback
        return random_init()

    for i_init, x0 in enumerate(inits):
        if time.time() > deadline:
            break
            
        x_curr = x0.copy()
        
        try:
            # Optimize using L-BFGS-B on the physics energy
            res = minimize(
                physics_energy, 
                x_curr, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 20}
            )
            
            x_opt = res.x.reshape(N, 2)
            
            # Calculate actual score (hard min)
            dists_bound = 1.0 - np.linalg.norm(x_opt, axis=1)
            diff = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
            dists_pair = np.linalg.norm(diff, axis=2) / 2.0
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            min_pair = np.min(dists_pair[mask])
            min_bound = np.min(dists_bound)
            score = min(min_pair, min_bound)
            
            # Check validity
            if min_bound < -1e-8 or min_pair < -1e-8:
                continue
            if not np.all(np.isfinite(x_opt)):
                continue
                
            if score > best_score:
                best_score = score
                best_x = x_opt
                
        except Exception:
            pass

    if best_x is None:
        # Fallback to first init
        return inits[0]
        
    return best_x

# EVOLVE-BLOCK-END
