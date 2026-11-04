"""Optimized packing of 100 equal spheres in a unit cube."""
import numpy as np
import time

N = 100
CUBE_HALF = 0.5


def _compute_radius(centres):
    """Compute the maximum feasible radius for given centres."""
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = np.minimum(
        CUBE_HALF - np.abs(centres),
        axis=1
    ).min()
    
    return min(min_pair_dist / 2.0, wall_dists)


def _fcc_lattice_init(n):
    """Initialize with FCC lattice points, then trim/pad to n."""
    # FCC lattice: points at (i,j,k) where i+j+k is even, scaled
    # We need a dense starting configuration. Let's use a body-centered cubic or FCC.
    # FCC basis vectors: a/2(0,1,1), a/2(1,0,1), a/2(1,1,0)
    # Let's just generate a grid and perturb, or use a known good structure.
    # For simplicity and robustness, let's use a simple cubic grid but with slight random perturbation
    # to break symmetry, or better, use a jittered grid.
    
    # Actually, let's use a simple approach: generate points on a grid and add small noise
    m = int(np.ceil(N ** (1.0 / 3.0)))
    # We need at least N points, so use m x m x m grid
    ticks = (np.arange(m) + 0.5) / m
    # Map to [-0.5, 0.5]
    ticks = ticks - 0.5
    
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    if len(grid) >= n:
        return grid[:n].copy()
    else:
        # Pad with random points
        extra = np.random.uniform(-0.5, 0.5, size=(n - len(grid), 3))
        return np.vstack([grid, extra])


def _soft_min_surrogate(centres, temperature):
    """
    Compute a smooth surrogate for the minimum radius.
    Uses log-sum-exp for soft-min of pairwise distances and wall distances.
    Returns the surrogate value and gradient.
    """
    eps = 1e-12
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq + eps)
    
    # Wall distances
    wall_dists = CUBE_HALF - np.abs(centres)
    wall_dists = np.min(wall_dists, axis=1)
    
    # Soft-min of all pairwise distances / 2 and wall distances
    # We want to maximize the minimum of these quantities
    # So we minimize the soft-max of their negatives, or equivalently
    # use soft-min directly
    
    # Combine all constraints into one vector
    pair_dists_half = dists / 2.0
    # Flatten pairwise (excluding diagonal)
    mask = ~np.eye(N, dtype=bool)
    pair_flat = pair_dists_half[mask]
    all_dists = np.concatenate([pair_flat, wall_dists])
    
    # Soft-min: -logsumexp(-x / T) * T
    # For numerical stability
    x = all_dists
    x_max = np.max(x)
    log_sum_exp = x_max + np.log(np.sum(np.exp(-(x - x_max) / temperature)))
    soft_min = -log_sum_exp * temperature
    
    # Gradient w.r.t. centres
    # d(soft_min)/d(x_i) = -exp(-x_i/T) / sum(exp(-x_j/T))
    weights = np.exp(-(x - x_max) / temperature)
    weights /= np.sum(weights)
    grad_x = -weights
    
    # Now backpropagate to centres
    grad_centres = np.zeros_like(centres)
    
    # Pairwise part
    # grad_x[:N*(N-1)] corresponds to pair distances
    pair_grads = grad_x[:N*(N-1)].reshape(N, N-1)
    # This is tricky to reshape back. Let's compute directly.
    
    # For each pair (i,j), i<j, the contribution to grad_centres[i] and grad_centres[j]
    # Let's recompute more carefully
    
    # Actually, let's compute gradient directly from the formula
    # soft_min = -T * log(sum_j exp(-x_j/T))
    # d(soft_min)/d(c_i) = sum_j [d(soft_min)/d(x_j) * dx_j/d(c_i)]
    
    # For pairwise distance d_ij = |c_i - c_j| / 2
    # dx_ij/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    # dx_ij/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
    
    # For wall distance w_i = min_k (0.5 - |c_ik|)
    # dx_i/d(c_ik) = -sign(c_ik) for the active k
    
    # This is getting complex. Let's use a simpler approach: 
    # use the hard min but with a small noise for smoothing, or just use 
    # the actual min and do gradient-based optimization with subgradients.
    
    # Alternative: use the energy function approach
    # Minimize sum of (1/d_ij^2) for pairs that are too close, etc.
    
    return soft_min


def construct():
    """Main optimization routine."""
    start_time = time.time()
    timeout = 140  # Safety margin
    
    # Initialize with FCC-like structure
    np.random.seed(42)
    centres = _fcc_lattice_init(N)
    
    # Compute initial radius
    current_radius = _compute_radius(centres)
    best_centres = centres.copy()
    best_radius = current_radius
    
    # Adam optimizer state
    m = np.zeros_like(centres)
    v = np.zeros_like(centres)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    lr = 0.01
    t = 0
    
    # Gradient descent with soft constraints
    # We'll use a penalty method: minimize -radius + penalty for violations
    
    max_iterations = 50000
    
    for it in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        # Compute gradient of the objective
        # Objective: maximize min(min_pair_dist/2, min_wall_dist)
        # Use subgradient: find the active constraint
        
        # Pairwise distances
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        # Wall distances
        wall_dists = CUBE_HALF - np.abs(centres)
        wall_dists = np.min(wall_dists, axis=1)
        
        # Find the minimum
        min_pair_dist = np.min(dists)
        min_wall_dist = np.min(wall_dists)
        
        # Gradient of -min_dist/2
        # If pair constraint is active:
        if min_pair_dist / 2.0 < min_wall_dist:
            # Find the pair achieving the minimum
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            if i == j:
                continue
            # Gradient of -|c_i - c_j|/2 w.r.t. c_i and c_j
            d = centres[i] - centres[j]
            norm = np.linalg.norm(d)
            if norm < 1e-10:
                continue
            grad = np.zeros_like(centres)
            grad[i] = -d / (2 * norm)
            grad[j] = d / (2 * norm)
        else:
            # Wall constraint is active
            idx = np.argmin(wall_dists)
            # Find which coordinate is closest to the wall
            c = centres[idx]
            abs_c = np.abs(c)
            min_coord = np.argmin(CUBE_HALF - abs_c)
            # Gradient of -(0.5 - |c_ik|) = sign(c_ik)
            grad = np.zeros_like(centres)
            grad[idx, min_coord] = np.sign(c[min_coord])
            if c[min_coord] == 0:
                grad[idx, min_coord] = 1.0
        
        # Adam update
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Step: we want to maximize radius, so we move in the direction of the gradient
        # of the radius. The gradient we computed is for -radius, so we move opposite.
        centres = centres - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Project to stay within bounds
        centres = np.clip(centres, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        
        # Check if we improved
        current_radius = _compute_radius(centres)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = centres.copy()
        
        # Decay learning rate
        lr = max(0.001, 0.01 / (1 + it * 0.001))
    
    return best_centres


if __name__ == "__main__":
    centres = construct()
    print("Radius:", _compute_radius(centres))
