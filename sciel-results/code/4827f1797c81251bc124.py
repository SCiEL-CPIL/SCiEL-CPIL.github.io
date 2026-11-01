"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    # c is (N, 3)
    # Pairwise distances
    # Using broadcasting for vectorization
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    # Exclude self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Wall distances
    # Distance to each of the 6 walls
    # x walls: 0.5 - c[:,0], 0.5 + c[:,0]
    # Since cube is [-0.5, 0.5], dist to wall is min(0.5 - x, 0.5 + x) = 0.5 - |x|
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _energy_and_grad(c):
    """
    Compute a repulsive energy and its gradient.
    We want to maximize min distance. 
    A common proxy is to minimize sum of 1/d^2 for pairs.
    Also add a large penalty if points go outside the cube.
    
    Energy E = sum_{i<j} 1/d_ij^2 + sum_i penalty_i
    
    Gradient dE/dc_i = sum_{j!=i} 2/d_ij^4 * (c_i - c_j) + dPenalty/dc_i
    """
    n = len(c)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    # Avoid division by zero
    np.fill_diagonal(dists_sq, 1e-10)
    
    # Energy: sum of 1/d^2
    # We compute sum_{i,j} 1/d_ij^2, which is 2 * sum_{i<j} 1/d_ij^2
    inv_dists_sq = 1.0 / dists_sq
    np.fill_diagonal(inv_dists_sq, 0)
    
    energy = 0.5 * np.sum(inv_dists_sq)
    
    # Gradient of pair energy
    # d(1/d^2)/dc_i = -2 * (c_i - c_j) / d^4
    # Sum over j
    # grad[i] = sum_{j!=i} 2 * (c_i - c_j) / dists_sq[i,j]^2
    # Note: dists_sq[i,j] is squared distance. So d^4 = dists_sq^2.
    # Term is 2 * diff[i,j] / dists_sq[i,j]^2
    # But wait, derivative of (c_i - c_j)^2 is 2(c_i - c_j).
    # d/dx (1/x^2) = -2/x^3. Here x = d^2.
    # Let's use chain rule properly.
    # E = 1/d^2. dE/dd = -2/d^3. dd/dc_i = (c_i - c_j)/d.
    # dE/dc_i = -2/d^3 * (c_i - c_j)/d = -2(c_i - c_j)/d^4 = 2(c_j - c_i)/d^4.
    # So gradient points AWAY from other particles? No, force is attractive in this metric?
    # We want to MINIMIZE energy. If particles are close, energy is high.
    # Gradient of energy w.r.t c_i should point in direction that INCREASES energy.
    # So -gradient points in direction that DECREASES energy (moves apart).
    
    # Let's just compute the force vector directly.
    # Force on i from j is repulsive: F_ij = 2 * (c_i - c_j) / d^4
    # This force pushes i away from j.
    # Gradient of Energy is sum of these forces?
    # E = sum 1/d^2. 
    # dE/dc_i = sum_j d(1/d_ij^2)/dc_i = sum_j 2(c_j - c_i)/d_ij^4.
    # So grad = - sum_j 2(c_i - c_j)/d_ij^4.
    
    # Vectorized gradient calculation
    # diff is c_i - c_j
    # We need sum over j of 2 * diff[i,j] / dists_sq[i,j]^2
    # dists_sq is (N, N)
    # coeff = 2 / dists_sq^2
    coeff = 2.0 / (dists_sq * dists_sq)
    np.fill_diagonal(coeff, 0)
    
    # grad[i] = sum_j coeff[i,j] * (c_j - c_i) = - sum_j coeff[i,j] * (c_i - c_j)
    # grad = - (coeff * diff).sum(axis=1)
    grad = -np.sum(coeff * diff, axis=1)
    
    # Boundary penalty
    # If c_i is outside [-0.5, 0.5], add large penalty.
    # Soft penalty: if c > 0.5, penalty = (c-0.5)^2 * k. Grad = 2(c-0.5)k.
    # If c < -0.5, penalty = (c+0.5)^2 * k. Grad = 2(c+0.5)k.
    k = 1000.0
    c_clipped = np.clip(c, -0.5, 0.5)
    # Penalty for going out
    # d = c - c_clipped is 0 if inside, else c - 0.5 or c - (-0.5)
    # Actually simpler: 
    # if c > 0.5: term = c - 0.5
    # if c < -0.5: term = c + 0.5
    # else 0
    pos_excess = np.maximum(c - 0.5, 0)
    neg_excess = np.maximum(-0.5 - c, 0)
    
    energy += k * (np.sum(pos_excess**2) + np.sum(neg_excess**2))
    
    grad += k * (2 * pos_excess - 2 * neg_excess)
    
    return energy, grad

def _adam_optimize(c_init, steps=2000, lr=0.01, time_limit=None):
    """
    Optimize configuration using Adam optimizer on the repulsive energy.
    """
    c = c_init.copy()
    n = len(c)
    
    # Adam state
    m = np.zeros_like(c)
    v = np.zeros_like(c)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    t = 0
    
    start_time = time.time()
    best_c = c.copy()
    best_energy = np.inf
    
    for step in range(steps):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        energy, grad = _energy_and_grad(c)
        
        # Update best if energy is lower (proxy for higher radius)
        # Note: Lower energy = larger min distance.
        if energy < best_energy:
            best_energy = energy
            best_c = c.copy()
            
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        c -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Clip to cube to ensure validity during search (helps stability)
        c = np.clip(c, -0.5, 0.5)
        
    return best_c

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    np.random.seed(42)
    
    best_c = None
    best_r = -np.inf
    
    # Multiple restarts
    # Estimate how many restarts we can do.
    # Each optimization takes ~0.5s to 2s depending on steps.
    # Let's try to do as many as possible.
    
    while time.time() < deadline - 5.0:
        # Random initialization
        # Use a jittered grid or purely random?
        # Purely random often works well with strong repulsion.
        c_init = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
        # Adaptive steps based on remaining time
        # If lots of time left, do more steps.
        time_left = deadline - time.time()
        if time_left > 10:
            steps = 1000
        elif time_left > 5:
            steps = 500
        else:
            steps = 200
            
        # Optimize
        c_opt = _adam_optimize(c_init, steps=steps, lr=0.02, time_limit=time_left - 5.0)
        
        # Evaluate
        r = _compute_radius(c_opt)
        
        if r > best_r:
            best_r = r
            best_c = c_opt.copy()
            
    return best_c

# EVOLVE-BLOCK-END
