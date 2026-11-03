"""Packing 200 equal circles in a unit square via repulsive-potential energy minimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(x):
    """Compute the achievable common radius for centre array x of shape (N,2)."""
    if x.shape != (N, 2) or not np.all(np.isfinite(x)):
        return 0.0
    # wall distances
    dx = 0.5 - np.abs(x[:, 0])
    dy = 0.5 - np.abs(x[:, 1])
    wall = np.minimum(np.min(dx), np.min(dy))
    if wall <= 0:
        return 0.0
    # pair distances
    # Use chunked computation to save memory for N=200
    # Actually N=200 is small enough for full pairwise
    d = np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    # zero out diagonal
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return min(wall, min_pair)


def _energy(x, beta=50.0):
    """
    Smooth surrogate energy: soft-min of all constraints.
    Lower energy = larger radius.
    """
    dx = 0.5 - np.abs(x[:, 0])
    dy = 0.5 - np.abs(x[:, 1])
    wall = np.minimum(dx, dy)  # shape (N,)
    
    # Pairwise distances
    d = np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    pair_dists = d / 2.0  # shape (N, N)
    
    # Soft-min of wall distances
    w_min = -np.log(np.sum(np.exp(-beta * wall))) / beta
    
    # Soft-min of pair distances
    p_min = -np.log(np.sum(np.exp(-beta * pair_dists))) / beta
    
    # Overall soft-min
    energy = -np.log(np.exp(beta * w_min) + np.exp(beta * p_min)) / beta
    return energy


def _grad_energy(x, beta=50.0):
    """
    Gradient of the soft-min energy w.r.t. x.
    """
    dx = 0.5 - np.abs(x[:, 0])
    dy = 0.5 - np.abs(x[:, 1])
    wall = np.minimum(dx, dy)  # shape (N,)
    
    # Pairwise
    d = np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    pair_dists = d / 2.0
    
    # Soft-min of wall
    w_exp = np.exp(-beta * wall)
    w_sum = np.sum(w_exp)
    w_min = -np.log(w_sum) / beta
    # d(energy)/d(wall_i) contribution
    # energy = -log(exp(b*w_min)+exp(b*p_min))/b
    # Let's compute via chain rule more carefully using the combined soft-min
    
    # Combined: E = -log(sum_exp)/b where sum_exp = sum(exp(-b*wall)) + sum(exp(-b*pair))
    total_exp = np.sum(np.exp(-beta * wall)) + np.sum(np.exp(-beta * pair_dists))
    E = -np.log(total_exp) / beta
    
    # Gradient: dE/dx = (1/b) * d/dx[-log(total_exp)] = -(1/b) * (1/total_exp) * d(total_exp)/dx
    # d(total_exp)/dx = sum over terms of exp(-b*val) * (-b) * d(val)/dx
    
    # Wall gradient
    # wall_i = min(0.5-|x_i0|, 0.5-|x_i1|)
    # d(wall_i)/dx_i0 = -sign(x_i0) if dx<dy else 0
    # d(wall_i)/dx_i1 = -sign(x_i1) if dy<dx else 0
    dw_dx = np.zeros_like(x)
    for i in range(N):
        if dx[i] < dy[i]:
            dw_dx[i, 0] = -np.sign(x[i, 0]) if x[i, 0] != 0 else 1.0
        else:
            dw_dx[i, 1] = -np.sign(x[i, 1]) if x[i, 1] != 0 else 1.0
    
    wall_contrib = np.exp(-beta * wall) * (-beta) * dw_dx  # shape (N,2)
    
    # Pair gradient
    # pair_dists[i,j] = ||x_i - x_j||/2
    # d(pair_dists[i,j])/dx_i = (x_i - x_j)/||x_i - x_j||
    # d(pair_dists[i,j])/dx_j = (x_j - x_i)/||x_i - x_j||
    # Only off-diagonal
    diff = x[:, None, :] - x[None, :, :]  # (N,N,2)
    norms = np.sqrt(np.sum(diff**2, axis=-1))  # (N,N)
    np.fill_diagonal(norms, 1.0)  # avoid div by zero
    unit = diff / norms[:, :, None]  # (N,N,2)
    
    pair_exp = np.exp(-beta * pair_dists)
    np.fill_diagonal(pair_exp, 0.0)
    
    # For each i, sum over j of pair_exp[i,j] * (-beta) * unit[i,j]
    pair_contrib = np.sum(pair_exp[:, :, None] * (-beta) * unit, axis=1)  # (N,2)
    
    total_dexp_dx = wall_contrib + pair_contrib  # (N,2)
    
    grad_E = -(1.0 / beta) * (1.0 / total_exp) * total_dexp_dx
    return E, grad_E


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    rng = np.random.default_rng(42)
    
    # Initialize on a hexagonal-like grid for better starting point
    m = int(np.ceil(np.sqrt(N)))
    # Start with square grid
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    x = grid[:N].copy().astype(np.float64)
    
    # Add small random perturbation
    x += rng.normal(0, 0.001, x.shape)
    # Clip to valid region
    x = np.clip(x, -0.499, 0.499)
    
    best_x = x.copy()
    best_r = _score(best_x)
    
    # Stochastic gradient descent with adaptive learning rate
    lr = 0.005
    beta = 30.0
    momentum = 0.9
    v = np.zeros_like(x)
    
    step = 0
    while time.time() < deadline:
        # Adaptive beta: increase as we converge
        if step % 500 == 0:
            beta = min(100.0, 30.0 * (1 + step / 2000.0))
        
        E, grad = _grad_energy(x, beta)
        
        # SGD update with momentum
        v = momentum * v - lr * grad
        x_new = x + v
        
        # Project back to valid region
        x_new = np.clip(x_new, -0.499, 0.499)
        
        # Check improvement
        r_new = _score(x_new)
        if r_new > best_r:
            best_r = r_new
            best_x = x_new.copy()
        
        # Adaptive learning rate
        if r_new < best_r * 0.99:
            lr *= 0.9
        elif r_new >= best_r:
            lr = min(lr * 1.01, 0.02)
        
        x = x_new
        step += 1
        
        # Occasionally do a simulated-annealing-style kick
        if step % 1000 == 0 and time.time() < deadline - 20:
            kick = rng.normal(0, 0.005, x.shape)
            x_kicked = np.clip(x + kick, -0.499, 0.499)
            r_kicked = _score(x_kicked)
            if r_kicked > best_r:
                best_r = r_kicked
                best_x = x_kicked.copy()
            elif rng.random() < 0.1:
                x = x_kicked
    
    # Final polish: local refinement on best_x
    x = best_x.copy()
    lr2 = 0.001
    beta2 = 80.0
    v2 = np.zeros_like(x)
    for _ in range(500):
        if time.time() > deadline - 5:
            break
        E, grad = _grad_energy(x, beta2)
        v2 = 0.8 * v2 - lr2 * grad
        x = np.clip(x + v2, -0.499, 0.499)
        r = _score(x)
        if r > best_r:
            best_r = r
            best_x = x.copy()
        lr2 *= 0.999
    
    return best_x
# EVOLVE-BLOCK-END
