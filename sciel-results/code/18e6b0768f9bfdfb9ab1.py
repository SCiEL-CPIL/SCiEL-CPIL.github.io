"""Rewritten packing program: local energy minimization with boundary-aware soft objective."""
import numpy as np
import time

N = 100


def _score(c):
    """Compute the achievable radius for a set of centres."""
    if not np.all(np.isfinite(c)):
        return -1.0
    if c.shape != (N, 3):
        return -1.0
    # Check all points inside cube
    if np.any(np.abs(c) > 0.5):
        return -1.0
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Distance to boundary
    min_boundary = np.min(0.5 - np.abs(c))
    return min(min_pair, min_boundary)


def _soft_min(x, k=10.0):
    """Soft minimum of an array."""
    return -np.log(np.sum(np.exp(-k * x))) / k


def _energy(c, k=50.0):
    """
    Energy to minimize: encourages large minimum distance.
    Uses soft-min of pairwise distances and boundary distances.
    """
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    # Soft min of pairwise distances
    # Use only lower triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    soft_pair = -np.log(np.sum(np.exp(-k * pair_dists))) / k
    
    # Boundary distances
    bound_dists = 0.5 - np.abs(c)  # shape (N, 3)
    min_bound = np.min(bound_dists, axis=1)  # shape (N,)
    soft_bound = -np.log(np.sum(np.exp(-k * min_bound))) / k
    
    # We want to maximize min(soft_pair, soft_bound)
    # Equivalently minimize -min(...)
    # Use soft-min of the two terms
    # Actually, let's just use a combination: minimize -soft_pair - soft_bound is not right
    # We want to maximize the minimum of all constraints
    # Let's use: energy = -soft_min_of_all_constraints
    # All constraints: pair_dists and min_bound
    all_constraints = np.concatenate([pair_dists, min_bound])
    soft_all = -np.log(np.sum(np.exp(-k * all_constraints))) / k
    return -soft_all


def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # 110s safety margin for 120s limit
    
    # Initial configuration: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = grid[:N].copy()
    
    best_c = c.copy()
    best_score = _score(c)
    
    # Gradient descent with adaptive step size
    k = 30.0  # soft-min parameter
    lr = 0.01
    patience = 0
    max_patience = 200
    
    while time.time() < deadline:
        # Compute gradient numerically
        grad = np.zeros_like(c)
        eps = 1e-5
        for i in range(N):
            for j in range(3):
                c_plus = c.copy()
                c_plus[i, j] += eps
                e_plus = _energy(c_plus, k)
                c_minus = c.copy()
                c_minus[i, j] -= eps
                e_minus = _energy(c_minus, k)
                grad[i, j] = (e_plus - e_minus) / (2 * eps)
        
        # Gradient step
        c_new = c - lr * grad
        
        # Project back to cube
        c_new = np.clip(c_new, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Check score
        new_score = _score(c_new)
        if new_score > best_score:
            best_score = new_score
            best_c = c_new.copy()
            patience = 0
        else:
            patience += 1
        
        c = c_new
        
        # Adaptive learning rate
        if patience > 50:
            lr *= 0.95
            patience = 0
        if lr < 1e-6:
            break
        
        if patience >= max_patience:
            break
    
    # Final polish: try to improve with smaller steps
    if time.time() < deadline - 5:
        lr = 1e-5
        for _ in range(1000):
            if time.time() > deadline - 2:
                break
            grad = np.zeros_like(c)
            eps = 1e-6
            for i in range(N):
                for j in range(3):
                    c_plus = c.copy()
                    c_plus[i, j] += eps
                    e_plus = _energy(c_plus, k)
                    c_minus = c.copy()
                    c_minus[i, j] -= eps
                    e_minus = _energy(c_minus, k)
                    grad[i, j] = (e_plus - e_minus) / (2 * eps)
            
            c_new = c - lr * grad
            c_new = np.clip(c_new, -0.5 + 1e-9, 0.5 - 1e-9)
            
            new_score = _score(c_new)
            if new_score > best_score:
                best_score = new_score
                best_c = c_new.copy()
            c = c_new
    
    return best_c
