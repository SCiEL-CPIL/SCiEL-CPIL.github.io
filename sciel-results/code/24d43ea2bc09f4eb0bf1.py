"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.spatial import cKDTree

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_centers = None
    best_score = 0.0
    random.seed(42)
    np.random.seed(42)
    
    # Initial guess: square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Parameters
    lr = 0.02
    momentum = 0.5
    v = np.zeros_like(centers)
    eps = 1e-6
    
    # Main optimization loop
    while time.time() < deadline:
        # Compute forces
        tree = cKDTree(centers)
        pairs = tree.query_pairs(2 * lr + 0.01)
        
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        for i, j in pairs:
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist < eps:
                diff = np.random.randn(2) * eps
                dist = np.linalg.norm(diff)
            # Force proportional to overlap, directed to separate
            overlap = 2 * lr - dist
            if overlap > 0:
                force = diff / dist * overlap
                forces[i] += force
                forces[j] -= force
        
        # Boundary forces (keep inside [-0.5, 0.5]^2)
        for i in range(N):
            for k in range(2):
                c = centers[i, k]
                # If too close to boundary, push inward
                if c < -0.5 + lr:
                    forces[i, k] += (-0.5 + lr - c) * 10
                elif c > 0.5 - lr:
                    forces[i, k] += (0.5 - lr - c) * 10
        
        # Update with momentum
        v = momentum * v + forces
        centers = centers + lr * 0.1 * v
        
        # Project back into square
        centers = np.clip(centers, -0.5 + lr, 0.5 - lr)
        
        # Score
        score = _score(centers)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
        
        # Occasionally try larger steps
        if random.random() < 0.01:
            centers = best_centers + np.random.randn(N, 2) * 0.05
            centers = np.clip(centers, -0.5 + lr, 0.5 - lr)
    
    return best_centers


def _score(centers):
    """Compute the score: min of pairwise distances/2 and boundary distances."""
    tree = cKDTree(centers)
    min_pair_dist = tree.query_pairs(1e-10)
    
    # Find minimum pairwise distance
    min_d = float('inf')
    for i, j in min_pair_dist:
        d = np.linalg.norm(centers[i] - centers[j])
        if d < min_d:
            min_d = d
    
    # Boundary distance
    min_boundary = 0.5 - np.max(np.abs(centers))
    
    return min(min_d / 2, min_boundary)
# EVOLVE-BLOCK-END
