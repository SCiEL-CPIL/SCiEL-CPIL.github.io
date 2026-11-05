import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 150s kill
    
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    # Calculate degrees
    degrees = A.sum(axis=1)
    
    # Calculate 2 * L(v) where L(v) = number of edges in N(v)
    # B = A^2, B[v,u] = number of common neighbors of v and u
    # two_L[v] = sum_{u in N(v)} B[v,u] = 2 * L(v)
    B = A @ A
    two_L = np.sum(A * B, axis=1)
    L = two_L / 2.0
    
    # Calculate local edge density
    # For d(v) >= 2: rho(v) = 2 * L(v) / (d(v) * (d(v) - 1))
    # For d(v) < 2: rho(v) = 0
    denom = degrees * (degrees - 1.0)
    denom[denom < 1e-10] = 1.0  # Avoid division by zero
    rho = np.where(degrees >= 2, two_L / denom, 0.0)
    
    # Score: local density minus normalized degree
    # We want high density and low degree
    scores = rho - degrees / n
    
    # Safety check
    if time.time() - start_time > deadline:
        return scores
    
    return scores
