"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    # Track time to ensure we finish well within limits
    start_time = time.time()
    
    # Convert to numpy array for efficient operations
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Calculate degrees of all vertices
    degrees = A.sum(axis=1)
    
    # Heuristic:
    # We want to prioritize vertices that are "cheap" to select.
    # A vertex is cheap if it has a low degree (blocks few others)
    # and if its neighbors are "expensive" (high degree), meaning we are
    # not "wasting" the opportunity to select those neighbors later.
    # Conversely, if a vertex is connected to many low-degree vertices,
    # selecting it removes many "easy" picks, which is suboptimal.
    
    # Weight of a neighbor j is inversely proportional to its degree.
    # Low degree neighbors get high weight (penalty for blocking them).
    # High degree neighbors get low weight (less penalty).
    # Using d + 1 to handle isolated nodes safely.
    neighbor_weights = 1.0 / (degrees + 1.0)
    
    # For each vertex i, calculate the sum of weights of its neighbors.
    # This is a matrix-vector product: A * neighbor_weights
    neighbor_penalty = A.dot(neighbor_weights)
    
    # Total cost to select vertex i:
    # Its own degree (direct cost of blocking)
    # Plus the penalty from its neighbors (indirect cost of blocking "good" candidates)
    total_cost = degrees + neighbor_penalty
    
    # The decoder selects in descending order of priority.
    # We want to select low cost vertices first.
    # So we return negative cost.
    scores = -total_cost
    
    # Safety check: if this took too long (unlikely for N=200), we could
    # potentially fall back to a simpler heuristic, but N=200 is very small.
    # The operation is O(N^2) which is ~40k flops, negligible.
    
    return scores
# EVOLVE-BLOCK-END
