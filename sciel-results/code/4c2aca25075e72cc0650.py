import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    # Initialize heuristic matrix
    eta = np.zeros((n, n))
    
    # Use Clarke-Wright Savings Heuristic
    # This measures the distance saved by connecting node i and node j in the same route
    # compared to serving them separately from the depot.
    # s_ij = d_0i + d_0j - d_ij
    # Higher savings indicate a more attractive edge for route construction.
    
    # Extract distances from the depot (node 0)
    d_0 = distance_matrix[0, :]
    
    # Calculate savings for all pairs using vectorization
    # d_0[:, None] has shape (n, 1), d_0[None, :] has shape (1, n)
    savings = d_0[:, None] + d_0[None, :] - distance_matrix
    
    # Ensure non-negative values (handling floating point precision issues)
    savings = np.clip(savings, 0.0, None)
    
    # The task specifies that non-positive heuristic entries become 1e-9.
    # We return the raw savings; the ACO loop will apply the 1e-9 floor.
    
    return savings
