import numpy as np

def heuristics(distance_matrix):
    # Ensure float type
    d = np.asarray(distance_matrix, dtype=float)
    # Compute inverse squared distance
    # Add epsilon to avoid division by zero on diagonal
    # The diagonal value will be large, but it's never used in ACO selection (i != j)
    # Off-diagonal distances are > 0, so 1/d^2 is well-defined and positive.
    return 1.0 / (d**2 + 1e-9)
