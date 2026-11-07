import numpy as np

def heuristics(distance_matrix):
    eps = 1e-9
    # Use a sharper heuristic: (d + eps)^-2.5
    # This further emphasizes short edges over long ones compared to the 
    # current (d + eps)^-2.0, potentially helping the 20 ants converge 
    # to better tours within the 30 iterations.
    return (distance_matrix + eps) ** -2.5
