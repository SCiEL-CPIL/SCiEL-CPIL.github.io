import numpy as np

def heuristics(distance_matrix):
    # Use a squared inverse distance to effectively increase the heuristic weight (beta)
    # This aligns with the observation that beta=2 performs well.
    return 1.0 / (distance_matrix + 1e-9)**2
