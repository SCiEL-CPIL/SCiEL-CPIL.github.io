import numpy as np

def heuristics(distance_matrix):
    # Use inverse distance (beta=1) to balance greediness and exploration.
    # Beta=2 (current) might be too greedy, causing premature convergence to local optima
    # with crossing edges. Beta=1 allows the colony to consider slightly longer edges
    # that may lead to a better global tour.
    # Epsilon is increased to smooth out extremely close points.
    epsilon = 1e-2
    return 1.0 / (distance_matrix + epsilon)
