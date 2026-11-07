import numpy as np

def heuristics(prizes, weights):
    # prizes and weights are (5, 100)
    # P_i is sum of prizes for item i
    P = np.sum(prizes, axis=0)
    
    # max weight per item across constraints
    # weights is (5, 100), so max over axis 0
    max_w = np.max(weights, axis=0)
    
    # Heuristic: Prize / Max Weight
    # Add epsilon to avoid division by zero
    eta = P / (max_w + 1e-9)
    
    # Ensure positive
    eta = np.maximum(eta, 1e-9)
    
    return eta
