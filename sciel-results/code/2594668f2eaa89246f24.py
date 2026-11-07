import numpy as np

def heuristics(prizes, weights):
    """
    Computes heuristic values for items in a multiple-constraint knapsack problem.
    
    Args:
        prizes: numpy array of shape (5, 100) with prize values.
        weights: numpy array of shape (5, 100) with weight values.
        
    Returns:
        numpy array of shape (100,) with heuristic values.
    """
    # Ensure inputs are numpy arrays of float
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential shape issues (though task says shape is fixed)
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
        
    # Add epsilon to avoid division by zero or log of zero
    eps = 1e-9
    
    # Per-constraint efficiency: prize / weight
    # Shape: (5, 100)
    efficiencies = prizes / (weights + eps)
    
    # We want a "soft-min" or robust average of these efficiencies.
    # Harmonic mean penalizes bottleneck constraints more heavily than geometric mean,
    # guiding ants away from items that risk violating the tightest constraint early.
    
    # Calculate reciprocal of efficiencies
    reciprocal_efficiencies = 1.0 / efficiencies
    
    # Average reciprocal efficiencies across constraints (axis 0)
    # Shape: (100,)
    avg_reciprocal_efficiency = np.mean(reciprocal_efficiencies, axis=0)
    
    # Invert to get the harmonic mean efficiency
    heuristic = 1.0 / avg_reciprocal_efficiency
    
    # Ensure values are finite and positive
    heuristic = np.where(np.isfinite(heuristic), heuristic, eps)
    heuristic = np.maximum(heuristic, eps)
    
    return heuristic
