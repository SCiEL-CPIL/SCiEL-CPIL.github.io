"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed network weights for the feature extractor
# Architecture: input(2) -> hidden1(8, tanh) -> hidden2(10, tanh) -> 10 features
# Weights are fixed; only the final linear combination (params) is optimized.
_rng = np.random.RandomState(42)

# Hidden layer 1: 2 -> 8
_W1 = _rng.uniform(-0.5, 0.5, size=(8, 2))
_b1 = _rng.uniform(-0.1, 0.1, size=8)

# Hidden layer 2: 8 -> 10
_W2 = _rng.uniform(-0.5, 0.5, size=(10, 8))
_b2 = _rng.uniform(-0.1, 0.1, size=10)


def _compute_features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute fixed non-linear features from (x, y). Returns shape (N, 10)."""
    # Input: (N, 2)
    inp = np.stack([x, y], axis=-1)
    
    # Hidden layer 1: (N, 2) -> (N, 8)
    h1 = np.tanh(inp @ _W1.T + _b1)
    
    # Hidden layer 2: (N, 8) -> (N, 10)
    h2 = np.tanh(h1 @ _W2.T + _b2)
    
    return h2


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Neural network with fixed hidden layers and optimizable output weights.
    params (length 10) are the output layer weights.
    The model is linear in params, so BFGS can optimize it efficiently.
    """
    features = _compute_features(x, y)  # shape (N, 10)
    return features @ params
# EVOLVE-BLOCK-END
