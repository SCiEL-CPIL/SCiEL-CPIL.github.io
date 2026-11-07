"""Deterministic Fixed-Geometry RBF Portfolio: 
Strategy: 
1. Pre-compute 10 Radial Basis Functions (RBFs) with centers and widths fixed by a Halton sequence over [-1, 1].
2. Return a prediction that is a linear combination of these 10 RBFs, weighted by the 10 `params`.
3. This decouples the structural complexity (fixed geometry) from the fitting process (BFGS fits amplitudes only).
4. This avoids the instability of optimizing geometric parameters and the overhead of internal CV/ridge searches.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Wall-clock deadline guard with a safety margin (evaluator kills at 150 s).
_T0 = time.time()
_DEADLINE = 115.0  # seconds of budget to spend inside equation()


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _halton_sequence(n, base):
    """Generate the first n values of the Halton sequence in the given base."""
    seq = []
    for i in range(1, n + 1):
        f = 1.0
        result = 0.0
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        seq.append(result)
    return np.array(seq)


def _generate_rbf_centers_and_widths(n_centers=10):
    """
    Generate deterministic centers and widths for RBFs using Halton sequences.
    Centers are in [-1, 1] x [-1, 1].
    Widths are chosen to cover the domain reasonably.
    """
    # Use Halton bases 2 and 3 for 2D coordinates
    h2 = _halton_sequence(n_centers, 2)
    h3 = _halton_sequence(n_centers, 3)
    
    # Map Halton [0, 1] to [-1, 1]
    centers_x = 2.0 * h2 - 1.0
    centers_y = 2.0 * h3 - 1.0
    
    # Widths: vary them to capture different scales.
    # A mix of narrow and wide RBFs.
    # Let's use a Halton sequence for width scaling too, mapped to a reasonable range.
    # Widths sigma in [0.1, 0.5] roughly.
    h5 = _halton_sequence(n_centers, 5)
    # Map to [0.1, 0.6]
    widths = 0.1 + 0.5 * h5
    
    centers = np.column_stack([centers_x, centers_y])
    return centers, widths


# Pre-compute the RBF geometry once.
_CENTERS, _WIDTHS = _generate_rbf_centers_and_widths(10)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns a linear combination of 10 fixed RBFs.
    The external BFGS optimizer will fit `params` (amplitudes).
    """
    X = np.column_stack([np.asarray(x, float).ravel(),
                         np.asarray(y, float).ravel()])
    n = X.shape[0]
    
    # Ensure params is a 1D array of length 10
    p = np.asarray(params, float).ravel()
    if len(p) < 10:
        # Pad with zeros if necessary, though contract says length 10
        p = np.pad(p, (0, 10 - len(p)))
    else:
        p = p[:10]
        
    # Compute RBF values: exp(-||X - c_i||^2 / (2 * sigma_i^2))
    # X: (n, 2), Centers: (10, 2)
    # Distances squared: (n, 10)
    diff = X[:, np.newaxis, :] - _CENTERS[np.newaxis, :, :]  # (n, 10, 2)
    dist_sq = np.sum(diff ** 2, axis=2)  # (n, 10)
    
    # RBF matrix: (n, 10)
    rbf_vals = np.exp(-dist_sq / (2.0 * _WIDTHS[np.newaxis, :] ** 2))
    
    # Prediction: (n, 10) @ (10,) -> (n,)
    pred = rbf_vals @ p
    
    # Safety check
    if not np.all(np.isfinite(pred)):
        # Fallback to a simple linear model if something goes wrong
        # But since RBFs are always finite, this is just a precaution.
        return np.zeros(n)
        
    return pred
# EVOLVE-BLOCK-END
