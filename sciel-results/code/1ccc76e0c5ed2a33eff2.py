"""Symbolic regression with RBF basis (fixed centers/widths, fit amplitudes)."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time


def _halton2d(n, skip=0):
    """Generate n 2D Halton sequence points in [0,1]^2, skipping the first 'skip' points."""
    def halton(i, base):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    points = []
    i = skip
    while len(points) < n:
        i += 1
        points.append((halton(i, 2), halton(i, 3)))
    return np.array(points)


def _compute_rbf_bases(x, y, n_centers=8, width=0.35):
    """
    Compute RBF basis values for given x, y.
    Returns an array of shape (n_points, n_centers).
    Centers are placed on a Halton grid over [-1, 1]^2.
    """
    # Generate centers using Halton sequence, mapped to [-1, 1]^2
    halton_pts = _halton2d(n_centers, skip=1)  # skip first point (0,0)
    centers = 2.0 * halton_pts - 1.0  # map [0,1] -> [-1,1]
    
    # Compute distances from each point to each center
    points = np.column_stack([x, y])
    dists = cdist(points, centers)  # shape (n_points, n_centers)
    
    # Gaussian RBF: exp(-width^2 * dist^2)
    rbf_vals = np.exp(-(width ** 2) * (dists ** 2))
    
    return rbf_vals


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict using a linear combination of RBFs with fixed centers/widths.
    params[0:8] are the RBF amplitudes, params[8] is a bias term.
    """
    n_centers = 8
    width = 0.35
    
    rbf_vals = _compute_rbf_bases(x, y, n_centers=n_centers, width=width)
    
    # Linear combination: sum(params[i] * rbf_i) + params[8]
    pred = rbf_vals @ params[:n_centers] + params[n_centers]
    
    return pred
