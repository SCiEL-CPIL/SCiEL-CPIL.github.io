"""Symbolic regression with a rich, smooth basis and QR-orthogonalized features."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_basis(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Build a rich set of smooth basis functions of (x, y).
    x, y: 1-D arrays of shape (n,).
    Returns: array of shape (n, m) with m basis columns.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size

    cols = []

    # 1) Polynomial terms up to degree 5 (all monomials x^a y^b with a+b <= 5)
    for a in range(6):
        for b in range(6 - a):
            cols.append((x ** a) * (y ** b))

    # 2) Radial and angular features
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    cols.append(r)
    cols.append(r * r)
    cols.append(r ** 3)
    cols.append(np.cos(theta))
    cols.append(np.sin(theta))
    cols.append(np.cos(2 * theta))
    cols.append(np.sin(2 * theta))
    cols.append(r * np.cos(theta))
    cols.append(r * np.sin(theta))
    cols.append(r * r * np.cos(theta))
    cols.append(r * r * np.sin(theta))

    # 3) Smooth non-polynomial features
    cols.append(np.sin(np.pi * x))
    cols.append(np.cos(np.pi * x))
    cols.append(np.sin(np.pi * y))
    cols.append(np.cos(np.pi * y))
    cols.append(np.sin(np.pi * (x + y)))
    cols.append(np.cos(np.pi * (x + y)))
    cols.append(np.sin(np.pi * (x - y)))
    cols.append(np.cos(np.pi * (x - y)))
    cols.append(np.tanh(x))
    cols.append(np.tanh(y))
    cols.append(np.tanh(x * y))
    cols.append(np.exp(-r * r))
    cols.append(np.exp(-r))
    cols.append(np.log1p(r))

    # 4) Interaction terms with higher-frequency smooth components
    cols.append(np.sin(2 * np.pi * x) * np.cos(np.pi * y))
    cols.append(np.cos(2 * np.pi * x) * np.sin(np.pi * y))
    cols.append(np.sin(np.pi * x) * np.sin(np.pi * y))
    cols.append(np.cos(np.pi * x) * np.cos(np.pi * y))

    B = np.column_stack(cols)

    # Replace any non-finite values (shouldn't happen, but be safe)
    B = np.where(np.isfinite(B), B, 0.0)

    return B


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form predictor. The evaluator fits `params` (length 10) via BFGS
    starting from all ones, minimizing training MSE.

    We build a rich basis, orthogonalize it with QR (keeping only the first
    10 independent directions), and return the linear combination of those
    orthogonal features weighted by the 10 params.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()

    # Ensure params has at least 10 entries; pad with zeros if needed
    if params.size < 10:
        params = np.concatenate([params, np.zeros(10 - params.size)])
    params = params[:10]

    # Build the full basis
    B = _build_basis(x, y)  # shape (n, m)

    # QR decomposition to orthogonalize and pick the first 10 independent directions
    # Use economy QR: Q has shape (n, min(n, m)), R has shape (min(n, m), m)
    # We want the first 10 columns of Q (or fewer if rank-deficient)
    try:
        Q, R = np.linalg.qr(B, mode='reduced')
        # Q shape: (n, k) where k = min(n, m)
        k = Q.shape[1]
        use_k = min(10, k)
        Q_use = Q[:, :use_k]
    except np.linalg.LinAlgError:
        # Fallback: just use the first 10 columns of B
        use_k = min(10, B.shape[1])
        Q_use = B[:, :use_k]

    # If we have fewer than 10 directions, pad params with zeros
    if use_k < 10:
        w = np.zeros(10)
        w[:use_k] = params[:use_k]
    else:
        w = params

    # Prediction: Q_use @ w
    pred = Q_use @ w

    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)

    return pred
# EVOLVE-BLOCK-END
