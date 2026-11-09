# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Hybrid basis: combines a normalized quadratic polynomial basis (for global
    # smooth structure) with a single adaptive RBF term (for local non-polynomial
    # features). The model is linear in params, so BFGS converges reliably.
    #
    # Normalization: map inputs from [0.5, 2] to [0, 1] for numerical stability.
    #   xn = (x - 0.5) / 1.5
    #   yn = (y - 0.5) / 1.5
    #   zn = (z - 0.5) / 1.5
    #
    # Basis functions (10 total):
    #   0: 1 (bias)
    #   1: xn
    #   2: yn
    #   3: zn
    #   4: xn^2
    #   5: yn^2
    #   6: zn^2
    #   7: xn*yn
    #   8: xn*zn
    #   9: yn*zn
    #
    # Wait, that's only 10 terms and leaves no room for RBF. Let me restructure.
    # I'll use 9 polynomial terms + 1 RBF term.
    #
    # Revised basis (10 total):
    #   0: 1 (bias)
    #   1: xn
    #   2: yn
    #   3: zn
    #   4: xn^2
    #   5: yn^2
    #   6: zn^2
    #   7: xn*yn
    #   8: xn*zn
    #   9: yn*zn
    #
    # Actually, let me think about this more carefully. The inspiration 2 (RBF with 9 centers)
    # scored 3.2322, and inspiration 3 (hybrid polynomial + 2 RBFs) scored 3.2172.
    # The current program scores 14.5194. Inspiration 1 (full quadratic) scores 7.3930.
    #
    # The best approach seems to be a combination of global polynomial structure with
    # local RBF features. Let me use 7 polynomial terms + 3 RBF terms.
    #
    # Basis functions (10 total):
    #   0: 1 (bias)
    #   1: xn
    #   2: yn
    #   3: zn
    #   4: xn^2
    #   5: yn^2
    #   6: zn^2
    #   7: RBF at (0, 0, 0) in normalized coords [i.e., (0.5, 0.5, 0.5) in original]
    #   8: RBF at (1, 1, 1) in normalized coords [i.e., (2.0, 2.0, 2.0) in original]
    #   9: RBF at (0.5, 0.5, 0.5) in normalized coords [i.e., (1.25, 1.25, 1.25) in original]
    #
    # This gives global quadratic structure (without all cross terms to make room for RBFs)
    # plus local flexibility at key points.
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    n = len(x)
    
    # Guard against non-finite inputs
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(z))):
        return np.zeros(n, dtype=np.float64)
    
    # Normalize inputs to [0, 1]
    xn = (x - 0.5) / 1.5
    yn = (y - 0.5) / 1.5
    zn = (z - 0.5) / 1.5
    
    # Clip to avoid any floating point edge cases
    xn = np.clip(xn, 0.0, 1.0)
    yn = np.clip(yn, 0.0, 1.0)
    zn = np.clip(zn, 0.0, 1.0)
    
    # Build feature matrix: shape (n, 10)
    F = np.empty((n, 10), dtype=np.float64)
    F[:, 0] = 1.0
    F[:, 1] = xn
    F[:, 2] = yn
    F[:, 3] = zn
    F[:, 4] = xn * xn
    F[:, 5] = yn * yn
    F[:, 6] = zn * zn
    
    # RBF features with fixed gamma for stability
    gamma = 2.0
    
    # RBF centered at (0, 0, 0) in normalized space
    dist_sq1 = xn*xn + yn*yn + zn*zn
    F[:, 7] = np.exp(-gamma * dist_sq1)
    
    # RBF centered at (1, 1, 1) in normalized space
    dx2 = xn - 1.0
    dy2 = yn - 1.0
    dz2 = zn - 1.0
    dist_sq2 = dx2*dx2 + dy2*dy2 + dz2*dz2
    F[:, 8] = np.exp(-gamma * dist_sq2)
    
    # RBF centered at (0.5, 0.5, 0.5) in normalized space
    dx3 = xn - 0.5
    dy3 = yn - 0.5
    dz3 = zn - 0.5
    dist_sq3 = dx3*dx3 + dy3*dy3 + dz3*dz3
    F[:, 9] = np.exp(-gamma * dist_sq3)
    
    # Compute prediction
    pred = F @ p
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return pred
# EVOLVE-BLOCK-END
