"""Adaptive RBF kernel regression: parameters encode centers and scales."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() > _DEADLINE:
        return np.zeros_like(x, dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # We have 10 parameters. Use them to define 5 RBF centers in 3D
    # and a shared width (or per-center widths encoded in the last few params).
    #
    # Design:
    #   params[0:3]  -> center 1 (cx1, cy1, cz1)
    #   params[3:6]  -> center 2 (cx2, cy2, cz2)
    #   params[6:9]  -> center 3 (cx3, cy3, cz3)
    #   params[9]    -> inverse bandwidth (beta), clamped to positive
    #
    # This gives 5 Gaussian RBFs in 3D space. The evaluator's BFGS will
    # optimize these 10 parameters to minimize training MSE.
    #
    # To improve expressiveness, we also add a linear trend term with
    # fixed (non-learned) coefficients derived from the data mean,
    # but that would require more params. Instead, we use a richer
    # parameterization:
    #
    # Alternative: 10 params = 5 centers (2D projected) + 5 widths?
    # That's too many for 3D. Let's stick with 3 centers + 1 beta + 
    # use the remaining 4 params as a linear offset vector.
    #
    # Final design (10 params):
    #   p[0:3]  = center 1
    #   p[3:6]  = center 2  
    #   p[6:9]  = center 3
    #   p[9]    = beta (inverse bandwidth), use softplus to ensure > 0
    #
    # Additionally, we include a constant offset implicitly via the
    # RBF values (sum of Gaussians can produce a DC component).
    #
    # To boost expressiveness, we also add a 4th and 5th RBF whose
    # centers are fixed at strategic points in the domain, and we
    # use p[9] as a global scale. Actually, let me reconsider.
    #
    # Better design for 10 params:
    #   Use 4 RBF centers in 3D = 12 params. Too many.
    #   Use 3 RBF centers in 3D = 9 params + 1 bandwidth = 10. 
    #   This is clean.
    #
    # To make the landscape smoother for BFGS starting from all-ones:
    #   - Map raw params to domain [0.5, 2.0] via sigmoid-like scaling
    #   - Use softplus for bandwidth

    # Map centers to [0.5, 2.0] using a smooth squashing
    # tanh maps R -> [-1,1], so 1.25 + 0.75*tanh(p) maps R -> [0.5, 2.0]
    c1 = 1.25 + 0.75 * np.tanh(p[0:3])
    c2 = 1.25 + 0.75 * np.tanh(p[3:6])
    c3 = 1.25 + 0.75 * np.tanh(p[6:9])

    # Bandwidth: softplus ensures positive, scale appropriately
    beta_raw = p[9]
    # We want beta to control width. Larger beta = narrower peaks.
    # Typical good range for 3D RBF on [0.5,2]^3: beta in [5, 50]
    # Use: beta = 5 + 20 * sigmoid(beta_raw)  -> range [5, 25]
    sigmoid_val = 1.0 / (1.0 + np.exp(-np.clip(beta_raw, -50, 50)))
    beta = 5.0 + 20.0 * sigmoid_val

    # Compute RBF values
    # Distance squared from each point to each center
    def rbf(X, center, beta):
        dx = X[0] - center[0]
        dy = X[1] - center[1]
        dz = X[2] - center[2]
        dist2 = dx * dx + dy * dy + dz * dz
        return np.exp(-beta * dist2)

    X = (x, y, z)
    phi1 = rbf(X, c1, beta)
    phi2 = rbf(X, c2, beta)
    phi3 = rbf(X, c3, beta)

    # We need weights for the RBFs. But we've used all 10 params for
    # centers + beta. We need to include weights too.
    #
    # Revised design:
    #   p[0:3]  = center 1 (mapped to domain)
    #   p[3:6]  = center 2
    #   p[6:9]  = center 3
    #   p[9]    = shared weight scale (applied to all three RBFs)
    #
    # This means all three RBFs have the same weight = p[9] (mapped).
    # That's limiting. Better:
    #
    # Use 2 RBF centers + 2 weights + 1 beta + 1 offset = 2*3 + 2 + 1 + 1 = 10
    #   p[0:3]  = center 1
    #   p[3:6]  = center 2
    #   p[6]    = weight 1
    #   p[7]    = weight 2
    #   p[8]    = beta control
    #   p[9]    = constant offset
    #
    # This is more balanced.

    # Let me redo with this design:
    c1 = 1.25 + 0.75 * np.tanh(p[0:3])
    c2 = 1.25 + 0.75 * np.tanh(p[3:6])
    w1 = p[6]
    w2 = p[7]

    # Beta: map to [2, 30]
    sigmoid_val = 1.0 / (1.0 + np.exp(-np.clip(p[8], -50, 50)))
    beta = 2.0 + 28.0 * sigmoid_val

    offset = p[9]

    phi1 = rbf(X, c1, beta)
    phi2 = rbf(X, c2, beta)

    pred = w1 * phi1 + w2 * phi2 + offset

    # Add a mild linear component to help BFGS converge from all-ones
    # (at all-ones, centers are at 1.25+0.75*tanh(1)≈1.74, which is off-center)
    # A small linear term stabilizes the landscape.
    # Use fixed small coefficients to not consume params:
    pred += 0.1 * (x - 1.25) + 0.1 * (y - 1.25) + 0.1 * (z - 1.25)

    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)

    return pred
# EVOLVE-BLOCK-END
