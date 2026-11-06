"""Rewritten program: shared odd-degree polynomial closure exploiting antisymmetry."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared odd-degree (1,3,5) polynomial to the unresolved subgrid tendency.
    
    Exploits the physical antisymmetry of the unresolved forcing with respect to
    the slow state X. Uses Tikhon (L2) regularization with scaled basis.
    
    The unresolved tendency in the two-scale Lorenz-96 model is antisymmetric
    in the slow state, so only odd powers (x, x^3, x^5) are physically meaningful.
    """
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    t = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n < 4:
        # Not enough data; fall back to linear odd
        if n < 2:
            return np.zeros(6)
        try:
            V_small = np.column_stack([x, x**3]) if n >= 3 else x.reshape(-1, 1)
            A_small = V_small.T @ V_small + 1e-4 * np.eye(V_small.shape[1])
            b_small = V_small.T @ t
            c_small = np.linalg.solve(A_small, b_small)
            out = np.zeros(6)
            out[1] = c_small[0]
            if len(c_small) > 1:
                out[3] = c_small[1]
            return out
        except Exception:
            return np.zeros(6)
    
    # Scale x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std
    
    # Build design matrix for odd powers only: x^1, x^3, x^5 in scaled space
    V = np.column_stack([x_scaled, x_scaled**3, x_scaled**5])
    
    # Tikhon regularization for the 3 odd coefficients
    # Heavier regularization on higher degrees to prevent overfitting
    lambda_reg = 1e-4
    reg_weights = np.array([1.0, 10.0, 50.0])
    reg_matrix = lambda_reg * np.diag(reg_weights)
    
    try:
        A = V.T @ V + reg_matrix
        b = V.T @ t
        coeff_scaled = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        try:
            coeff_scaled, _, _, _ = np.linalg.lstsq(V, t, rcond=None)
        except Exception:
            return np.zeros(6)
    
    # Convert from scaled x to original x
    # P(x) = c1 * ((x-m)/s) + c3 * ((x-m)/s)^3 + c5 * ((x-m)/s)^5
    # Expand each term: ((x-m)/s)^j = (1/s^j) * sum_{k=0}^{j} C(j,k) * x^k * (-m)^(j-k)
    
    coeff = np.zeros(6)
    for j_idx, j in enumerate([1, 3, 5]):
        cj = coeff_scaled[j_idx]
        if abs(cj) < 1e-15:
            continue
        scale = (1.0 / x_std) ** j
        m = x_mean
        for k in range(j + 1):
            binom = comb(j, k)
            coeff[k] += cj * scale * binom * ((-m) ** (j - k))
    
    # Enforce odd-only: zero out even coefficients (numerical cleanup)
    coeff[0] = 0.0
    coeff[2] = 0.0
    coeff[4] = 0.0
    
    # Verify finiteness
    if not np.all(np.isfinite(coeff)):
        # Fallback: fit odd-only in original space
        try:
            V_orig = np.column_stack([x, x**3, x**5])
            A_orig = V_orig.T @ V_orig + 1e-4 * np.diag([1.0, 10.0, 50.0])
            b_orig = V_orig.T @ t
            c_orig = np.linalg.solve(A_orig, b_orig)
            coeff = np.zeros(6)
            coeff[1] = c_orig[0]
            coeff[3] = c_orig[1]
            coeff[5] = c_orig[2]
        except Exception:
            coeff = np.zeros(6)
    
    return coeff
# EVOLVE-BLOCK-END
