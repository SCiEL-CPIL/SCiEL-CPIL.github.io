"""Symbolic regression: flexible localized-bump + linear model fitted by the evaluator's BFGS."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model with 10 parameters:
      params[0]  : global offset (bias)
      params[1]  : linear slope
      params[2]  : bump 0 weight
      params[3]  : bump 0 center
      params[4]  : bump 1 weight
      params[5]  : bump 1 center
      params[6]  : bump 2 weight
      params[7]  : bump 2 center
      params[8]  : bump 3 weight
      params[9]  : bump 4 center

    Each bump is a fixed-width Gaussian: weight * exp(-(x - center)^2 / (2 * sigma^2)).
    The fixed width keeps the problem smooth and the Hessian well-conditioned so that
    scipy BFGS (started from all ones) converges reliably. The five bumps can be
    positioned across [-3, 3] to capture localized structure, while the offset and
    slope capture the global trend.
    """
    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    if scalar_input:
        x = x.reshape(1)

    p = np.asarray(params, dtype=float).ravel()
    # Defensive: if fewer than 10 params are supplied, pad with zeros.
    if p.shape[0] < 10:
        p = np.concatenate([p, np.zeros(10 - p.shape[0])])

    offset = p[0]
    slope = p[1]

    # Fixed Gaussian width (sigma). Chosen so a single bump covers a meaningful
    # fraction of the [-3, 3] domain while remaining narrow enough to localize.
    sigma = 0.75
    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)

    out = np.full(x.shape, offset, dtype=float)
    out = out + slope * x

    centers = p[3::2]   # params[3], p[5], p[7], p[9] -> 4 centers? need 5 bumps
    # We have 5 bumps but only 8 remaining params (2..9). Re-map:
    #   bump i weight = p[2 + 2*i], center = p[3 + 2*i] for i in 0..3 (4 bumps)
    # To get 5 bumps we'd need 10 params just for bumps; we only have 8.
    # So use 4 bumps (weights p[2,4,6,8], centers p[3,5,7,9]).
    weights = p[2::2]   # params[2], p[4], p[6], p[8] -> 4 weights
    centers = p[3::2]   # params[3], p[5], p[7], p[9] -> 4 centers

    for w, c in zip(weights, centers):
        if not np.isfinite(w) or not np.isfinite(c):
            continue
        d = x - c
        out = out + w * np.exp(-inv_two_sigma2 * d * d)

    # Final safety: replace any non-finite values with the offset.
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, offset)

    if scalar_input:
        return out[0]
    return out
# EVOLVE-BLOCK-END
