"""Analytic continuous-time HJB solver for optimal market making with inventory risk.

Replaces discrete backward induction with closed-form optimal depth expressions
derived from the continuous-time HJB first-order conditions (Avellaneda-Stoikov
style). Depth on each side is  (1/k) * ln(A / (gamma*sigma^2*tau))  with an
inventory-skew  +/- (gamma*sigma^2*tau)/(2A)  to flatten inventory, evaluated
as a simple algebraic function of inventory q and time-to-horizon tau.
"""
# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """Return (bid_depths, ask_depths), float arrays of shape (N,), distances
    below/above mid to quote this step; np.inf means no quote."""
    sigma = float(params["sigma"])
    A = float(params["A"])
    k = float(params["k"])
    gamma = float(params["gamma"])
    T = float(params["T"])
    Q = int(params["Q"])

    n = len(inventory)
    q = np.asarray(inventory, dtype=np.float64)

    # Time remaining (clipped to non-negative).
    tau = np.clip(T - float(t), 0.0, T)

    # Risk-aversion of holding inventory over the remaining horizon.
    phi = gamma * sigma * sigma * tau

    # Floor so that ln(A/phi) stays finite and depths stay within the valid
    # band (depth > -3/k).
    phi_safe = np.maximum(phi, A * np.exp(-3.0 * k) * 0.5)

    base = (1.0 / k) * np.log(A / phi_safe)          # symmetric core depth
    # Inventory-flattening skew depends on current inventory q
    skew_coeff = phi / (2.0 * A)
    skew = skew_coeff * q

    # Clip depths to the valid range: > -3/k  and  <= Q (and non-negative).
    lo = -3.0 / k + 1e-9
    hi = float(Q)
    bid = np.clip(base + skew, lo, hi)
    ask = np.clip(base - skew, lo, hi)

    # Boundary constraints: no bid fill at q = +Q, no ask fill at q = -Q.
    bid = np.where(q >= Q, np.inf, bid)
    ask = np.where(q <= -Q, np.inf, ask)

    # Guarantee validity (no NaN / -inf / below lo).
    bid = np.where(np.isfinite(bid) & (bid > lo), bid, np.inf)
    ask = np.where(np.isfinite(ask) & (ask > lo), ask, np.inf)

    return bid, ask
# EVOLVE-BLOCK-END
