"""Optimal market making with inventory risk - direct derivation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 285  # Safety margin before 330s kill


def quote(t, inventory, mid, params):
    """
    Derive bid/ask depths from first principles using a smooth log-formulation:
    - Equate marginal fill profit to inventory risk cost
    - Use d = (1/k) * log(1 + c * |q| * time_factor) for smoothness near q=0
    - When q > 0 (long): widen bid (reduce buy fills), tighten ask (more sell fills)
    - When q < 0 (short): tighten bid, widen ask
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    steps = params['steps']
    T = params['T']

    dt = T / steps
    q = inventory.astype(np.float64)

    # Time decay: full risk adjustment at t=0, fades to 0 at t=T
    remaining = T - t
    time_factor = np.sqrt(np.maximum(remaining / T, 0.0))

    # Risk parameter
    risk_param = gamma * sigma * sigma * dt

    # Base depth: 1/k
    d_base = 1.0 / k

    # Calibration: choose c so that at |q| = Q, the depth adjustment is reasonable
    # We want the adjustment at q=Q to be approximately 0.5/k
    # d = (1/k) * log(1 + c * Q * 1) = 0.5/k => log(1 + c*Q) = 0.5 => c*Q = exp(0.5)-1
    c_calib = (np.exp(0.5) - 1.0) / Q if Q > 0 else 0.0

    # Smooth risk scaling via softplus-like transform (avoids hard clip discontinuity)
    # Maps risk_ratio in (0, ~3) to a smooth multiplier in (0.5, 2.0)
    risk_ratio = (gamma * sigma * sigma) / (0.05 * 4.0)
    # softplus(x) = log(1 + exp(x)); use scaled version for smooth saturation
    risk_scale = 0.5 + 1.5 * (1.0 / (1.0 + np.exp(-2.0 * (risk_ratio - 1.0))))
    risk_scale = float(np.clip(risk_scale, 0.5, 2.0))

    # Effective inventory with time decay
    q_eff = np.abs(q) * time_factor

    # Smooth depth adjustment using log formulation
    # This avoids the singularity at q=0 and provides smooth gradients
    c_eff = c_calib * risk_scale
    adjustment = (1.0 / k) * np.log1p(c_eff * q_eff)

    # Bid and ask adjustments with sign
    # When long (q>0): bid_adj > 0 (widen bid), ask_adj < 0 (tighten ask)
    # When short (q<0): bid_adj < 0 (tighten bid), ask_adj > 0 (widen ask)
    sign_q = np.sign(q)
    bid_adj = sign_q * adjustment
    ask_adj = -sign_q * adjustment

    bid_depth = d_base + bid_adj
    ask_depth = d_base + ask_adj

    # Clamp to valid range: > -3/k, > 0
    min_depth = np.maximum(0.01, -3.0/k + 0.01)
    bid_depth = np.maximum(bid_depth, min_depth)
    ask_depth = np.maximum(ask_depth, min_depth)

    # Soft boundary: heavily penalize quoting on the side that would push inventory
    # further beyond Q, but use a very large finite value instead of inf to keep
    # the optimization landscape smooth.  Fill prob at d=50/k is ~1-exp(-A*exp(-50)*dt)
    # which is negligible for all given regimes.
    big_depth = 50.0 / k
    bid_over = (inventory >= Q).astype(np.float64)
    ask_over = (inventory <= -Q).astype(np.float64)
    bid_depth = np.where(bid_over > 0.5, big_depth, bid_depth)
    ask_depth = np.where(ask_over > 0.5, big_depth, ask_depth)

    # Final safety: ensure shape and no NaN/inf in wrong places
    bid_depth = np.where(np.isfinite(bid_depth), bid_depth, big_depth)
    ask_depth = np.where(np.isfinite(ask_depth), ask_depth, big_depth)

    return bid_depth, ask_depth
# EVOLVE-BLOCK-END
