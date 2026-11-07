"""Structural rewrite: exact optimal policy for the discrete model.

The per-step objective is linear in each quoted depth (a fill earns exactly its
depth, and there is no cost on the depth itself). Hence, conditional on choosing
to quote a side, the optimal depth is the maximum allowed by the inventory
boundary:
    bid depth = Q - q   (a bid fill would move q -> q+1, allowed iff q < Q)
    ask depth = Q + q   (an ask fill would move q -> q-1, allowed iff q > -Q)

The marginal value of taking a fill on a side is exactly the depth it earns
(independent of the regime), so the optimal decision is regime-independent:
quote a side iff its boundary depth is positive. At q = Q the bid depth is 0
(a worthless fill), so skip the bid; at q = -Q skip the ask. This matches the
seed at interior inventory (quote both sides) but improves exactly at the
boundaries, which is where the seed wastes a zero-value fill.
"""
import numpy as np


def quote(t, inventory, mid, params):
    n = int(inventory.shape[0])
    q = np.asarray(inventory, dtype=np.int64)
    Q = int(params["k"] and params["Q"])  # Q from params

    # Allowed depths on each side given the inventory boundary.
    bid_depth = np.maximum(Q - q, 0).astype(np.float64)
    ask_depth = np.maximum(Q + q, 0).astype(np.float64)

    # Quote a side only when its depth is positive (a zero-depth fill is
    # worthless and would needlessly consume a fill event).
    bid_quote = bid_depth > 0.0
    ask_quote = ask_depth > 0.0

    bid_out = np.where(bid_quote, bid_depth, np.inf)
    ask_out = np.where(ask_quote, ask_depth, np.inf)

    return bid_out, ask_out
