import numpy as np

# Consolidation bias: small positive weight on current bin fill to prefer
# packing into already-used bins (reduces total bin count vs. L1 bound).
_BETA = 0.05
_INFEASIBLE = -1e9


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Score each bin with enough remaining capacity; higher = preferred.

    Primary term: tightest fit (minimize remaining space after placement).
    Secondary term: consolidation bias toward already-used bins.
    """
    bins = np.asarray(bins, dtype=float)
    remaining = bins - item
    feasible = remaining >= 0.0

    # Tightest fit: prefer smallest remaining (negative remaining).
    score = -remaining

    # Consolidation bias: prefer bins that already hold more.
    score = score + _BETA * bins

    # Mask infeasible bins so they are never selected.
    score = np.where(feasible, score, _INFEASIBLE)

    return score
