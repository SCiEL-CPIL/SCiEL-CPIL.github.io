"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    The item goes to the highest-scoring bin.
    
    Strategy: Polarization + Best-Fit hybrid.
    We want to push bins towards either full or empty to reduce fragmentation.
    - For bins that can fit the item, we prefer bins where the remaining space
      after placing the item is very small (fills the bin) OR the remaining space
      is very large (keeps the bin mostly empty for larger future items).
    - However, simply minimizing remaining space is Best-Fit.
    - Polarization suggests we should avoid 'middle' states.
    
    Let r = bins - item (remaining space after placing item).
    We want to maximize a score such that small r (full bin) or large r (empty bin) are good.
    A common approach: Score = -|r - target| where target is 0 or Capacity? 
    But we don't know Capacity directly in the function signature, but bins are capacities?
    No, `bins` array contains the *remaining* capacity of each bin.
    
    Wait, the interface says: `bins: np.ndarray` -> one score per bin.
    Usually in these tasks, `bins` holds the current *remaining* capacity.
    If `bins[i] >= item`, the bin is eligible.
    
    Let's analyze the "Polarization" insight:
    "Polarization scoring encourages bins to become either full or empty... Measured: 1 -> 1.02."
    
    If `bins` is remaining capacity:
    - Full bin after placement: `bins[i] - item` is close to 0.
    - Empty bin (mostly) after placement: `bins[i] - item` is close to `bins[i]` (i.e., item is small relative to bin, or we are keeping the bin "fresh").
    
    Actually, if we place an item in a bin, the new remaining capacity is `r = bins[i] - item`.
    If we want to keep bins "empty", we prefer bins where `r` is large? That means we prefer bins with large initial capacity? That's Worst-Fit.
    If we want to keep bins "full", we prefer bins where `r` is small? That's Best-Fit.
    
    Polarization might mean:
    Prefer Best-Fit if the bin is already somewhat filled?
    Prefer Worst-Fit if the bin is mostly empty?
    
    Let's look at the remaining capacity `r`.
    If `r` is very small, the bin is almost full. Good to fill it up (Best Fit).
    If `r` is very large, the bin is almost empty. Maybe keep it empty for a big item? (Worst Fit).
    
    A smooth score function that favors both extremes of `r` (small and large) would be:
    Score = - (r - C/2)^2 ? No, C is not known.
    
    Let's assume the total capacity per bin is constant, say C.
    We don't have C in the signature. But we can infer it? No.
    
    Alternative interpretation of polarization:
    Minimize the "waste" in a way that clusters items.
    
    Let's try a score that penalizes intermediate remaining capacities.
    Let `r = bins - item`.
    We want to maximize the score.
    If we use Best Fit: Score = -r. (Minimizes r).
    If we use Worst Fit: Score = r. (Maximizes r).
    
    Polarization:
    Maybe: Score = -|r - alpha * bins| ?
    
    Let's look at the measured improvement: 1.0 -> 1.02.
    This suggests a modest improvement.
    
    Let's try a hybrid:
    If `bins[i]` is large (mostly empty), use Worst Fit logic (keep it empty).
    If `bins[i]` is small (mostly full), use Best Fit logic (fill it up).
    
    We can define a threshold. But without knowing C, we can use a relative measure.
    Let `fill_ratio = 1 - (bins[i] / C)`? We don't have C.
    
    However, we can look at the distribution of `bins`.
    Or simply:
    Score = - (bins[i] - item) if (bins[i] - item) < bins[i] / 2  # Best fit for nearly full bins?
    Score = (bins[i] - item) if (bins[i] - item) >= bins[i] / 2   # Worst fit for nearly empty bins?
    
    This creates a discontinuity.
    
    Let's try a smooth version:
    Let `r = bins - item`.
    Let `mid = bins / 2`.
    We want to push `r` away from `mid`.
    Score = -(r - mid)**2 ?
    This maximizes when r is close to 0 or close to bins.
    This seems like a good "polarization" score.
    
    Let's test this hypothesis.
    If r is close to 0 (bin becomes full), score is high (negative square is small negative).
    If r is close to bins (bin stays empty), score is high.
    If r is close to bins/2 (bin becomes half full), score is low (large negative).
    
    This matches the description "encourages bins to become either full or empty".
    
    Implementation:
    1. Filter bins where `bins >= item`.
    2. For eligible bins, calculate `r = bins - item`.
    3. Calculate `mid = bins / 2.0`.
    4. Score = -(r - mid)**2.
    5. Return scores. Ineligible bins get -inf or very low score.
    
    Wait, the function returns an array of scores. The caller picks the max.
    Ineligible bins should have a score lower than any eligible bin.
    
    Let's refine the score.
    Is `-(r - mid)**2` bounded? Yes.
    Is it monotonic in a useful way?
    
    Let's consider if we should scale it.
    The magnitude of the score doesn't matter, only the relative order.
    
    Let's write the code.
    
    Edge cases:
    - No eligible bins: return array of -inf? Or raise error? 
      The task says "scores each bin with enough remaining capacity".
      If none, the caller might fail. Standard practice in these challenges is to return -inf for ineligible.
    
    Let's implement:
    """
    eligible = bins >= item
    if not np.any(eligible):
        return np.full_like(bins, -np.inf)
    
    # Calculate remaining capacity after placing item
    r = bins - item
    # We only care about eligible bins
    r_eligible = r[eligible]
    bins_eligible = bins[eligible]
    
    # Polarization score: penalize intermediate fill levels
    # Target: r close to 0 (full) or r close to bins (empty)
    # Midpoint of remaining capacity range [0, bins] is bins/2
    mid = bins_eligible / 2.0
    scores_eligible = -(r_eligible - mid) ** 2
    
    # Construct full score array
    scores = np.full_like(bins, -np.inf, dtype=float)
    scores[eligible] = scores_eligible
    
    return scores
# EVOLVE-BLOCK-END
