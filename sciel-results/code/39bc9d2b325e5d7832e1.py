"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
    
    # Strategy: Minimize fragmentation by scoring based on negative variance of remainders.
    # We want to maintain a tight cluster of remainders.
    # However, simply minimizing variance of ALL bins isn't right because we are choosing ONE bin.
    # The directive says: "scoring each bin based on the negative variance of the current bin remainder distribution".
    # This is slightly ambiguous. Does it mean the variance of the distribution of remainders of all *other* bins?
    # Or the variance of the distribution of remainders including this one?
    # "penalizing strategies that create a wide spread of unusable small gaps and favoring those that maintain a tight cluster"
    
    # Let's interpret: For each valid bin i, if we place item there, the new remainder is r_i.
    # We want to choose the bin that results in a state with lower "fragmentation".
    # A proxy for fragmentation is the variance of the remainders of the *used* bins (or all bins?).
    # Usually, in bin packing, we care about the bins that are currently in use (or have been opened).
    # But here `bins` is an array of capacities. It's likely `bins` represents the *remaining capacity* of available bins?
    # The prompt says: "bins: np.ndarray of one score per bin with enough remaining capacity".
    # And "each arriving item goes to the highest-scoring bin".
    # The input `bins` is likely the array of *remaining capacities* of the bins that are currently open/available.
    # Or is it the *total* capacity? "bins with enough remaining capacity" implies `bins` holds the remaining capacity.
    # Let's assume `bins` contains the remaining capacity for each bin.
    
    # If `bins` is remaining capacities, then `remaining` calculated above is `bins - item`.
    # Wait, if `bins` is remaining capacity, then `bins - item` is the new remaining capacity after placing item.
    # Let's call `new_remaining = bins - item`.
    
    # The directive suggests: score bin i based on -Var(remainders).
    # If we pick bin i, the set of remainders becomes:
    # {bins[j] for j != i} U {bins[i] - item}
    # We want to minimize the variance of this new set.
    # Minimizing variance encourages all remainders to be similar.
    # This is a "balance" heuristic.
    
    # However, calculating variance for each bin i is O(N^2) if done naively, or O(N) if we use incremental updates.
    # N is likely small (number of open bins).
    
    # Let's compute the mean and variance of the current remainders `bins`.
    # Let R be the vector of current remainders.
    # If we update bin i, the new vector R' has R'[i] = R[i] - item, and R'[j] = R[j] for j != i.
    
    # We want to minimize Var(R').
    # Var(R') = E[R'^2] - (E[R'])^2
    
    # Let S = sum(R), S2 = sum(R^2), N = len(R).
    # New sum S' = S - item
    # New sum of squares S2' = S2 - R[i]^2 + (R[i] - item)^2
    #   = S2 - R[i]^2 + R[i]^2 - 2*R[i]*item + item^2
    #   = S2 - 2*R[i]*item + item^2
    
    # Mean' = S' / N
    # E[R'^2] = S2' / N
    # Var' = S2'/N - (S'/N)^2 = (N*S2' - S'^2) / N^2
    
    # We want to minimize Var'. Since N is constant, we minimize (N*S2' - S'^2).
    # N*S2' = N*(S2 - 2*R[i]*item + item^2)
    # S'^2 = (S - item)^2 = S^2 - 2*S*item + item^2
    
    # Objective to minimize:
    # J(i) = N*(S2 - 2*R[i]*item + item^2) - (S^2 - 2*S*item + item^2)
    # J(i) = N*S2 - 2*N*R[i]*item + N*item^2 - S^2 + 2*S*item - item^2
    # J(i) = -2*N*item*R[i] + (N*item^2 - item^2) + (2*S*item) + (N*S2 - S^2)
    
    # Constants (independent of i):
    # C = N*item^2 - item^2 + 2*S*item + N*S2 - S^2
    # Variable part: -2*N*item*R[i]
    
    # So minimizing J(i) is equivalent to minimizing -2*N*item*R[i], which is equivalent to maximizing R[i].
    # Wait. If we minimize variance by maximizing R[i], that means we prefer bins with LARGER remaining capacity?
    # That sounds like Worst-Fit, not Best-Fit.
    # Let's re-evaluate.
    
    # If all bins have the same remainder, variance is 0.
    # If we have bins [50, 50] and item 10.
    # Option 1: Put in bin 0. New remainders [40, 50]. Var = 25.
    # Option 2: Put in bin 1. New remainders [50, 40]. Var = 25.
    # Same.
    
    # If we have bins [10, 90] and item 10.
    # Option 1: Put in bin 0 (R=10). New [0, 90]. Var = 2025.
    # Option 2: Put in bin 1 (R=90). New [10, 80]. Var = 1600.
    # Min variance prefers Option 2 (Worst Fit).
    
    # If we have bins [10, 20] and item 10.
    # Option 1: Put in bin 0 (R=10). New [0, 20]. Var = 100.
    # Option 2: Put in bin 1 (R=20). New [10, 10]. Var = 0.
    # Min variance prefers Option 2 (Worst Fit relative to size, but Best Fit relative to "filling up" the larger one to match the smaller one).
    
    # So minimizing variance tends to equalize remainders. This is a "Balance" strategy.
    # Is this better than Best Fit for Weibull?
    # Best Fit is known to be very strong.
    # The directive specifically asks to try this strategy.
    
    # Let's implement the score as -J(i) (since we want highest score to be chosen, and we want to minimize J(i)).
    # Score(i) = -J(i) = 2*N*item*R[i] - C
    # Since C is constant, we can just return 2*N*item*R[i] for valid bins.
    # This is equivalent to Worst-Fit (maximize remaining capacity).
    
    # Wait, is it exactly Worst Fit?
    # J(i) = -2*N*item*R[i] + C
    # Minimizing J(i) <=> Maximizing R[i].
    # Yes, this heuristic is mathematically equivalent to Worst-Fit when minimizing the variance of the *new* state of remainders.
    
    # Let's double check if there's any other interpretation.
    # "negative variance of the current bin remainder distribution"
    # Maybe it means: Score(i) = -Var(current remainders excluding i)?
    # Or Score(i) = -Var(current remainders including i's *potential* new remainder)?
    
    # If the strategy is just Worst-Fit, it's a known algorithm.
    # Let's implement it as described: Score based on -Var(new_state).
    # Which simplifies to maximizing R[i].
    
    # However, the prompt says "Keep the function names...".
    # And "Score: mean excess over the L1 lower bound".
    
    # Let's write the code to calculate the score based on the variance reduction logic, even if it simplifies to Worst Fit, to be faithful to the directive's phrasing "scoring each bin based on the negative variance".
    
    # Actually, if I just return `bins`, that is Worst Fit.
    # Let's see if there are edge cases.
    # If `bins` are remaining capacities.
    
    # Let's stick to the directive's specific phrasing which might imply a more complex calculation if I'm missing a nuance, but the math points to Worst Fit.
    # Let's implement the explicit variance calculation to be safe and clear, or just the simplified version?
    # The simplified version is faster.
    
    # Score = -Var(New Remainders)
    # Var(New) = (N*S2' - S'^2) / N^2
    # We want to maximize Score, so minimize Var(New).
    # Minimizing (N*S2' - S'^2)
    # N*S2' = N*(S2 - 2*R[i]*item + item^2)
    # S'^2 = (S - item)^2
    # Term = N*S2 - 2*N*R[i]*item + N*item^2 - (S^2 - 2*S*item + item^2)
    # Term = -2*N*item*R[i] + (N*S2 + N*item^2 - item^2 - S^2 + 2*S*item)
    # To minimize Term, we maximize R[i].
    
    # So, Score(i) = -Term(i) = 2*N*item*R[i] - Constant
    # We can just return R[i] (since 2*N*item is positive constant for a given item).
    # So this is exactly Worst-Fit.
    
    # Is Worst-Fit better than Best-Fit for Weibull?
    # Often Best-Fit is better for high utilization. Worst-Fit keeps bins balanced.
    # The seed program was Best-Fit with a tweak.
    # Let's try this Worst-Fit variant.
    
    scores = np.full_like(bins, -np.inf)
    if np.any(valid):
        # Score is proportional to remaining capacity (Worst Fit)
        scores[valid] = remaining[valid]
        
    return scores
# EVOLVE-BLOCK-END
