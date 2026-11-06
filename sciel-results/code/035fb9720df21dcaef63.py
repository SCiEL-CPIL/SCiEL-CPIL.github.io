"""
Strategy: Dynamic Conflict-Aware Priority with Structural Heuristics.

Unlike the previous strategy which searched for a static ordering, this approach
recognizes that the optimal next element depends on the current state of the cap set.
The priority function computes a score based on:
1. Conflict Cost: The number of existing pairs in the set that would form a line with the candidate vector.
   We prefer vectors with lower conflict cost (greedy local optimization).
   To make this fast, we use a precomputed conflict matrix for n <= 6.
   For n > 6, we use a fast approximate conflict check or rely more heavily on structure.
2. Structural Balance: Vectors with symbol counts close to n/3 are preferred.
   This avoids early saturation of specific patterns (e.g., all 0s or 1s) which
   tends to create more lines later. This was measured to improve scores significantly.

The function `priority` maintains a global `selected_set`.
On each call, it updates the internal state.
If the vector is "accepted" (i.e., it doesn't form a line with existing pairs), 
it is added to `selected_set`.
The priority returned is designed to guide the external greedy sort.
However, since the external evaluator sorts ALL vectors by priority BEFORE adding them,
we cannot change the order dynamically in the standard sense if the evaluator 
collects all priorities first.

WAIT: The contract says "vectors ... are added greedily in descending priority".
This implies the evaluator might do:
1. Generate all vectors.
2. Call priority(v, n) for all v.
3. Sort v by priority descending.
4. Iterate through sorted v, adding to set if valid.

If the evaluator calls priority() for all vectors *before* the greedy loop starts, 
then `priority` cannot know the state of the set during the sorting phase.
It only knows the state if it is called *during* the addition loop.

Let's re-read carefully: "vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set".
Standard interpretation of such interfaces in these benchmarks (like FunSearch):
The evaluator usually does:
`ordered_vectors = sorted(all_vectors, key=lambda v: priority(v, n), reverse=True)`
Then:
`for v in ordered_vectors: if is_cap_set(current_set U {v}): current_set.add(v)`

If this is the case, `priority` is called in a batch. The state `selected_set` inside `priority` would be empty or static during the sorting phase.
Therefore, a purely dynamic "conflict with current set" strategy is impossible if the sorting happens before the loop.

HOWEVER, the previous program (score 1.1094) used a static priority.
The inspiration mentions "Measured: 1 -> 1.219" for a program that improved island 1 using "balanced symbol counts".
This suggests that structural heuristics are key.

Let's look at the "Dynamic" possibility again.
If the evaluator calls `priority` *inside* the loop (i.e., it asks for the next best vector repeatedly), then dynamic works.
But "added greedily in descending priority" strongly implies a sort-first approach.

Let's assume the Sort-First approach.
In this case, `priority(v, n)` must return a value that *predicts* how good `v` will be in the final set, without knowing the final set.
The best predictor is:
1. Vectors that are "harder" to place (more restrictive) should have lower priority? Or higher?
   If we place restrictive vectors first, we might block many others.
   If we place flexible vectors first, we fill the set with "safe" vectors.
   Actually, for Cap Sets, vectors with balanced counts (1, 1, 1...) are often the "core" of large cap sets.
   Vectors with many 0s or 2s are often "extremes".
   
   Let's test the hypothesis: **Prioritize vectors with the most balanced symbol counts.**
   This aligns with the inspiration: "Prioritizing vectors with balanced symbol counts ... measured to improve score".

   Additionally, we can use a secondary tie-breaker based on a hash to ensure determinism and spread.
   
   Let's refine the "Balanced" metric.
   For n dimensions, the ideal count for each symbol is n/3.
   Let `count_0`, `count_1`, `count_2` be the counts.
   Deviation = sum(|count_i - n/3|).
   Lower deviation = Higher Priority.
   
   Let's combine this with a "Sparsity" or "Entropy" measure.
   
   Another idea: **Look-ahead / Local Density**.
   Even if we can't know the set, we can estimate the "local density" of conflicts in the vector space.
   Vectors that are "far" from other vectors in some metric might be safer.
   
   Let's stick to the strongest measured signal: **Balanced Symbol Counts**.
   We will define priority as:
   `score = -deviation_from_balance + small_hash_tiebreaker`
   
   To make it more robust, we can also consider the "sum" of coordinates.
   Balanced vectors have sum ~ n.
   
   Let's implement a priority function that:
   1. Calculates the deviation of symbol counts from n/3.
   2. Rewards lower deviation.
   3. Uses a high-resolution hash as a tie-breaker to ensure all vectors have unique priorities (avoiding ties in the sort).
   
   This is fundamentally different from the "Offline Search for Static Order" because it doesn't simulate the greedy process to find a global ordering. It directly optimizes the local properties of the vectors that are known to correlate with large cap sets. It's a heuristic based on structural properties rather than a search over orderings.
"""

import time
import math
import itertools
import random
import hashlib

# EVOLVE-BLOCK-START

# We don't need offline search if the heuristic is strong.
# But let's add a small amount of "tuning" for the balance weight if possible?
# No, keep it simple and deterministic.

def _calculate_balance_score(vector, n):
    """
    Calculate a score based on how balanced the symbol counts are.
    Higher score = more balanced.
    """
    count_0 = vector.count(0)
    count_1 = vector.count(1)
    count_2 = vector.count(2)
    
    # Ideal count is n/3
    ideal = n / 3.0
    
    # Deviation from ideal
    # Using squared deviation to penalize large imbalances more
    deviation = (count_0 - ideal)**2 + (count_1 - ideal)**2 + (count_2 - ideal)**2
    
    # We want higher priority for lower deviation.
    # So score is negative deviation.
    return -deviation

def _calculate_hash_tiebreaker(vector, n):
    """
    Generate a deterministic, unique-ish float for tie-breaking.
    """
    # Convert vector to string for hashing
    v_str = ''.join(str(x) for x in vector)
    # Use a fast hash
    h = hashlib.md5(v_str.encode('utf-8')).hexdigest()
    # Convert to float in [0, 1)
    # Take first 16 hex chars
    val = int(h[:16], 16)
    max_val = 16**16
    return val / max_val

def priority(vector, n):
    """
    Returns a priority score for the vector.
    Strategy:
    1. Primary: Balance of symbol counts (0, 1, 2).
       Vectors closer to (n/3, n/3, n/3) are prioritized.
    2. Secondary: Deterministic hash to break ties and add slight randomization
       in the ordering of equally balanced vectors, which can help escape local minima
       in the greedy selection process.
    """
    # Calculate balance score
    balance_score = _calculate_balance_score(vector, n)
    
    # Calculate tiebreaker
    # We want the tiebreaker to be small so it doesn't override the balance score significantly
    # unless the balance scores are very close.
    # The deviation score is an integer or simple float.
    # Let's scale the hash to [0, 0.01] so it only affects the 4th decimal place roughly.
    tiebreaker = _calculate_hash_tiebreaker(vector, n) * 0.01
    
    # Final priority
    # Higher is better.
    # Balance score is negative deviation (so -0.0 is best, -large is bad).
    # So we want to maximize balance_score.
    return balance_score + tiebreaker

# EVOLVE-BLOCK-END
