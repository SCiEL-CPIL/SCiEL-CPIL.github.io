"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes a heuristic matrix based on 'Global Difficulty Prioritization'.
    
    Strategy:
    The standard ACO heuristic often focuses on local complementarity (how well
    item i and j fit together). However, in bin packing, the primary failure mode
    is leaving "hard" items (large items that have few compatible partners) for
    the end of the packing sequence, where they are forced into new bins.
    
    This heuristic shifts the focus to **item difficulty**.
    1. We calculate a "difficulty" score for each item based on how few other
       items it can pair with within the capacity limit.
       - Large items have fewer valid partners -> Higher Difficulty.
       - Small items have many valid partners -> Lower Difficulty.
    
    2. The heuristic matrix H[i, j] represents the attractiveness of placing
       item j immediately after item i.
       
    3. To encourage placing hard items early, we make the transition INTO a
       difficult item (j) highly attractive, regardless of the specific pair
       complementarity (as long as they fit).
       
       Formula:
       - If demands[i] + demands[j] > capacity: H[i, j] = 1e-9 (invalid)
       - Else: H[i, j] = 1.0 + Difficulty[j]
       
       Why this works:
       - The ACO probability is proportional to Pheromone[i, j] * Heuristic[i, j].
       - By boosting H[i, j] based on Difficulty[j], the ants are statistically
         more likely to pick hard items (high Difficulty[j]) at any step.
       - This effectively implements a "First Fit Decreasing" bias within the
         stochastic ACO framework, ensuring that large items are placed while
         there is still plenty of space, rather than getting stranded.
       
       We also add a small bonus for tight fits to maintain some local packing
       quality, but the dominant term is the difficulty of the next item.
    """
    start_time = time.time()
    # Safety margin: return a valid default if we run out of time, though this is fast.
    # The computation is O(N^2) which is tiny for N=120.
    
    n = len(demands)
    demands = np.asarray(demands, dtype=float)
    
    # 1. Calculate Pair Sums and Validity
    pair_sums = demands[:, None] + demands[None, :]
    valid = pair_sums <= capacity
    
    # 2. Calculate Difficulty for each item
    # Difficulty is inversely proportional to the number of valid partners.
    # We count how many other items k can fit with item i.
    # valid[i, :] is a boolean array where valid[i, k] is True if i+k <= capacity.
    # Note: diagonal is valid (i+i <= capacity is possible), but we should probably 
    # consider distinct items. However, in the matrix context, H[i,j] is for i!=j usually,
    # but the evaluator might allow i=j if multiple copies exist? 
    # The task says "item j right after item i". In standard ACO bin packing, 
    # items are distinct indices. So i != j is implied for the placement, 
    # but the matrix is full. 
    # Let's count valid partners excluding self to be safe, or just use the row sum.
    # If an item is very large, it has few valid partners.
    
    # Count valid partners for each item i (excluding self for difficulty metric robustness)
    valid_counts = np.sum(valid, axis=1)
    # Ensure no division by zero. If an item fits with nothing (not even itself?), 
    # it's extremely hard. But demands are 20-99, cap 150. Min demand 20.
    # 20+20=40 <= 150. So every item fits with at least itself.
    # So valid_counts is at least 1.
    
    # Normalize difficulty.
    # We want high difficulty for items with few partners.
    # Difficulty = 1 / valid_counts
    # Let's scale it. Max valid counts is N. Min is 1.
    # Difficulty range: [1/N, 1].
    # Let's amplify the difference.
    difficulty = 1.0 / valid_counts
    
    # 3. Construct Heuristic Matrix
    # Base heuristic: 1.0 for all valid pairs.
    # Bonus: Difficulty of the *next* item (j).
    # This makes H[i, j] high if j is a "hard" item.
    
    # Create a column vector of difficulties to broadcast to columns j.
    # H[i, j] depends on difficulty[j].
    diff_col = difficulty[None, :]  # Shape (1, N)
    
    # Initialize with small value
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # For valid pairs, set eta = 1.0 + alpha * difficulty[j]
    # Let's use alpha = 5.0 to strongly bias towards hard items.
    # This means placing a hard item is 5x more attractive than an easy item, 
    # all else being equal (pheromone).
    
    # We also want to slightly favor tight fits to avoid wasting space, 
    # but the "hard first" strategy is usually dominant in bin packing.
    # Let's add a small tightness bonus: 1 / (gap + 1).
    # But keep it small so Difficulty dominates.
    
    remaining = capacity - pair_sums
    
    # Tightness score: higher for smaller gaps.
    # Only for valid pairs.
    tightness = np.zeros((n, n))
    tightness[valid] = 1.0 / (remaining[valid] + 1.0)
    
    # Combine
    # H = Base + Difficulty_Bonus + Tightness_Bonus
    # Base = 1.0
    # Difficulty_Bonus = 5.0 * difficulty[j]
    # Tightness_Bonus = 0.5 * tightness[i, j]
    
    eta[valid] = 1.0 + 5.0 * diff_col[valid] + 0.5 * tightness[valid]
    
    # Ensure diagonal is low (can't place item after itself in distinct item set)
    np.fill_diagonal(eta, 1e-9)
    
    # Check time
    if time.time() - start_time > 280:
        # Fallback to simple heuristic if somehow slow (unlikely)
        return np.full((n, n), 1.0)
        
    return eta
# EVOLVE-BLOCK-END
