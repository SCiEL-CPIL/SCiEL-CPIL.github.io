import numpy as np
import time

def heuristics(prizes, weights):
    # 1. Input processing and validation
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    if prizes.ndim != 2 or weights.ndim != 2:
        # Fallback for unexpected shapes, though contract says (5, 100)
        try:
            prizes = prizes.reshape(5, -1)
            weights = weights.reshape(5, -1)
        except:
            # Last resort: return uniform heuristics
            n = prizes.size // 5 if prizes.size >= 5 else prizes.size
            return np.ones(n)
            
    if prizes.shape != weights.shape:
        # Try to fix if one is 1D
        if prizes.ndim == 1:
            prizes = prizes.reshape(1, -1)
            # This is risky, but better than crashing
        elif weights.ndim == 1:
            weights = weights.reshape(1, -1)
            
    m, n = weights.shape
    capacity = 25.0
    
    # Total prize per item (sum across constraints if 5D, or just the value if 1D constraint)
    # The problem states prizes shape (5, 100). "Highest total prize" implies summing the prizes.
    P = prizes.sum(axis=0)
    w = weights # (m, n)
    
    # 2. Time Guard
    t_start = time.time()
    deadline = t_start + 280.0 # 20s safety margin for 300s limit
    
    def check_time():
        return time.time() > deadline
        
    # 3. Pareto Dominance Filter Strategy
    # We define an item i as dominated by item j if:
    # P_j >= P_i (j has higher or equal prize)
    # AND w_j <= w_i in ALL constraints (j uses less or equal resource in every dimension)
    # AND (P_j > P_i OR any w_jk < w_ik) (strictly better in at least one aspect)
    
    # Since m=5, n=100, we can do an O(n^2 * m) check. 100*100*5 = 50k ops. Very fast.
    
    is_dominated = np.zeros(n, dtype=bool)
    
    # Precompute for speed? 
    # For each i, check if any j dominates it.
    # Vectorized approach:
    # P is (n,), w is (m, n)
    
    # Let's compute dominance matrix D where D[i,j] is True if j dominates i
    # Condition 1: P[j] >= P[i]
    # Condition 2: w[:, j] <= w[:, i] for all k
    # Condition 3: (P[j] > P[i]) OR (exists k s.t. w[k, j] < w[k, i])
    
    # We can loop over j (the potential dominators) and mark dominated items.
    # This is O(n^2 * m) but with small constants.
    
    for j in range(n):
        if is_dominated[j]:
            continue # j is dominated, so it cannot dominate others in a strict Pareto sense? 
                     # Actually, a dominated item CAN dominate another item if it's "less bad"?
                     # No, if j is dominated by k, and j dominates i, then k dominates i (transitivity).
                     # So we only need to check non-dominated items as potential dominators.
                     # However, for the *first* pass, we don't know who is non-dominated.
                     # Standard algorithm: 
                     # 1. Assume all non-dominated.
                     # 2. For each pair (i, j), if j dominates i, mark i as dominated.
                     # 
                     # Let's just do the full pair check for simplicity and robustness, 
                     # n=100 is tiny.
        
    # Reset for full check
    is_dominated = np.zeros(n, dtype=bool)
    
    # Vectorized dominance check
    # P_j >= P_i: (n, n) boolean matrix, M_p[i, j] = (P[j] >= P[i])
    M_p = P[None, :] >= P[:, None]
    
    # w_j <= w_i for all k:
    # w is (m, n). 
    # w[k, j] <= w[k, i] for all k.
    # Let's create a matrix M_w where M_w[i, j] is True if w[:, j] <= w[:, i].
    # w[:, j] is shape (m,). w[:, i] is shape (m,).
    # We can do: (w.T <= w.T[:, None, :]) -> shape (n, n, m)
    # Then check all along axis 2.
    
    # w.T is (n, m)
    # w.T[:, None, :] is (n, 1, m) -> w_i
    # w.T[None, :, :] is (1, n, m) -> w_j
    # Condition: w_j <= w_i  => w.T[None, :, :] <= w.T[:, None, :]
    # This results in (n, n, m) where indices are [i, j, k].
    # We want M_w[i, j] = all( w[j, k] <= w[i, k] for k )
    
    # Memory: 100 * 100 * 5 = 50k floats. Trivial.
    
    w_expanded_i = w.T[:, None, :] # (n, 1, m)
    w_expanded_j = w.T[None, :, :] # (1, n, m)
    
    # Check if j's weights are <= i's weights in all dimensions
    M_w_le = np.all(w_expanded_j <= w_expanded_i, axis=2) # (n, n)
    
    # Strict improvement check:
    # (P[j] > P[i]) OR (exists k s.t. w[j, k] < w[i, k])
    
    M_p_gt = P[None, :] > P[:, None]
    
    # Exists k s.t. w[j, k] < w[i, k]
    M_w_lt = np.any(w_expanded_j < w_expanded_i, axis=2)
    
    M_strict = M_p_gt | M_w_lt
    
    # j dominates i if: M_p[i, j] AND M_w_le[i, j] AND M_strict[i, j]
    # Note: M_p[i, j] is P[j] >= P[i].
    # M_w_le[i, j] is w[j] <= w[i].
    # M_strict[i, j] is strict improvement.
    
    dominance_mask = M_p & M_w_le & M_strict
    
    # If any j dominates i, then i is dominated.
    # dominance_mask[i, j] is True if j dominates i.
    # We want to know for each i, if ANY j dominates it.
    is_dominated = np.any(dominance_mask, axis=1)
    
    # 4. Construct Heuristic
    # Items NOT dominated are "Pareto Optimal".
    # Assign high score to them, low score to others.
    
    # To make it a valid heuristic for ACO (proportional probability),
    # we can assign:
    # Pareto items: 1.0
    # Dominated items: 1e-5 (very low, but positive to avoid zero prob issues if needed, 
    #                   though the prompt says non-positive becomes 1e-9)
    
    # However, a flat 1.0 for all Pareto items might be too uniform.
    # We can scale by prize among Pareto items to add some discrimination.
    # Or simply use the prize itself for Pareto items and a tiny epsilon for others.
    
    # Let's try:
    # h[i] = P[i] if not dominated, else 1e-5
    # This forces the ACO to focus almost exclusively on the Pareto front.
    # Since P is uniform [0,1] sum, P is in [0, 5].
    
    h = np.where(~is_dominated, P, 1e-5)
    
    # Ensure finite and positive
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    # 5. Optional: Quick Simulation to Verify/Adjust?
    # The Pareto set is a strong structural prior. 
    # Given the time budget, we could run a very fast simulation to see if this heuristic
    # yields a good result, or if we should blend it with a ratio.
    # But the directive asks for a STRUCTURAL change. The Pareto filter IS the change.
    # Tuning it with simulation might drift back to the old strategy.
    # Let's stick to the pure Pareto heuristic but add a small noise or scaling if needed.
    
    # Actually, let's consider that some Pareto items might be "expensive" (high weight) 
    # and some "cheap". The ACO pheromone will handle the "which ones fit" part.
    # The heuristic just says "these are the best candidates".
    
    # One refinement: If the Pareto set is too large (e.g., >50 items), 
    # we might want to rank them by Prize/Weight ratio within the Pareto set.
    # If the set is small, keep them all high.
    
    n_pareto = np.sum(~is_dominated)
    
    if n_pareto > 0:
        # Normalize Pareto scores to be around 1.0 to compete with initial pheromone (1.0)
        # If P is large (up to 5), h is up to 5. ph=1. prob ~ 5.
        # If we normalize, we might lose information.
        # Let's just scale P by 0.2 so max is ~1.0? 
        # Or leave it. The relative probability is what matters.
        # h_pareto = P[~is_dominated]
        # h_dominated = 1e-5
        
        # Let's try a hybrid: 
        # Score = (1 - is_dominated) * (P / max_P) + 1e-5
        # This makes Pareto items have score proportional to their prize, 
        # and dominated items near 0.
        
        max_P = P.max() if P.max() > 0 else 1.0
        h = (1.0 - is_dominated.astype(float)) * (P / max_P) + 1e-5
        
        # Ensure the dominated ones are definitely low
        h[is_dominated] = 1e-5
        
    else:
        # Fallback if no Pareto items found (should not happen with unique random data)
        h = P / (P.max() + 1e-9)
        
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h

