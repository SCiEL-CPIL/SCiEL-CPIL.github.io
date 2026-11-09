"""Bermudan exercise-policy: Longstaff-Schwarz Monte Carlo Least Squares (LSM) method."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import curve_fit

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)

    N, n_total_steps = training_paths.shape
    n_exercise = 12  # monthly exercise dates
    dt = T / n_exercise
    discount = np.exp(-r * dt)

    # Ensure we have enough steps. If training_paths has fewer than n_exercise+1 steps,
    # we might need to handle it, but the contract says 12 monthly dates, so likely 13 steps (0-12).
    # Indices 1..12 correspond to exercise dates t=1/12 .. 12/12.
    
    if n_total_steps < n_exercise + 1:
        # Fallback: if data is insufficient, use a simple heuristic
        # But usually n_total_steps = 13
        pass

    # We work with paths at exercise dates: indices 1 to n_exercise
    # training_paths[:, 1:n_exercise+1] shape (N, 12)
    
    # We need to compute the continuation value at each step k=11 down to 1.
    # V_k = max(Payoff_k, Continuation_k)
    # Continuation_k = Discount * E[V_{k+1} | S_k]
    
    # In LSM, we estimate E[V_{k+1} | S_k] by regressing V_{k+1} on features of S_k.
    # However, standard LSM uses the *simulated* future payoffs/values.
    # We have the full paths, so we can compute V_{k+1} exactly on the training paths
    # (since we know the future prices in the training set) and then regress V_{k+1} on S_k.
    
    # Wait, V_{k+1} depends on the optimal policy from k+1 onwards.
    # We compute V backwards.
    # At k=12 (last step), V_12 = Payoff_12 = max(K - S_12, 0).
    # At k=11, we regress V_12 on S_11 to get E[V_12 | S_11].
    # Then Cont_11 = Discount * Regressed_V_12.
    # V_11 = max(Payoff_11, Cont_11).
    # At k=10, we regress V_11 on S_10 to get E[V_11 | S_10].
    # And so on.
    
    # We store V_k for each path at each step k.
    V = np.zeros((N, n_exercise))
    
    # Step 12 (index 11)
    S_12 = training_paths[:, n_exercise]
    V[:, 11] = np.maximum(K - S_12, 0.0)
    
    # We also need to store the decision to exercise or continue to define the policy.
    # The policy is: exercise if Payoff_k >= Cont_k.
    # We will store the threshold or the boolean mask for each k.
    # Since the state is 1D (Spot), the optimal policy is a threshold policy: Exercise if S <= Threshold_k.
    # We can find the threshold by checking where Payoff > Cont.
    
    # To make the policy robust for evaluation paths (which are independent),
    # we should fit a function that approximates the boundary or use the regressed values
    # to determine the boundary on the training data, and then generalize.
    # A simple generalization: find the threshold on the training data for each k.
    # Since the underlying model is GBM and the payoff is linear, the boundary should be monotonic.
    
    thresholds = np.zeros(n_exercise)
    
    # We need to be careful with the regression.
    # Features: S_k, S_k^2, S_k^3, ... or just S_k?
    # For a put, the value is roughly linear in S for S < K and 0 for S > K (roughly).
    # A polynomial of degree 3 or 4 in S is usually sufficient.
    # Let's use a basis: [1, S, S^2, S^3]
    
    def make_features(S):
        return np.column_stack([np.ones_like(S), S, S**2, S**3])
    
    # We perform the backward induction
    for k in range(n_exercise - 2, -1, -1):
        if time.time() > deadline - 5.0:
            break
            
        S_k = training_paths[:, k + 1] # Exercise date k is at index k+1 in training_paths (0-indexed, 0 is t=0)
        # Wait, indices:
        # training_paths[:, 0] is t=0
        # training_paths[:, 1] is t=1/12 (k=1)
        # ...
        # training_paths[:, 12] is t=12/12 (k=12)
        
        # So for k=11 (t=11/12), S_k is training_paths[:, 11].
        # V_{k+1} is V[:, k+1].
        
        # Regress V[:, k+1] on S_k
        X = make_features(S_k)
        y = V[:, k+1]
        
        try:
            # Solve least squares
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            # Predicted continuation value (undiscounted)
            cont_pred = X @ coeffs
            cont_val = discount * cont_pred
            
            # Payoff at step k
            payoff = np.maximum(K - S_k, 0.0)
            
            # Update V for step k
            V[:, k] = np.maximum(payoff, cont_val)
            
            # Determine exercise boundary
            # Exercise if payoff >= cont_val
            exercise_mask = payoff >= cont_val
            
            # Find the threshold
            # Sort by S_k to find the transition
            sort_idx = np.argsort(S_k)
            S_sorted = S_k[sort_idx]
            mask_sorted = exercise_mask[sort_idx]
            
            # Find the last index where mask is True
            # If all True, threshold is max S. If all False, threshold is 0 (or min S)
            # Since it's a put, we expect True for low S, False for high S.
            
            # Find the transition point
            # diff of mask
            if np.all(mask_sorted):
                thresholds[k] = S_sorted[-1]
            elif not np.any(mask_sorted):
                thresholds[k] = S_sorted[0] # Or 0? If we never exercise, threshold should be low.
            else:
                # Find the last True
                last_true_idx = np.where(mask_sorted)[0][-1]
                # Interpolate between last_true_idx and last_true_idx+1
                s1 = S_sorted[last_true_idx]
                s2 = S_sorted[last_true_idx + 1]
                # At s1, payoff >= cont. At s2, payoff < cont.
                # We want the point where payoff = cont.
                # payoff(s) = K - s
                # cont(s) is linear-ish from regression.
                # Let's just take the midpoint or the upper bound of the exercise region.
                # Using the upper bound is safer for a put (exercise more aggressively if uncertain).
                # Or interpolate.
                p1 = payoff[sort_idx][last_true_idx] - cont_val[sort_idx][last_true_idx]
                p2 = payoff[sort_idx][last_true_idx + 1] - cont_val[sort_idx][last_true_idx + 1]
                if p1 != p2:
                    frac = -p1 / (p2 - p1)
                    thresholds[k] = s1 + frac * (s2 - s1)
                else:
                    thresholds[k] = s1
        except Exception:
            # Fallback: use previous threshold or K
            if k > 0:
                thresholds[k] = thresholds[k+1]
            else:
                thresholds[k] = K

    # Handle the last step k=11 (index 11)
    # At maturity, always exercise if ITM.
    thresholds[11] = K
    
    # Refine thresholds: ensure monotonicity? 
    # For a Bermudan put, the exercise boundary is generally non-increasing as time approaches maturity?
    # Actually, for American puts, the boundary increases as time to maturity decreases (you're more likely to exercise closer to expiry if the option is ITM).
    # So Threshold(t) should be increasing as t -> T?
    # Let's check: At t=0, boundary is low. At t=T, boundary is K.
    # So thresholds[0] < thresholds[1] < ... < thresholds[11] = K.
    # Let's enforce this monotonicity to reduce noise from regression.
    
    for i in range(1, n_exercise):
        if thresholds[i] < thresholds[i-1]:
            thresholds[i] = thresholds[i-1]
            
    # Also, thresholds should not exceed K for puts (exercise only if ITM).
    thresholds = np.minimum(thresholds, K)
    
    class Policy:
        def __init__(self, thr, K_val):
            self.thresholds = thr
            self.K = K_val
            
        def exercise(self, current_state, time):
            # time is integer 1..12
            current_state = np.asarray(current_state, dtype=np.float64)
            if time >= 12:
                return current_state < self.K
            if time < 1 or time > 11:
                return current_state < self.K
            # time is 1..11, use thresholds[time-1]
            thr = self.thresholds[time - 1]
            return current_state < thr

    return Policy(thresholds, K)

# EVOLVE-BLOCK-END
