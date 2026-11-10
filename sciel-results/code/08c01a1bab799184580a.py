import numpy as np
import math
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option using a high-resolution
    binomial tree backward induction.
    """
    start_wall = time.time()
    deadline = 140.0  # Safety margin before the 150s kill

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    
    # Use a high-resolution tree for accuracy
    # 1200 steps provides high accuracy for GBM
    N = 1200
    
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    
    # Map exercise dates to tree indices
    # Exercise dates: 1/12, 2/12, ..., 12/12
    # Tree step for exercise k (1-indexed): round(k * N / 12)
    ex_steps = [int(round(k * N / n_exercises)) for k in range(1, n_exercises + 1)]
    
    # Initialize the value array at the final step
    # At step N, the spot price ranges from S0 * d^N to S0 * u^N
    # We don't know S0 exactly from the contract in the same way, but 
    # the binomial tree is built around S0. 
    # The problem states Spot 90/100/110 are evaluation points, but the model is GBM.
    # The tree is typically centered at S0. 
    # The training_paths likely start at S0. Let's assume S0 is the initial spot.
    # However, the contract doesn't explicitly give S0. 
    # But the binomial tree method requires an S0.
    # Looking at the task: "Spot 90/100/110...". This implies the evaluation is done at these spots?
    # No, "Candidate receives current states only during scoring".
    # The training_paths will have an initial spot. We can infer S0 from training_paths[:, 0].
    # Or we can just build the tree relative to the spot at that time.
    # The binomial tree is a deterministic model. The "current_state" in evaluation is the spot price.
    # The policy should be: Exercise if S < Boundary(t).
    # The boundaries are derived from the tree.
    # The tree must be built starting from S0.
    # Let's assume S0 is the mean of the first column of training_paths, or just 100.0?
    # The task says "Spot 90/100/110... K 100". This suggests S0 is likely 100 or the paths start at 100.
    # Let's check the first element of training_paths.
    if training_paths is not None and len(training_paths) > 0:
        S0 = np.median(training_paths[:, 0])
    else:
        S0 = 100.0

    # Number of nodes at step j is j+1
    # We store values for the current step and next step
    # V[j, i] where i is the up-move index.
    # To save memory and time, we use 1D arrays for current and next level.
    
    # Final step N
    # Spot prices at step N: S0 * u^i * d^(N-i) for i=0..N
    # For a put, payoff is max(K - S, 0)
    # We only need to store the values.
    
    # Optimization: We only need to track the value and the boundary.
    # We can do the backward induction using arrays of size N+1.
    
    # V_next[i] is the value at step N, node i
    i_vals = np.arange(N + 1)
    S_next = S0 * (u ** i_vals) * (d ** (N - i_vals))
    V_next = np.maximum(K - S_next, 0.0)
    
    # Store boundaries for each exercise step
    # boundaries[k] for exercise date k (1..12)
    boundaries = [0.0] * n_exercises
    
    # We need to know which tree step corresponds to which exercise date
    # ex_steps is a list of 12 indices.
    # Let's create a mapping: step_index -> exercise_index (1-12) or -1
    ex_map = {}
    for idx, step in enumerate(ex_steps):
        ex_map[step] = idx
        
    # Backward induction
    # j goes from N-1 down to 0
    # Note: We don't exercise at step 0 (time 0).
    # Exercise dates are at steps in ex_steps.
    
    for j in range(N - 1, -1, -1):
        # Current step j
        # Nodes i = 0..j
        # Spot prices: S0 * u^i * d^(j-i)
        i_vals_curr = np.arange(j + 1)
        S_curr = S0 * (u ** i_vals_curr) * (d ** (j - i_vals_curr))
        
        # Continuation value
        # V_cont = exp(-r*dt) * (p * V_next[i] + (1-p) * V_next[i+1])
        # V_next has length j+2 (from step j+1)
        # V_next[i] and V_next[i+1]
        V_cont = math.exp(-r * dt) * (p * V_next[:j+1] + (1 - p) * V_next[1:j+2])
        
        # Check if this step is an exercise step
        if j in ex_map:
            ex_idx = ex_map[j]
            intrinsic = np.maximum(K - S_curr, 0.0)
            V_curr = np.maximum(V_cont, intrinsic)
            
            # Find the boundary
            # Exercise if S < Boundary
            # The boundary is where V_cont = intrinsic
            # We can find the index where the decision switches
            # For a put, if S is low, we exercise. If S is high, we continue.
            # So V_curr = intrinsic for i < i_switch, V_curr = V_cont for i >= i_switch.
            # We want the spot price at the switch.
            
            # Find the first i where V_cont >= intrinsic (do not exercise)
            # Actually, we want the critical spot.
            # Let's find the index i_crit such that for i < i_crit, we exercise, and i >= i_crit, we don't.
            # Note: i=0 is lowest spot (most down moves), i=j is highest spot.
            # So low i -> low S -> high intrinsic -> likely exercise.
            
            # Find the boundary in the array
            # We can interpolate between the last exercising node and the first continuing node.
            
            # Identify indices where we exercise
            exercise_mask = intrinsic > V_cont
            if np.any(exercise_mask):
                # Last exercising index
                i_last_ex = np.max(np.where(exercise_mask)[0])
                # First continuing index
                i_first_cont = i_last_ex + 1
                
                if i_first_cont <= j:
                    # We have a switch
                    S_ex = S_curr[i_last_ex]
                    S_cont = S_curr[i_first_cont]
                    V_ex = intrinsic[i_last_ex]
                    V_cont_val = V_cont[i_first_cont]
                    
                    # Linear interpolation for the boundary
                    # Assume linearity between these two nodes for the boundary
                    # This is an approximation, but with N=1200, the nodes are very close.
                    # A better way: The boundary is the root of V_cont(S) - Intrinsic(S) = 0.
                    # Given the high resolution, simple interpolation of S is very accurate.
                    delta_S = S_cont - S_ex
                    delta_V = V_cont_val - V_ex
                    if delta_V != 0:
                        # We want S where V_cont(S) = Intrinsic(S)
                        # This is not simply linear in S for the values, but the nodes are close.
                        # Let's just take the midpoint or the lower one?
                        # Actually, the optimal boundary is well-defined.
                        # With N=1200, the error is tiny.
                        # Let's use the spot of the last exercising node as a conservative estimate?
                        # Or interpolate?
                        # Let's interpolate the spot price.
                        # The value difference is small.
                        # S_boundary = S_ex + delta_S * (Intrinsic(S_ex) - V_cont(S_ex)) / (V_cont(S_cont) - V_cont(S_ex) - (Intrinsic(S_cont) - Intrinsic(S_ex)))
                        # This is getting complicated.
                        # Simple approach: The boundary is between S_ex and S_cont.
                        # Let's just use the average or the lower one.
                        # Using the lower one (S_ex) ensures we don't exercise when we shouldn't?
                        # No, if we exercise at S_ex, and S_boundary > S_ex, we might miss value.
                        # If we set boundary to S_cont, we might exercise when we shouldn't.
                        # Let's use linear interpolation of the "switching" point.
                        # The function f(S) = V_cont(S) - Intrinsic(S) changes sign.
                        # f(S_ex) < 0 (Exercise)
                        # f(S_cont) > 0 (Continue)
                        # Assume f is linear between S_ex and S_cont.
                        # f(S) = f(S_ex) + (S - S_ex) * (f(S_cont) - f(S_ex)) / (S_cont - S_ex)
                        # We want f(S) = 0.
                        # S_boundary = S_ex - f(S_ex) * (S_cont - S_ex) / (f(S_cont) - f(S_ex))
                        
                        f_ex = V_ex - V_cont[i_last_ex] # Should be > 0? No, intrinsic > V_cont, so f_ex = V_cont - Intrinsic < 0.
                        f_cont = V_cont_val - intrinsic[i_first_cont] # Should be > 0.
                        
                        if f_cont != f_ex:
                            S_boundary = S_ex - f_ex * (S_cont - S_ex) / (f_cont - f_ex)
                        else:
                            S_boundary = S_ex
                    else:
                        S_boundary = S_ex
                else:
                    # All nodes exercise? Unlikely for a put unless S is very low.
                    S_boundary = S_curr[j] # Highest spot
            else:
                # No nodes exercise
                S_boundary = S_curr[0] # Lowest spot, but if none exercise, boundary is below lowest spot.
                # Actually, if no nodes exercise, the boundary is effectively 0 or below the range.
                # We can set it to 0.
                S_boundary = 0.0
                
            boundaries[ex_idx] = S_boundary
        
        else:
            V_curr = V_cont
            
        V_next = V_curr
        
        # Deadline guard
        if time.time() > deadline:
            break

    # If we broke early, we might not have all boundaries.
    # Fill remaining with 0 (always exercise? No, 0 means never exercise if S>0).
    # Wait, if boundary is 0, we never exercise (since S>0).
    # If we didn't compute a boundary, it's better to default to a heuristic.
    # But N=1200 is very fast, so we should finish.
    
    class ExercisePolicy:
        def __init__(self, bounds, n_ex):
            self.bounds = bounds
            self.n_ex = n_ex
            
        def exercise(self, current_state, time_val):
            # time_val is in [0, T]
            # Map time_val to step index
            # Exercise dates: 1/12, 2/12, ..., 12/12
            step_float = time_val * self.n_ex
            step_idx = int(round(step_float)) - 1 # 0-indexed
            
            if step_idx < 0:
                return np.zeros_like(current_state, dtype=bool)
            if step_idx >= self.n_ex:
                step_idx = self.n_ex - 1
                
            boundary = self.bounds[step_idx]
            return current_state < boundary

    return ExercisePolicy(boundaries, n_exercises)
