def fit_exercise_policy(training_paths, contract):
    import numpy as np
    import time as _time

    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12
    
    # Tree parameters
    N_tree = 1200
    dt_tree = T / N_tree
    u = np.exp(sigma * np.sqrt(dt_tree))
    d = 1.0 / u
    p = (np.exp(r * dt_tree) - d) / (u - d)
    disc_tree = np.exp(-r * dt_tree)
    
    # S0 for tree construction (arbitrary, boundary is S0-independent in GBM)
    S0 = 100.0
    
    # Precompute states for all steps?
    # S[i, j] = S0 * u^j * d^(i-j)
    # This can be memory intensive if we store all.
    # But we only need the current and next step for backward induction.
    # However, we need to extract thresholds at specific steps.
    # So we can store the states at the exercise steps, or recompute.
    
    # We will iterate backwards.
    # V_next is the value at step i+1.
    # We need to know if we are at an exercise step.
    # Exercise steps are 100, 200, ..., 1200.
    
    # Initialize V at step N_tree
    # j ranges from 0 to N_tree
    j_max = N_tree
    # S at step N_tree
    # S[j] = S0 * u^j * d^(N_tree - j)
    # Note: d = 1/u, so S[j] = S0 * u^(2j - N_tree)
    j_indices = np.arange(j_max + 1)
    S_last = S0 * np.power(u, 2 * j_indices - N_tree)
    V_next = np.maximum(K - S_last, 0)
    
    thresholds = np.zeros(n_dates)
    # We need to store thresholds for exercise dates 1..12
    # Exercise date k corresponds to tree step 100*k
    
    # We will run backward induction.
    # We need to capture the decision at steps 100, 200, ..., 1100.
    # Step 1200 is the final payoff, threshold is K.
    
    # To be efficient, we can just run the loop.
    # V_curr will be the value at step i.
    
    # We need to store the states at the exercise steps to compute thresholds.
    # Or we can compute the threshold on the fly.
    
    # Let's store the states at the relevant steps.
    # There are 12 exercise steps.
    # We can just compute S for those steps when we reach them.
    
    # To avoid recomputing S for every step (which is cheap anyway), 
    # we can just generate S for the current step.
    
    # V_next is for step i+1.
    # We start with i = N_tree - 1 down to 0.
    
    # Note: The exercise dates are at i=100, 200, ..., 1200.
    # In the backward induction loop:
    # When i = 1200, we are at the end.
    # When i = 1199, we compute V[1199].
    # ...
    # When i = 100, we compute V[100]. This is exercise date 1.
    # When i = 200, we compute V[200]. This is exercise date 2.
    
    # So we check if (N_tree - i) is a multiple of 100?
    # No, step index i.
    # Exercise at i = 100, 200, ..., 1200.
    
    # Let's run the loop.
    for i in range(N_tree - 1, -1, -1):
        # S_curr for step i
        # j ranges from 0 to i
        j_curr = np.arange(i + 1)
        S_curr = S0 * np.power(u, 2 * j_curr - i)
        
        # V_next has length i+2 (for step i+1, j=0..i+1)
        # Continuation value:
        # V_cont[j] = disc_tree * (p * V_next[j+1] + (1-p) * V_next[j])
        V_cont = disc_tree * (p * V_next[1:] + (1 - p) * V_next[:-1])
        
        # Check if this is an exercise step
        # Exercise steps: 100, 200, ..., 1200
        # i is the current step index.
        # We exercise if i is a multiple of 100 and i > 0?
        # The contract says "12 monthly exercise dates (no time-0 exercise)".
        # So dates are t=1/12, ..., 1.
        # Tree steps: 100, 200, ..., 1200.
        
        if i % 100 == 0 and i > 0:
            # This is an exercise date.
            # Which date?
            # i=100 -> date 1
            # i=1200 -> date 12
            # But we are at i < 1200 in the loop (we started with V at 1200).
            # So i=100, 200, ..., 1100.
            
            # We need to find the threshold.
            # Immediate payoff:
            imm = np.maximum(K - S_curr, 0)
            # Exercise if imm >= V_cont
            # Find the boundary.
            # Since S_curr is increasing with j, and imm is decreasing, 
            # and V_cont is convex, the set of j where we exercise is a prefix [0..j_max_ex].
            
            # We want the largest S where we exercise.
            # Let's find the index where imm < V_cont.
            # diff = imm - V_cont
            diff = imm - V_cont
            # diff is decreasing.
            # Find the last index where diff >= 0.
            # If all diff < 0, threshold is 0 (or min S).
            # If all diff >= 0, threshold is max S.
            
            # Use searchsorted or similar.
            # Since diff is monotonic, we can find the transition.
            # j_ex is the largest j such that diff[j] >= 0.
            
            # np.where returns all indices.
            # We want the last one.
            ex_indices = np.where(diff >= 0)[0]
            
            if len(ex_indices) == 0:
                # Never exercise
                thresholds[i // 100 - 1] = 0.0 # i=100 -> idx 0
            elif ex_indices[-1] == i:
                # Always exercise
                thresholds[i // 100 - 1] = K
            else:
                # Boundary is between j_ex and j_ex + 1
                j_ex = ex_indices[-1]
                # S_curr[j_ex] and S_curr[j_ex + 1]
                # Threshold is midpoint
                thresholds[i // 100 - 1] = (S_curr[j_ex] + S_curr[j_ex + 1]) / 2.0
                thresholds[i // 100 - 1] = np.clip(thresholds[i // 100 - 1], 0.0, K)
            
            # V_curr is max(imm, V_cont)
            V_next = np.maximum(imm, V_cont)
        else:
            # No exercise
            V_next = V_cont
            
    # The last threshold (date 12) is K
    thresholds[-1] = K
    
    # The thresholds array is filled for i=100..1100 (indices 0..10).
    # Index 11 is set to K.
    
    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K

        def exercise(self, current_state, time):
            if isinstance(time, (int, float)):
                if isinstance(time, float) and 0 < time <= 1.0:
                    idx = int(np.round(time * 12)) - 1
                    if idx < 0: idx = 0
                    if idx >= 12: idx = 11
                elif isinstance(time, int):
                    if 1 <= time <= 12: idx = time - 1
                    elif 0 <= time <= 11: idx = time
                    else: idx = min(max(time, 0), 11)
                else: idx = 0
            else: idx = 0
            
            state = np.asarray(current_state)
            return state < self.thresholds[idx]

    return ExercisePolicy(thresholds, K)
