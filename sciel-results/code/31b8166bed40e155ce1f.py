"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    best_score = np.inf
    best_x = None

    # Helper to calculate initial correlation and score
    def init_corr(x):
        # Linear auto-correlation
        # C[k] = sum_{i=0}^{N-1-k} x[i] * x[i+k]
        C = np.zeros(N)
        for k in range(N):
            C[k] = np.dot(x[:N-k], x[k:])
        return C

    def calc_score_from_corr(C, S):
        if S == 0:
            return np.inf
        # C[0] is S^2. We want max |C[k]| for k > 0.
        max_abs_corr = np.max(np.abs(C[1:]))
        return max_abs_corr / (S * S)

    # Local search with incremental correlation update
    def local_search(x, max_sweeps=50):
        current_x = x.copy()
        current_S = np.sum(current_x)
        current_C = init_corr(current_x)
        current_score = calc_score_from_corr(current_C, current_S)
        
        for sweep in range(max_sweeps):
            improved = False
            # Iterate through indices
            for i in range(N):
                # Flip x[i]
                old_val = current_x[i]
                current_x[i] = -old_val
                
                # Update S
                new_S = current_S - 2 * old_val
                
                # Update C
                # C[k] changes if x[i] or x[i+k] is involved.
                # C[k] = sum_{j=0}^{N-1-k} x[j] x[j+k]
                # x[i] appears in C[k] if:
                # 1. j = i => x[i] x[i+k]. Valid if i+k < N => k < N-i.
                #    Term changes from x[i]x[i+k] to -x[i]x[i+k]. Delta = -2 x[i] x[i+k].
                # 2. j+k = i => j = i-k. Valid if i-k >= 0 => k <= i.
                #    Term changes from x[i-k]x[i] to x[i-k](-x[i]). Delta = -2 x[i-k]x[i].
                
                # We need to update C for all k.
                # This is O(N) per flip.
                
                # Let's vectorize the update if possible, or just loop k?
                # Looping k in Python is slow.
                # We can update C using slicing.
                
                # Delta for k where i+k < N (k = 0 to N-i-1)
                # Term x[i] x[i+k]
                # Indices in C: k
                # Values: x[i] * x[i+k]
                # Delta: -2 * x[i] * x[i+k]
                
                # Delta for k where i-k >= 0 (k = 0 to i)
                # Term x[i-k] x[i]
                # Indices in C: k
                # Values: x[i-k] * x[i]
                # Delta: -2 * x[i-k] * x[i]
                
                # Note: C[0] is updated by both if i=0?
                # If k=0:
                # Case 1: x[0]x[0].
                # Case 2: x[0]x[0].
                # So C[0] gets -2 x[0]x[0] - 2 x[0]x[0] = -4?
                # Wait. C[0] = sum x[j]^2 = N. It is constant for +/-1.
                # My formula for C[0] update:
                # k=0.
                # Term 1: x[0]x[0]. Delta -2 x[0]^2 = -2.
                # Term 2: x[0]x[0]. Delta -2 x[0]^2 = -2.
                # Total delta -4.
                # But C[0] should not change!
                # Ah, C[0] = sum x[i]^2. Flipping x[i] does not change x[i]^2.
                # So C[0] is invariant.
                # My derivation:
                # C[k] = sum x[j] x[j+k].
                # If we flip x[i], C[k] changes.
                # For k=0, C[0] = sum x[j]^2. No change.
                # The terms x[i]x[i] appears only once in the sum for k=0?
                # Yes, j=i.
                # So why did I count it twice?
                # Case 1: j=i. Term x[i]x[i+k]. For k=0, x[i]x[i].
                # Case 2: j+k=i => j=i-k. For k=0, j=i. Term x[i]x[i].
                # So for k=0, it is the SAME term.
                # So we must not double count k=0.
                
                # Let's handle k=0 separately or adjust logic.
                # For k > 0:
                # Term 1: j=i, k < N-i.
                # Term 2: j=i-k, k <= i.
                # These are distinct terms unless i = i-k => k=0.
                
                # So for k > 0, we add both deltas.
                # For k = 0, delta is 0.
                
                # Vectorized update:
                # 1. Update range k=1 to N-i-1 (if any)
                #    Delta = -2 * old_val * current_x[i+1:N]
                #    current_C[1:N-i] += -2 * old_val * current_x[i+1:N]
                
                # 2. Update range k=1 to i (if any)
                #    Delta = -2 * old_val * current_x[i-1::-1] ?
                #    current_x[i-k] for k=1..i is current_x[i-1], current_x[i-2], ... current_x[0].
                #    So current_x[0:i][::-1]
                #    current_C[1:i+1] += -2 * old_val * current_x[0:i][::-1]
                
                # Check bounds
                if N - i > 1:
                    # k from 1 to N-i-1
                    # Indices in C: 1 to N-i
                    # Values: current_x[i+1 : N]
                    current_C[1:N-i] += -2 * old_val * current_x[i+1:N]
                
                if i > 0:
                    # k from 1 to i
                    # Indices in C: 1 to i
                    # Values: current_x[0:i] reversed
                    current_C[1:i+1] += -2 * old_val * current_x[0:i][::-1]
                
                # C[0] is unchanged.
                
                # Calculate new score
                if new_S == 0:
                    new_score = np.inf
                else:
                    max_abs_corr = np.max(np.abs(current_C[1:]))
                    new_score = max_abs_corr / (new_S * new_S)
                
                if new_score < current_score:
                    current_score = new_score
                    current_S = new_S
                    improved = True
                else:
                    # Revert
                    current_x[i] = old_val
                    current_S = current_S + 2 * old_val # Revert S
                    # Revert C
                    if N - i > 1:
                        current_C[1:N-i] += 2 * old_val * current_x[i+1:N]
                    if i > 0:
                        current_C[1:i+1] += 2 * old_val * current_x[0:i][::-1]
            
            if not improved:
                break
            
            if time.time() - start_time > time_limit - 5.0:
                break
        
        return current_x, current_score

    # Run multiple random restarts
    while time.time() - start_time < time_limit - 5.0:
        # Initialize random sequence
        x = np.random.choice([-1, 1], size=N)
        
        x_opt, score = local_search(x)
        
        if score < best_score:
            best_score = score
            best_x = x_opt
            
        # Also try starting from all ones
        if time.time() - start_time < time_limit - 5.0:
            x_seed = np.ones(N)
            x_seed_opt, score_seed = local_search(x_seed)
            if score_seed < best_score:
                best_score = score_seed
                best_x = x_seed_opt

    if best_x is None:
        return np.ones(N)
        
    return best_x

# EVOLVE-BLOCK-END
