import numpy as np
import time
import math

N = 400

def _compute_autocorr(x):
    """Compute periodic autocorrelation R[k] for all k using FFT."""
    X = np.fft.fft(x.astype(np.float64))
    R = np.real(np.fft.ifft(X * np.conj(X)))
    return R

def _compute_c3_bound(R):
    """Compute C3 upper bound from autocorrelation array."""
    # Peak sidelobe level: max absolute value of R[k] for k != 0
    # R[0] is the energy, which is N for +/-1 sequences.
    return np.max(np.abs(R[1:]))

def _fast_flip_autocorr_update(R, x, pos):
    """
    Fast update of autocorrelation when flipping x[pos].
    
    Returns the new autocorrelation array.
    """
    v = x[pos]
    new_R = R.copy()
    # We need to update R[k] for all k.
    # R[k] = sum_i x[i] x[(i+k)%N]
    # Flipping x[pos] changes terms where i=pos or (i+k)%N=pos.
    # Term 1: i=pos -> x[pos] x[(pos+k)%N] becomes -x[pos] x[(pos+k)%N]. Change: -2 x[pos] x[(pos+k)%N]
    # Term 2: (i+k)%N=pos -> i=(pos-k)%N. x[(pos-k)%N] x[pos] becomes x[(pos-k)%N] (-x[pos]). Change: -2 x[(pos-k)%N] x[pos]
    # If k=0, both terms are the same: x[pos]^2 becomes -x[pos]^2. Change: -2 x[pos]^2 = -2.
    # If k!=0, the two terms are distinct unless (pos+k)%N == (pos-k)%N, which implies 2k=0 mod N.
    # For N=400, 2k=0 mod 400 => k=0 or k=200.
    
    # Vectorized update might be faster, but let's stick to the loop for clarity and correctness,
    # or try to vectorize.
    # Let's try vectorization.
    # k ranges from 0 to N-1.
    # indices1 = (pos + k) % N
    # indices2 = (pos - k) % N
    
    ks = np.arange(N)
    idx1 = (pos + ks) % N
    idx2 = (pos - ks) % N
    
    # Delta R[k] = -2 * v * x[idx1] - 2 * v * x[idx2]
    # But if k==0, idx1==idx2==pos, so delta = -4*v*x[pos] = -4*v^2 = -4. 
    # Wait, the formula derived:
    # R_new[k] = R_old[k] - 2*v*x[(pos+k)%N] - 2*v*x[(pos-k)%N]
    # If k=0: R_new[0] = R_old[0] - 2*v*x[pos] - 2*v*x[pos] = R_old[0] - 4v^2 = R_old[0] - 4.
    # Is this correct?
    # R[0] = sum x[i]^2 = N.
    # Flip x[pos] from 1 to -1. x[pos]^2 is still 1. So R[0] should remain N.
    # My derivation:
    # Term i=pos: x[pos]x[pos] -> -x[pos]x[pos]. Change -2 x[pos]^2 = -2.
    # Term (i+k)%N=pos: for k=0, i=pos. Same term.
    # So total change is -2.
    # Why did the formula give -4? Because I counted the term twice?
    # Let's re-verify.
    # R[k] = sum_i x[i] x[(i+k)%N]
    # Let S = {i | i=pos or (i+k)%N=pos}
    # If k=0, S={pos}. Term is x[pos]^2. New term -x[pos]^2. Change -2 x[pos]^2 = -2.
    # If k!=0, S={pos, (pos-k)%N}.
    # Term 1 (i=pos): x[pos]x[(pos+k)%N]. Change -2 x[pos]x[(pos+k)%N].
    # Term 2 (i=(pos-k)%N): x[(pos-k)%N]x[pos]. Change -2 x[(pos-k)%N]x[pos].
    # Total change: -2v (x[(pos+k)%N] + x[(pos-k)%N]).
    # If k=200 (N/2), pos+k == pos-k mod N.
    # Then idx1 == idx2.
    # Change = -4 v x[(pos+200)%N].
    
    # So the vectorized formula:
    delta = -2 * v * (x[idx1] + x[idx2])
    # Correct for k=0?
    # idx1[0] = pos, idx2[0] = pos.
    # delta[0] = -2 * v * (x[pos] + x[pos]) = -4 v^2 = -4.
    # But we established change is -2.
    # Why? Because when k=0, the term is counted only once in the sum?
    # No, the sum is over i. For k=0, R[0] = sum x[i]^2.
    # The term corresponding to i=pos is x[pos]^2.
    # In the update formula, we identified terms where x[pos] appears.
    # For k=0, x[pos] appears in x[i] when i=pos, and in x[(i+k)%N] when i=pos.
    # It is the SAME term x[pos]*x[pos].
    # So we should only subtract 2*x[pos]^2, not 4.
    # The formula -2v(x[idx1] + x[idx2]) assumes the two occurrences are distinct.
    # They are distinct if (pos+k)%N != (pos-k)%N.
    # They are same if k=0 or k=N/2 (if N even).
    
    # Let's handle k=0 and k=N/2 separately or adjust the formula.
    # Actually, simpler:
    # new_R = R.copy()
    # for k in range(N):
    #    if k == 0:
    #       new_R[k] = R[k] - 2
    #    else:
    #       new_R[k] = R[k] - 2 * v * (x[(pos+k)%N] + x[(pos-k)%N])
    
    # This loop is O(N). N=400. 400 operations per flip.
    # If we do 100,000 flips, that's 40 million operations. Python loop might be slow.
    # Let's try to vectorize correctly.
    
    # mask_same = (idx1 == idx2)
    # For k=0, idx1==idx2==pos.
    # For k=N/2, idx1==idx2==(pos+N/2)%N.
    
    # If we use the loop, it's safer.
    # Let's benchmark in head: 400 iterations in Python is ~50-100 microseconds?
    # 100,000 flips -> 5-10 seconds. This is acceptable.
    
    # Let's use the loop for correctness and simplicity, as vectorization has edge cases.
    # However, we can optimize the loop by using numpy arrays for x.
    
    new_R = R.copy()
    # Precompute indices? No, they change with pos.
    
    # We can do:
    # new_R[0] = R[0] - 2
    # new_R[1:] = R[1:] - 2 * v * (x[(pos + np.arange(1,N)) % N] + x[(pos - np.arange(1,N)) % N])
    
    # Let's do that.
    new_R[0] = R[0] - 2 * v * v # Which is -2 since v is +/-1
    
    if N > 1:
        ks = np.arange(1, N)
        idx1 = (pos + ks) % N
        idx2 = (pos - ks) % N
        # Check if any idx1 == idx2 for k>0
        # This happens if 2k = 0 mod N => k = N/2.
        # If N is even, k=N/2 is in range 1..N-1.
        # At k=N/2, idx1 == idx2.
        # The term is -2 * v * (x[idx] + x[idx]) = -4 v x[idx].
        # Is this correct?
        # R[k] = sum x[i] x[i+k].
        # Terms with x[pos]:
        # i=pos: x[pos] x[pos+k]. Change -2 x[pos] x[pos+k].
        # i=pos-k: x[pos-k] x[pos]. Change -2 x[pos-k] x[pos].
        # If k=N/2, pos+k = pos-k mod N.
        # So the two terms are the same term in the sum?
        # No, they are different i values: i=pos and i=pos-N/2.
        # Unless pos = pos-N/2 mod N, which implies N/2=0, impossible.
        # So they are distinct terms in the sum.
        # So the change is indeed sum of both changes.
        # So the vectorized formula holds for all k.
        
        new_R[1:] = R[1:] - 2 * v * (x[idx1] + x[idx2])
        
    return new_R

def _generate_m_sequence(m):
    """Generate an m-sequence of length 2^m - 1 using LFSR."""
    if m == 9:
        taps = (1 << 4)
        length = (1 << m) - 1
    else:
        # Fallback to random if m not supported, but we only use 9
        m = 9
        taps = (1 << 4)
        length = (1 << m) - 1
    
    state = (1 << m) - 1
    seq = []
    for i in range(length):
        bit = state & 1
        seq.append(1 if bit == 1 else -1)
        fb = 0
        for t in range(m):
            if (taps >> t) & 1:
                fb ^= (state >> t) & 1
        state = (state >> 1) | (fb << (m - 1))
    
    return np.array(seq, dtype=np.float64)

def construct():
    """Construct a signed step function (sequence of ±1) to minimize C3 upper bound."""
    deadline = time.time() + 115  # Safety margin
    
    # Initialize with m-sequence
    mseq = _generate_m_sequence(9)
    x = mseq[:N].copy()
    
    R = _compute_autocorr(x)
    current_metric = _compute_c3_bound(R)
    best_metric = current_metric
    best_x = x.copy()
    best_R = R.copy()
    
    rng = np.random.default_rng(42)
    
    # Simulated Annealing parameters
    T_start = 5.0
    T_end = 0.01
    # Number of steps is dynamic based on time
    # Let's run until deadline
    
    # We can also try multiple restarts if time permits
    # But first, let's see if one long SA run is better.
    
    # To speed up, we can batch flips? No, SA is sequential.
    # We can try to reduce the number of iterations if it's too slow, but N=400 is small.
    
    steps = 0
    last_improve_time = time.time()
    no_improve_limit = 5.0 # seconds
    
    while time.time() < deadline:
        # Cooling schedule
        # Let's use a time-based cooling or step-based.
        # Step-based is easier to tune.
        # Let's assume we can do ~50,000 steps in 100s?
        # Let's just run and see.
        
        # Pick random position
        pos = int(rng.integers(0, N))
        old_val = x[pos]
        x[pos] = -old_val
        
        # Update R
        R_new = _fast_flip_autocorr_update(best_R, x, pos)
        # Wait, we should update from current R, not best_R, for SA to work correctly on the current state.
        # If we always update from best_R, we are doing a form of hill climbing with noise, but the "current" state is not tracked properly for the energy difference calculation.
        # Actually, if we revert, we need to revert R too.
        # Better: maintain current_R.
        
        # Let's restructure to maintain current_R.
        # But _fast_flip_autocorr_update takes R and x.
        # If we flip x, we get R_new.
        # If we reject, we flip x back. Do we need to update R back?
        # Yes, or we can just copy R_old.
        # Copying R (400 floats) is fast.
        
        # Let's stick to:
        # 1. Save R_old
        # 2. Flip x
        # 3. Compute R_new from R_old
        # 4. Evaluate
        # 5. If reject, revert x and set R = R_old (copy)
        
        # This is safer.
        
        # Optimization: Instead of copying R every time, we can compute delta and apply, then revert delta.
        # But the vectorized update creates a new array.
        
        # Let's try to be efficient.
        # current_R is the R corresponding to x.
        # We flip x.
        # R_candidate = _fast_flip_autocorr_update(current_R, x, pos)
        # metric_candidate = _compute_c3_bound(R_candidate)
        
        # If accept:
        #    current_R = R_candidate
        #    if metric_candidate < best_metric: update best
        
        # If reject:
        #    x[pos] = old_val
        #    # current_R remains unchanged (it corresponds to the reverted x)
        
        # This works.
        
        # Let's implement this logic.
        
        # We need to re-initialize the loop variables properly.
        # I will refactor the loop below.
        
        pass # Placeholder to indicate the logic change
    
    # Refactored Loop
    current_R = R.copy()
    T = T_start
    # Estimate steps per second to adjust T?
    # Let's just run.
    
    while time.time() < deadline:
        pos = int(rng.integers(0, N))
        old_val = x[pos]
        x[pos] = -old_val
        
        R_candidate = _fast_flip_autocorr_update(current_R, x, pos)
        metric_candidate = _compute_c3_bound(R_candidate)
        
        delta_E = metric_candidate - current_metric
        
        # Metropolis acceptance
        accept = False
        if delta_E < 0:
            accept = True
        else:
            # Accept with probability exp(-delta_E / T)
            # If T is very small, exp(-large) is 0.
            if T > 0:
                prob = math.exp(-delta_E / T)
                if rng.random() < prob:
                    accept = True
        
        if accept:
            current_R = R_candidate
            current_metric = metric_candidate
            if current_metric < best_metric:
                best_metric = current_metric
                best_x = x.copy()
                best_R = current_R.copy()
                last_improve_time = time.time()
        else:
            # Revert
            x[pos] = old_val
            # current_R is unchanged
        
        # Cooling
        # Simple exponential cooling
        # We need to know how many steps we've done to cool appropriately.
        # Or time-based cooling.
        # Let's use time-based.
        # T = T_start * (T_end/T_start) ** (elapsed_time / total_time)
        # But we don't know total_time.
        # Let's just decrease T slowly.
        T *= 0.99995 # Very slow cooling
        
        # If stuck for too long, reheat
        if time.time() - last_improve_time > no_improve_limit:
            T = T_start * 0.5
            # Optional: random restart
            # x = rng.choice([-1, 1], size=N)
            # current_R = _compute_autocorr(x)
            # current_metric = _compute_c3_bound(current_R)
            # if current_metric < best_metric: ...
            
    return best_x

# Wait, the logic in the first while loop was a placeholder. I need to replace the entire construct function body with the refined logic.

def construct():
    """Construct a signed step function (sequence of ±1) to minimize C3 upper bound."""
    deadline = time.time() + 115  # Safety margin
    
    # Initialize with m-sequence
    mseq = _generate_m_sequence(9)
    x = mseq[:N].copy()
    
    current_R = _compute_autocorr(x)
    current_metric = _compute_c3_bound(current_R)
    best_metric = current_metric
    best_x = x.copy()
    best_R = current_R.copy()
    
    rng = np.random.default_rng(42)
    
    # Simulated Annealing parameters
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9999
    
    last_improve_time = time.time()
    no_improve_limit = 5.0 # seconds
    
    while time.time() < deadline:
        pos = int(rng.integers(0, N))
        old_val = x[pos]
        x[pos] = -old_val
        
        R_candidate = _fast_flip_autocorr_update(current_R, x, pos)
        metric_candidate = _compute_c3_bound(R_candidate)
        
        delta_E = metric_candidate - current_metric
        
        accept = False
        if delta_E < 0:
            accept = True
        else:
            if T > T_min:
                # Avoid overflow/underflow issues with exp
                # delta_E is usually small
                prob = math.exp(-delta_E / T)
                if rng.random() < prob:
                    accept = True
        
        if accept:
            current_R = R_candidate
            current_metric = metric_candidate
            if current_metric < best_metric:
                best_metric = current_metric
                best_x = x.copy()
                best_R = current_R.copy()
                last_improve_time = time.time()
        else:
            # Revert
            x[pos] = old_val
        
        # Cooling
        T *= cooling_rate
        
        # Reheating strategy
        if time.time() - last_improve_time > no_improve_limit:
            T = max(T, 2.0)
            # Try a few random restarts if stuck badly?
            # For now, just reheat.
            
    return best_x
