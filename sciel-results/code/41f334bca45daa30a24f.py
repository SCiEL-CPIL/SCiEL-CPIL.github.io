import time
import numpy as np
from scipy.optimize import minimize

N = 100

def _fft_max_overlap(h):
    """Fast computation of max overlap using FFT."""
    # overlap(s) = sum_i h[i] * (1 - h[(i-s) mod N])
    #            = sum_i h[i] - sum_i h[i]*h[(i-s) mod N]
    #            = N/2 - autocorrelation(s)
    # where autocorrelation(s) = sum_i h[i]*h[(i-s) mod N]
    
    # Note: np.fft.ifft(fft(h) * conj(fft(h))) gives circular autocorrelation
    # corr[k] = sum_i h[i] * h[(i+k) mod N]
    # We need sum_i h[i] * h[(i-s) mod N] = corr[N-s] (or corr[-s])
    
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    
    # autocorr[s] corresponds to shift +s.
    # We want overlap for shift s: sum_i h[i] h[i-s].
    # This is autocorr[-s] = autocorr[N-s].
    
    # Let's verify:
    # h = [1, 0, 0, 0]. N=4.
    # H = [1, 1, 1, 1]. autocorr = [1, 0, 0, 0].
    # s=1: sum h[i] h[i-1] = h[0]h[3] + h[1]h[0] + ... = 0.
    # autocorr[N-1] = autocorr[3] = 0. Correct.
    # s=0: sum h[i] h[i] = 1. autocorr[0] = 1. Correct.
    
    # So C(s) = autocorr[(-s) % N]
    shifts = np.arange(N)
    C = autocorr[(-shifts) % N]
    
    overlaps = N/2.0 - C
    return np.max(overlaps)

def _project(h):
    """Project h to valid space: [0,1] and sum N/2."""
    # Simple projection might not preserve sum exactly if clipping is aggressive.
    # We will use iterative clipping and adjustment.
    h = np.clip(h, 0, 1)
    diff = N/2.0 - np.sum(h)
    h = h + diff / N
    h = np.clip(h, 0, 1)
    # Final tiny correction
    diff = N/2.0 - np.sum(h)
    h = h + diff / N
    return h

def construct():
    deadline = time.time() + 110  # 10s safety margin
    
    # Phase 1: Generate Initial Binary Candidates
    candidates = []
    
    # 1. Thue-Morse
    # Thue-Morse sequence: 0, 1, 1, 0, 1, 0, 0, 1 ...
    # t(n) = parity of popcount(n)
    tm = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=float)
    # Ensure sum is N/2. If not, flip some bits.
    if np.sum(tm) != N/2:
        # Flip bits to match sum
        idx_1 = np.where(tm == 1)[0]
        idx_0 = np.where(tm == 0)[0]
        curr_sum = np.sum(tm)
        target = N/2.0
        if curr_sum > target:
            to_flip = int(curr_sum - target)
            tm[idx_1[:to_flip]] = 0
        else:
            to_flip = int(target - curr_sum)
            tm[idx_0[:to_flip]] = 1
    candidates.append(tm.copy())
    
    # 2. Alternating 0011
    alt = np.array([0 if (i % 4) < 2 else 1 for i in range(N)], dtype=float)
    if np.sum(alt) != N/2:
        # Adjust
        idx_1 = np.where(alt == 1)[0]
        idx_0 = np.where(alt == 0)[0]
        curr_sum = np.sum(alt)
        target = N/2.0
        if curr_sum > target:
            to_flip = int(curr_sum - target)
            alt[idx_1[:to_flip]] = 0
        else:
            to_flip = int(target - curr_sum)
            alt[idx_0[:to_flip]] = 1
    candidates.append(alt.copy())
    
    # 3. Random Binary Sequences
    np.random.seed(42)
    for _ in range(20):
        h_rand = np.zeros(N)
        # Pick N/2 indices to be 1
        ones_idx = np.random.choice(N, N//2, replace=False)
        h_rand[ones_idx] = 1.0
        candidates.append(h_rand)
        
    # 4. "Sine" binary (thresholded sine)
    for freq in [1, 2, 3]:
        h_sine = np.sin(2 * np.pi * freq * np.arange(N) / N)
        # Threshold to binary
        h_bin = (h_sine > 0).astype(float)
        if np.sum(h_bin) != N/2:
            idx_1 = np.where(h_bin == 1)[0]
            idx_0 = np.where(h_bin == 0)[0]
            curr_sum = np.sum(h_bin)
            target = N/2.0
            if curr_sum > target:
                to_flip = int(curr_sum - target)
                h_bin[idx_1[:to_flip]] = 0
            else:
                to_flip = int(target - curr_sum)
                h_bin[idx_0[:to_flip]] = 1
        candidates.append(h_bin)

    best_h = None
    best_val = float('inf')
    
    # Evaluate all candidates
    for h in candidates:
        val = _fft_max_overlap(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    # Phase 2: Randomized Local Search on Binary Domain
    # Time budget: 60 seconds
    search_deadline = time.time() + 60
    
    # Precompute indices for potential flips
    # We will randomly pick pairs (i, j) where h[i]=1 and h[j]=0
    
    while time.time() < search_deadline:
        # Get current indices of 1s and 0s
        idx_1 = np.where(best_h == 1)[0]
        idx_0 = np.where(best_h == 0)[0]
        
        if len(idx_1) == 0 or len(idx_0) == 0:
            break
            
        # Sample a batch of moves
        # Let's try 1000 random moves per iteration
        num_moves = 1000
        improved = False
        
        for _ in range(num_moves):
            i = idx_1[np.random.randint(len(idx_1))]
            j = idx_0[np.random.randint(len(idx_0))]
            
            # Flip
            h_new = best_h.copy()
            h_new[i] = 0
            h_new[j] = 1
            
            val_new = _fft_max_overlap(h_new)
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new
                improved = True
                # Update idx_1, idx_0 immediately for consistency in this iteration?
                # It's safer to just break and re-evaluate idx_1/idx_0 if we want strict local search,
                # but for random search, we can just continue.
                # However, idx_1 and idx_0 are now stale.
                # Let's just break and restart the sampling for this "iteration"
                break
        
        if not improved:
            # No improvement in 1000 random moves. Try a different strategy?
            # Maybe accept slightly worse moves (simulated annealing)?
            # For now, let's just continue. If stuck, the continuous phase might help.
            pass
            
    # Phase 3: Continuous Optimization (SLSQP)
    # Use the best binary solution as start
    h_start = best_h.copy()
    
    # Objective: Minimize max overlap.
    # Surrogate: Log-sum-exp of overlaps.
    # O(s) = N/2 - C(s).
    # We want to minimize max(O(s)).
    # f(h) = (1/beta) * log(sum(exp(beta * O(s))))
    
    beta = 20.0 # Temperature
    
    def objective(h_flat):
        h = np.clip(h_flat, 0, 1)
        # Note: SLSQP might violate sum constraint slightly during iterations,
        # but the constraint will handle it.
        # However, for the objective calculation, we should use the h as is.
        # The constraint will force sum to N/2.
        
        H = np.fft.fft(h)
        autocorr = np.real(np.fft.ifft(H * np.conj(H)))
        C = autocorr[(-np.arange(N)) % N]
        overlaps = N/2.0 - C
        
        # Shift to avoid overflow in exp
        max_ov = np.max(overlaps)
        # If max_ov is large, exp(beta * (ov - max_ov)) is safe.
        # log(sum(exp(beta * ov))) = max_ov * beta + log(sum(exp(beta * (ov - max_ov))))
        # We want to minimize this.
        # Actually, we just need the value.
        # To be stable:
        scaled = beta * (overlaps - max_ov)
        sum_exp = np.sum(np.exp(scaled))
        return max_ov + np.log(sum_exp) / beta
        
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2}]
    bounds = [(0, 1)] * N
    
    # Run SLSQP
    try:
        # maxiter needs to be reasonable.
        res = minimize(objective, h_start, method='SLSQP', bounds=bounds,
                       constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
        
        if res.success:
            h_res = res.x
            # Ensure exact sum
            h_res = _project(h_res)
            val_res = _fft_max_overlap(h_res)
            
            if val_res < best_val:
                best_h = h_res
                best_val = val_res
    except Exception as e:
        print(f"SLSQP failed: {e}")
        
    # Final Check and Return
    best_h = _project(best_h)
    return best_h
