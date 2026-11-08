import time
import numpy as np

N = 100

def construct():
    """
    Main entry point to find the optimal sequence h.
    Returns a sequence of step heights in [0, 1] whose sum is exactly len/2.
    """
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    # We work with binary sequences {0, 1} as they are likely optimal or near-optimal
    # for this type of combinatorial objective.
    # h has N/2 ones and N/2 zeros.
    
    def compute_corr(h):
        """Compute circular autocorrelation C_k = sum h_i h_{i+k}."""
        # Use FFT for initial computation
        H = np.fft.fft(h)
        corr = np.fft.ifft(H * np.conj(H)).real
        # corr[k] is sum h_i h_{i-k} ? 
        # np.fft.ifft(FFT(h) * conj(FFT(h))) gives circular autocorrelation.
        # Specifically, result[k] = sum_i h[i] h[(i-k) % N]
        # We want C_k = sum_i h[i] h[(i+k) % N].
        # Due to symmetry of real autocorrelation, C_k = C_{N-k}.
        # So corr[k] is C_{-k} = C_k.
        return corr

    def get_score(h, C):
        """Calculate max overlap score. Lower is better."""
        # Overlap at shift k is sum h_i (1 - h_{i+k}) = sum h_i - C_k = N/2 - C_k
        # We want to minimize max(N/2 - C_k) for k != 0
        # This is equivalent to maximizing min(C_k) for k != 0
        # But the score returned is the raw max overlap value.
        # max_overlap = max_{k!=0} (N/2 - C_k)
        C_nonzero = C[1:] # Exclude k=0
        max_overlap = np.max(N/2.0 - C_nonzero)
        return max_overlap

    # Initialization
    # Try a few good starting points
    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = float('inf')
    best_C = None
    
    # Pattern 1: Alternating
    h_init = np.zeros(N)
    h_init[::2] = 1.0
    
    # Pattern 2: Block
    h_block = np.zeros(N)
    h_block[:N//2] = 1.0
    
    # Pattern 3: Random
    h_rand = np.zeros(N)
    idx = rng.choice(N, size=N//2, replace=False)
    h_rand[idx] = 1.0
    
    # Pattern 4: Based on Golden Ratio (often good for low correlation)
    h_golden = np.zeros(N)
    golden = (1 + np.sqrt(5)) / 2
    for i in range(N):
        if (i * golden) % 1.0 < 0.5:
            h_golden[i] = 1.0
    # Adjust to exactly N/2 ones
    ones_count = np.sum(h_golden)
    if ones_count > N//2:
        ones_idx = np.where(h_golden == 1.0)[0]
        for i in ones_idx[:int(ones_count - N//2)]:
            h_golden[i] = 0.0
    elif ones_count < N//2:
        zeros_idx = np.where(h_golden == 0.0)[0]
        for i in zeros_idx[:int(N//2 - ones_count)]:
            h_golden[i] = 1.0

    inits = [h_init, h_block, h_rand, h_golden]
    
    # Add a few more random restarts
    for _ in range(10):
        h_r = np.zeros(N)
        idx = rng.choice(N, size=N//2, replace=False)
        h_r[idx] = 1.0
        inits.append(h_r)

    # Local Search / Simulated Annealing
    def local_search(h_init, max_iters=20000, start_temp=1.0, end_temp=0.1):
        h = h_init.copy()
        C = compute_corr(h)
        current_score = get_score(h, C)
        
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            return h, current_score

        best_local_h = h.copy()
        best_local_score = current_score
        
        # Precompute indices for speed? 
        # We will update C vector directly.
        
        for it in range(max_iters):
            if time.time() > deadline - 10: # Keep 10s buffer
                break
                
            # Cooling schedule
            progress = it / max_iters
            temp = start_temp * (end_temp / start_temp) ** progress
            
            # Pick random swap
            i = ones[rng.integers(len(ones))]
            j = zeros[rng.integers(len(zeros))]
            
            if i == j:
                continue
            
            # Calculate Delta C
            # h_i becomes 0, h_j becomes 1
            # C_k = sum h_l h_{l+k}
            
            # Terms removed (from i):
            # l=i: h_i h_{i+k} -> 0. Delta: -h_{i+k}
            # l=i-k: h_{i-k} h_i -> 0. Delta: -h_{i-k}
            
            # Terms added (from j):
            # l=j: 0 h_{j+k} -> h_{j+k}. Delta: +h_{j+k}
            # l=j-k: h_{j-k} 0 -> h_{j-k}. Delta: +h_{j-k}
            
            # Cross terms if i and j are related:
            # If j == i+k:
            #   Old term l=i: h_i h_j = 1*1 = 1.
            #   New term l=i: 0 * 1 = 0. (Already handled by -h_{i+k} where h_{i+k}=h_j=1)
            #   Old term l=j: h_j h_{j+k} = 1 * h_{i+2k}.
            #   New term l=j: 1 * h_{i+2k}. No change?
            #   Wait, the "l=i" term in the sum for C_k is h_i h_{i+k}.
            #   If we change h_i and h_j, we must be careful if the indices overlap.
            
            # Let's just compute the delta using the formula:
            # Delta C_k = -h_{i+k} - h_{i-k} + h_{j+k} + h_{j-k}
            # This formula assumes i and j are distinct and don't interact in the sum for a specific k?
            # Actually, the term h_l h_{l+k} involves pairs (l, l+k).
            # Changing h_i affects pairs (i, i+k) and (i-k, i).
            # Changing h_j affects pairs (j, j+k) and (j-k, j).
            # If i+k == j, then pair (i, j) is affected by both changes.
            # Old: h_i h_j = 1*1 = 1.
            # New: h'_i h'_j = 0*1 = 0.
            # My formula:
            # From i: -h_{i+k} = -h_j = -1.
            # From j: +h_{j-k} = +h_i = +1.
            # Sum: 0. Correct.
            
            # If i+k == j-k (i.e., j = i+2k)?
            # Pair (i, i+2k) and (i, i+k)? No.
            # The pairs are (l, l+k).
            # If i = j+k, then pair (j, i) is affected.
            # Old: h_j h_i = 1.
            # New: 1 * 0 = 0.
            # Formula:
            # From i (now 0): -h_{i-k} = -h_j = -1.
            # From j (now 1): +h_{j+k} = +h_i = +1.
            # Sum: 0. Correct.
            
            # So the formula Delta C_k = -h_{i+k} - h_{i-k} + h_{j+k} + h_{j-k} holds for all k.
            
            idx_k = np.arange(N)
            # Use np.roll for circular shifts
            # h_{i+k} is h[(i+k)%N]
            # We need to compute this for all k in 0..N-1.
            # This is equivalent to shifting the vector h.
            
            # Vectorized calculation of delta for all k:
            # delta[k] = -h[(i+k)%N] - h[(i-k)%N] + h[(j+k)%N] + h[(j-k)%N]
            
            # To do this efficiently in numpy:
            # We can create an array of indices.
            # But simpler: 
            # h_i_plus_k = np.roll(h, -i) # h[k-i]? No.
            # np.roll(h, -k) gives h[(i-k)%N] at index i?
            # Let's just use indexing. N=100 is small.
            
            # delta_C = np.zeros(N)
            # for k in range(N):
            #    delta_C[k] = -h[(i+k)%N] - h[(i-k)%N] + h[(j+k)%N] + h[(j-k)%N]
            
            # Vectorized version:
            # Create indices for i+k, i-k, j+k, j-k
            # k = 0..N-1
            # i+k mod N, i-k mod N, etc.
            
            # Since N is small, the loop is fast enough if we do many iterations.
            # But let's try vectorization.
            
            # h_shifted_i_plus = np.roll(h, -i) # This shifts h left by i. h[k] becomes h[k+i].
            # We want h[(i+k)%N]. This is h_shifted_i_plus[k].
            # h[(i-k)%N] is h_shifted_i_minus[k] where h_shifted_i_minus = np.roll(h, i).
            
            h_ip = np.roll(h, -i)
            h_im = np.roll(h, i)
            h_jp = np.roll(h, -j)
            h_jm = np.roll(h, j)
            
            delta_C = -h_ip - h_im + h_jp + h_jm
            
            new_C = C + delta_C
            new_score = get_score(h, new_C)
            
            # Acceptance criterion
            if new_score < current_score:
                # Always accept improvement
                h[i] = 0.0
                h[j] = 1.0
                C = new_C
                current_score = new_score
                
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local_h = h.copy()
            else:
                # Accept with probability exp(-delta_score / temp)
                # Note: score is "max overlap", lower is better.
                # delta_score = new_score - current_score > 0
                delta = new_score - current_score
                if delta < 0:
                    accept = True
                else:
                    # If temp is small, rarely accept
                    accept = rng.random() < np.exp(-delta / temp)
                
                if accept:
                    h[i] = 0.0
                    h[j] = 1.0
                    C = new_C
                    current_score = new_score
                    if current_score < best_local_score:
                        best_local_score = current_score
                        best_local_h = h.copy()
            
            # Update ones and zeros sets
            # Instead of rebuilding, just swap in the lists?
            # Since we only need random indices, and N is small, rebuilding is O(N).
            # Or we can just keep the arrays and filter?
            # Rebuilding ones/zeros every iteration is O(N).
            # With 20k iterations, 20k * 100 = 2M ops. Fine.
            ones = np.where(h == 1.0)[0]
            zeros = np.where(h == 0.0)[0]
            
        return best_local_h, best_local_score

    # Run local search from each initialization
    for h_init in inits:
        if time.time() > deadline - 20:
            break
            
        h_opt, score_opt = local_search(h_init, max_iters=10000, start_temp=0.5, end_temp=0.01)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
    # If we have time, do a final polish on the best solution found
    if best_h is not None and time.time() < deadline - 5:
         h_opt, score_opt = local_search(best_h, max_iters=20000, start_temp=0.1, end_temp=0.001)
         if score_opt < best_score:
             best_score = score_opt
             best_h = h_opt.copy()

    # Ensure sum is exactly N/2
    if best_h is None:
        best_h = np.full(N, 0.5)
    else:
        # best_h is binary, sum is N/2 by construction of swaps
        # Just to be safe, verify
        s = np.sum(best_h)
        if s != N/2:
             # Project to sum N/2
             # This shouldn't happen with swaps
             best_h = np.clip(best_h, 0, 1)
             diff = N/2 - np.sum(best_h)
             best_h += diff / N
             best_h = np.clip(best_h, 0, 1)

    return best_h
