import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150 s kill

def _score_from_set(u):
    """
    Compute the score for a set of integers u.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    u_arr = np.array(u, dtype=np.int64)
    n = len(u_arr)
    
    # Compute all pairwise sums and differences
    sums = u_arr[:, None] + u_arr[None, :]
    diffs = u_arr[:, None] - u_arr[None, :]
    
    unique_sums = np.unique(sums)
    unique_diffs = np.unique(diffs)
    
    if len(unique_sums) == 0:
        return 0.0
        
    ratio = len(unique_diffs) / float(len(unique_sums))
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _compute_set_sizes_fft(indicator, L):
    """
    Compute |U+U| and |U-U| using FFT on the indicator vector.
    indicator: numpy array of 0s and 1s (or floats in [0,1]) of length L
    L: length of the vector (assumed to cover range [0, 2*max_u] for sums and diffs)
    Returns: count_sums, count_diffs
    """
    # Zero-pad to 2*L to avoid circular convolution issues
    # Sums range from 0 to 2*max_u. If max_u ~ L/2, sums go up to L.
    # Diffs range from -max_u to max_u.
    
    # For sums: Convolution of U with U
    # For diffs: Cross-correlation of U with U (or convolution with reversed U)
    
    # We need a size that fits the max sum.
    # If the set U is contained in [0, M], sums are in [0, 2M].
    # Let's assume the indicator vector length L is at least 2*M + 1.
    
    # Perform FFT
    N_fft = 1
    while N_fft < 2 * L:
        N_fft *= 2
        
    fft_u = np.fft.fft(indicator, N_fft)
    
    # Sums: |U+U| is the number of non-zero entries in U * U
    # This is the convolution
    conv = np.fft.ifft(fft_u * fft_u).real
    # The convolution result has length N_fft, but valid range is [0, 2*L-1]
    # We only care about the part that corresponds to valid sums
    # Since input is non-negative, sums are non-negative.
    count_sums = np.sum(conv[:2*L] > 1e-9)
    
    # Diffs: |U-U| is the number of non-zero entries in U * reverse(U)
    # This is the cross-correlation
    # corr[k] = sum_i u[i] u[i+k]
    # Using FFT: ifft(fft(u) * conj(fft(u))) gives circular correlation.
    # To get linear correlation, we need to handle the negative lags.
    # Alternatively, construct a vector that represents the difference set directly?
    # No, standard trick:
    # Let v = [u, 0, 0, ..., 0] (length L)
    # Let w = [0, ..., 0, u_reversed] 
    # Actually, simpler:
    # The difference set size is the support of the autocorrelation function.
    # Autocorrelation R[k] = sum_i u[i] u[i+k].
    # The number of distinct differences is the number of k where R[k] != 0.
    # Note: R[k] for k>0 corresponds to diffs u[i+k]-u[i] = k.
    # R[k] for k<0 corresponds to diffs u[i]-u[i+k] = -k.
    # Since U is a set, if d is in U-U, then -d is in U-U.
    # So we just need the support of the autocorrelation.
    
    # Compute autocorrelation via FFT
    # If we zero pad u to size N_fft, ifft(fft(u)*conj(fft(u))) gives circular autocorrelation.
    # The linear autocorrelation is embedded in this.
    # Indices 0..L-1 correspond to lags 0..L-1.
    # Indices N_fft-L+1 .. N_fft-1 correspond to lags -(L-1) .. -1.
    
    auto = np.fft.ifft(fft_u * np.conj(fft_u)).real
    
    # The valid lags are - (L-1) to L-1.
    # In the circular result of length N_fft:
    # Lag 0 is at index 0.
    # Lag k (k>0) is at index k.
    # Lag -k (k>0) is at index N_fft - k.
    
    # We sum up all non-zero entries in these positions.
    # To avoid double counting or missing, we just check all positions that correspond to |lag| < L.
    # Since N_fft >= 2*L, the indices 0..L-1 and N_fft-L+1..N_fft-1 are distinct and cover all linear lags.
    
    # Create a mask for valid lags
    mask = np.zeros(N_fft, dtype=bool)
    mask[0:L] = True
    mask[N_fft-L+1:N_fft] = True
    
    count_diffs = np.sum(auto[mask] > 1e-9)
    
    return int(count_sums), int(count_diffs)

def _gradient_descent_optimize(initial_u, L, num_steps=200, lr=0.01, temperature=1.0):
    """
    Optimize the set using gradient descent on a continuous relaxation.
    initial_u: list of integers (initial guess)
    L: size of the grid (max value + 1)
    """
    # Initialize indicator vector
    ind = np.zeros(L)
    for x in initial_u:
        if x < L:
            ind[x] = 1.0
    
    # We optimize the log-odds z such that ind = sigmoid(z)
    # ind = 1 / (1 + exp(-z))
    # z = log(ind / (1 - ind))
    # To avoid log(0), we clamp ind
    
    # Initialize z from ind
    eps = 1e-6
    ind_clamped = np.clip(ind, eps, 1 - eps)
    z = np.log(ind_clamped / (1 - ind_clamped))
    
    best_score = -np.inf
    best_ind = ind.copy()
    
    for step in range(num_steps):
        if time.time() > _DEADLINE - 10.0:
            break
            
        # Compute current sizes using FFT on the continuous ind?
        # FFT works on linear operators, so it works on continuous ind too.
        # The "count" becomes the L1 norm or just sum of indicators > threshold?
        # The task asks for exact integer set sizes.
        # However, for gradient descent, we need a differentiable proxy.
        # Proxy for |U+U|: Sum of (conv > 0). This is non-differentiable.
        # Smooth proxy: Sum of sigmoid(alpha * conv) or just Sum(conv) / max(conv)?
        # Actually, the ratio |U-U|/|U+U| is what we want to maximize.
        # Let S = sum(conv > 0) and D = sum(auto > 0).
        # Let's use a soft count: C_soft = sum(sigmoid(beta * value)).
        # As beta -> inf, this approaches the hard count.
        
        beta = 10.0 # Sharpness of the soft count
        
        # Compute conv and auto
        N_fft = 1
        while N_fft < 2 * L:
            N_fft *= 2
        fft_u = np.fft.fft(ind, N_fft)
        conv = np.fft.ifft(fft_u * fft_u).real
        auto = np.fft.ifft(fft_u * np.conj(fft_u)).real
        
        # Define valid ranges
        mask_sum = np.zeros(N_fft, dtype=bool)
        mask_sum[:2*L] = True
        mask_diff = np.zeros(N_fft, dtype=bool)
        mask_diff[0:L] = True
        mask_diff[N_fft-L+1:N_fft] = True
        
        # Soft counts
        # We want to maximize log(D/S).
        # D_soft = sum(sigmoid(beta * auto[mask_diff]))
        # S_soft = sum(sigmoid(beta * conv[mask_sum]))
        
        # To make it differentiable and stable, we can use the sum of the values themselves scaled,
        # or the log-sum-exp.
        # Let's try: Objective = log(D_soft) - log(S_soft)
        # D_soft = sum(exp(beta * auto) / (1+exp(-beta * auto))) ... sigmoid
        # Actually, simpler: just use the sum of the positive parts? No, that's not a count.
        # Let's use: D_soft = sum(1 / (1 + exp(-beta * auto)))
        
        def soft_count(arr, mask, beta):
            vals = arr[mask]
            # Sigmoid
            return np.sum(1.0 / (1.0 + np.exp(-beta * vals)))
        
        D_soft = soft_count(auto, mask_diff, beta)
        S_soft = soft_count(conv, mask_sum, beta)
        
        if S_soft < 1e-6:
            S_soft = 1e-6
            
        # Score proxy: log(D_soft / S_soft) / log(2*max_u + 1)
        # max_u is the largest index where ind > 0.5
        active_indices = np.where(ind > 0.5)[0]
        if len(active_indices) == 0:
            max_u = 0
        else:
            max_u = active_indices[-1]
            
        if max_u < 1:
            max_u = 1
            
        denom = math.log(2 * max_u + 1)
        obj = math.log(D_soft / S_soft) / denom
        
        if obj > best_score:
            best_score = obj
            best_ind = ind.copy()
            
        # Gradient of obj w.r.t z
        # obj = log(D) - log(S) - log(denom)
        # d obj / d ind = (1/D) * dD/dind - (1/S) * dS/dind
        # dD/dind = sum over lags k of (d auto_k / d ind_i) * sigmoid'(beta * auto_k)
        # This is expensive to compute analytically.
        # Finite difference or backprop through FFT?
        # Backprop through FFT is possible but complex.
        # Let's use a simpler heuristic: 
        # The gradient of the sum of sigmoid(beta * conv) w.r.t ind is related to the convolution of the derivative.
        # d/dind_i [ sum_j sigmoid(beta * conv_j) ] = sum_j sigmoid'(beta * conv_j) * dconv_j/dind_i
        # conv_j = sum_k ind_k ind_{j-k}
        # dconv_j/dind_i = ind_{j-i} + ind_{i-j} (if i!=j) or 2*ind_i (if i==j)
        # This is O(L^2). With L=1000, L^2=1e6. Num steps 200. 2e8 ops. Might be too slow in Python loops.
        # We can use FFT for the gradient too?
        # Let g_conv = sigmoid'(beta * conv)
        # grad_ind = g_conv * ind + ind * g_conv (reversed)?
        # Actually, grad of sum(g(conv)) w.r.t ind is (g * ind) * (ind * g) (convolution)?
        # No, chain rule: 
        # Let F(ind) = sum_j h(conv_j)
        # dF/dind_i = sum_j h'(conv_j) * (ind_{j-i} + ind_{i-j})
        # This is a convolution of h'(conv) with ind?
        # Let A = h'(conv). Then grad_i = sum_j A_j (ind_{j-i} + ind_{i-j})
        # = (A * ind)_i + (A * ind_reversed)_i
        # Yes! We can compute the gradient using FFT.
        
        # 1. Compute h'(conv) and h'(auto)
        # h(x) = sigmoid(beta * x)
        # h'(x) = beta * sigmoid(beta * x) * (1 - sigmoid(beta * x))
        
        sig_conv = 1.0 / (1.0 + np.exp(-beta * conv))
        h_prime_conv = beta * sig_conv * (1 - sig_conv)
        
        sig_auto = 1.0 / (1.0 + np.exp(-beta * auto))
        h_prime_auto = beta * sig_auto * (1 - sig_auto)
        
        # Mask out invalid regions for gradient? 
        # The soft counts only sum over valid regions.
        # So h_prime should be 0 outside valid regions.
        h_prime_conv[~mask_sum] = 0
        h_prime_auto[~mask_diff] = 0
        
        # Compute gradients
        # grad_S = ifft(fft(h_prime_conv) * fft(ind)) + ifft(fft(h_prime_conv) * fft(ind_reversed))
        # ind_reversed is ind[::-1]
        
        fft_ind = np.fft.fft(ind, N_fft)
        fft_ind_rev = np.fft.fft(ind[::-1], N_fft)
        
        fft_hp_conv = np.fft.fft(h_prime_conv, N_fft)
        fft_hp_auto = np.fft.fft(h_prime_auto, N_fft)
        
        # Convolution of h_prime_conv with ind
        grad_conv_part1 = np.fft.ifft(fft_hp_conv * fft_ind).real
        grad_conv_part2 = np.fft.ifft(fft_hp_conv * fft_ind_rev).real
        grad_S = grad_conv_part1 + grad_conv_part2
        
        grad_auto_part1 = np.fft.ifft(fft_hp_auto * fft_ind).real
        grad_auto_part2 = np.fft.ifft(fft_hp_auto * fft_ind_rev).real
        grad_D = grad_auto_part1 + grad_auto_part2
        
        # d obj / d ind = (1/S) * grad_D - (1/D) * grad_S
        # Wait, obj = log(D) - log(S)
        # d obj / d ind = (1/D) * grad_D - (1/S) * grad_S
        
        grad_obj = (1.0 / D_soft) * grad_D - (1.0 / S_soft) * grad_S
        
        # Now, d ind / d z = ind * (1 - ind)
        grad_z = grad_obj * ind * (1 - ind)
        
        # Update z
        z += lr * grad_z
        
        # Update ind
        ind = 1.0 / (1.0 + np.exp(-z))
        
        # Periodically project back to a hard set to evaluate true score
        if step % 10 == 0:
            hard_ind = (ind > 0.5).astype(int)
            # Ensure 0 is in the set
            hard_ind[0] = 1
            # Extract set
            u_hard = np.where(hard_ind > 0)[0].tolist()
            if len(u_hard) >= 2:
                true_score = _score_from_set(u_hard)
                if true_score > best_score:
                    best_score = true_score
                    best_ind = hard_ind.copy()
    
    # Return the best hard set found
    final_ind = (best_ind > 0.5).astype(int)
    final_ind[0] = 1
    u_final = np.where(final_ind > 0)[0].tolist()
    
    if len(u_final) < 2:
        return [0, 1]
        
    return u_final

def construct():
    global _DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    best_u = [0, 1]
    best_s = _score_from_set(best_u)
    
    # Strategy:
    # 1. Generate a few diverse initial sets (Golomb rulers, geometric).
    # 2. Run gradient descent on a continuous relaxation for each.
    # 3. Keep the best result.
    
    # Determine grid size L
    # The paper mentions sizes 2003 and 54265.
    # If max U is around 5000, L=10000.
    # FFT of size 10000 is fast.
    # Let's try L=2000 to start, then maybe larger if time permits.
    # Actually, the score depends on log(2*max U + 1).
    # Smaller max U is better for the denominator, unless the ratio drops significantly.
    # Let's try a few L values.
    
    L_values = [1000, 2000, 5000]
    
    for L in L_values:
        if time.time() > _DEADLINE - 30.0:
            break
            
        # Generate initial sets
        initial_sets = []
        
        # 1. Golomb ruler
        u_golomb = [0]
        for i in range(1, 20):
            # Greedy Golomb
            found = False
            for next_val in range(u_golomb[-1] + 1, L):
                # Check if adding next_val creates duplicate diffs
                ok = True
                for j in range(len(u_golomb)):
                    d = u_golomb[j] - next_val
                    # Check if d is already a diff
                    # This is O(n^2) check, but n is small
                    # We can just check against existing diffs
                    pass
                # Simplified: just add random small increments
                # Actually, let's just use a known Golomb ruler or random start
                break
            # Just use a random set for simplicity in init
            initial_sets.append(sorted(random.sample(range(L), 10)))
            
        # 2. Geometric
        u_geo = [0]
        curr = 1
        for i in range(1, 15):
            curr = int(curr * 1.5)
            if curr >= L: break
            u_geo.append(curr)
        initial_sets.append(u_geo)
        
        # 3. Random
        initial_sets.append(sorted(random.sample(range(L), 15)))
        
        for init_u in initial_sets:
            if time.time() > _DEADLINE - 10.0:
                break
            
            # Ensure 0 is in init_u
            if 0 not in init_u:
                init_u.append(0)
            init_u = sorted(set(init_u))
            
            # Run gradient descent
            u_opt = _gradient_descent_optimize(init_u, L, num_steps=100, lr=0.05, temperature=1.0)
            
            s_opt = _score_from_set(u_opt)
            
            if s_opt > best_s:
                best_s = s_opt
                best_u = u_opt[:]
                
    # Final polish with local search on the best continuous result
    if time.time() < _DEADLINE - 5.0:
        # Simple local search
        u_curr = best_u[:]
        s_curr = best_s
        for _ in range(50):
            if time.time() > _DEADLINE - 2.0:
                break
            idx = random.randint(1, len(u_curr) - 1)
            old_val = u_curr[idx]
            new_val = old_val + random.randint(-5, 5)
            if new_val < 0 or new_val in u_curr:
                continue
            u_curr[idx] = new_val
            s_new = _score_from_set(u_curr)
            if s_new > s_curr:
                s_curr = s_new
                best_u = u_curr[:]
                best_s = s_curr
            else:
                u_curr[idx] = old_val
                
    # Sanity check
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
    if len(best_u) < 2:
        best_u = [0, 1]
        
    return best_u
# EVOLVE-BLOCK-END
