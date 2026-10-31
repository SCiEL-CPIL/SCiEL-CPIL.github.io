import numpy as np

def construct():
    # We can increase N to get a better approximation of the continuous problem.
    # The seed used N=100. Let's try N=200 or N=400.
    # However, larger N makes the search space harder.
    # Let's start with N=200.
    N = 200
    
    # Initial binary sequence with N/2 ones
    h = np.zeros(N)
    h[:N//2] = 1.0
    
    # Shuffle to break symmetry
    np.random.shuffle(h)
    
    def calc_max_c(h):
        """Calculate max C_k for a given h"""
        max_c = 0.0
        # We only need to check k=0 to N-1
        # Optimization: r_k for binary h is integer count of overlaps.
        for k in range(N):
            if k == 0:
                r = np.sum(h) # N/2
            else:
                # Efficient correlation for binary? 
                # np.sum(h[k:] * h[:-k])
                r = np.sum(h[k:] * h[:-k])
            c_val = (2.0 / N) * (N / 2.0 - r)
            if c_val > max_c:
                max_c = c_val
        return max_c

    # Simulated Annealing
    current_h = h.copy()
    current_cost = calc_max_c(current_h)
    
    best_h = current_h.copy()
    best_cost = current_cost
    
    T = 1.0
    cooling_rate = 0.999
    
    # Number of iterations
    # Must finish in 120s. 
    # N=200, loop over N shifts in calc_max_c is O(N^2). 
    # 10000 iterations * 40000 ops = 4e8 ops. Might be slow in pure Python loops.
    # We can optimize calc_max_c using FFT?
    # Correlation via FFT is O(N log N).
    
    def calc_max_c_fast(h):
        # Use FFT to compute autocorrelation
        # h is real.
        H = np.fft.fft(h, n=2*N)
        corr = np.fft.ifft(H * np.conj(H)).real
        # corr[k] corresponds to circular correlation?
        # np.fft.ifft(fft(h)*conj(fft(h))) gives circular autocorrelation.
        # We need linear autocorrelation?
        # The seed uses linear (zero-padded effectively by slicing).
        # Circular correlation r_circ[k] = sum_i h_i h_{(i+k)%N}.
        # Linear correlation r_lin[k] = sum_{i=0}^{N-1-k} h_{i+k} h_i.
        # For binary sequences, circular and linear differ by the wrap-around terms.
        # If N is large, they are similar.
        # Let's stick to linear to match the seed's metric exactly.
        # But O(N^2) is risky.
        # Let's use a faster linear correlation or just reduce iterations.
        
        # Alternative: Use the fact that we only change 2 elements.
        # Delta correlation can be computed in O(N).
        
        max_c = 0.0
        for k in range(N):
            if k == 0:
                r = N / 2.0
            else:
                r = np.dot(h[k:], h[:-k])
            c_val = (2.0 / N) * (N / 2.0 - r)
            if c_val > max_c:
                max_c = c_val
        return max_c

    # To speed up, we can use a delta update.
    # But for simplicity and correctness, let's limit iterations and use N=100 or 120.
    # The seed got 1.22 with N=100 gradient.
    # Binary search with N=100 might be faster per step.
    
    N = 100
    current_h = np.zeros(N)
    current_h[:N//2] = 1.0
    np.random.shuffle(current_h)
    
    current_cost = calc_max_c_fast(current_h)
    best_h = current_h.copy()
    best_cost = current_cost
    
    T = 0.5
    cooling_rate = 0.9995
    max_iter = 20000
    
    for i in range(max_iter):
        # Pick a random 1 and a random 0 to swap
        ones = np.where(current_h == 1.0)[0]
        zeros = np.where(current_h == 0.0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        idx_one = np.random.choice(ones)
        idx_zero = np.random.choice(zeros)
        
        # Apply swap
        old_one_val = current_h[idx_one]
        old_zero_val = current_h[idx_zero]
        
        current_h[idx_one] = old_zero_val
        current_h[idx_zero] = old_one_val
        
        new_cost = calc_max_c_fast(current_h)
        
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = current_h.copy()
        else:
            # Revert
            current_h[idx_one] = old_one_val
            current_h[idx_zero] = old_zero_val
            
        T *= cooling_rate
        
        if T < 1e-5:
            break
            
    return best_h
