"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    # Safety margin: finish before 120s, aim for 110s max compute
    deadline = start_time + 110.0
    
    # 1. Generate Thue-Morse sequence as initial guess
    # Thue-Morse has low autocorrelation sidelobes
    def thue_morse(n):
        t = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            t[i] = t[i >> 1] ^ (i & 1)
        return 2 * t - 1  # Map 0,1 to 1,-1
        
    s = thue_morse(N)
    
    # 2. Compute Autocorrelation using FFT
    def calc_autocorr(s):
        # Zero pad to 2N to avoid circular convolution artifacts
        f = np.fft.fft(s, 2*N)
        r = np.fft.ifft(f * np.conj(f))
        return r[:N].real
        
    R = calc_autocorr(s)
    # We are interested in sidelobes k=1..N-1
    # R[0] is the main peak (N for +/-1 sequences)
    
    # Metric: Max absolute value of sidelobes
    # For seed (all ones), R[k] = N - k. Max is N-1 = 399.
    # We want to minimize this value.
    
    current_max = np.max(np.abs(R[1:]))
    best_s = s.copy()
    best_R = R.copy()
    best_max = current_max
    
    # 3. Hill Climbing Optimization
    # Strategy: Try flipping each bit. If it reduces the global max sidelobe, accept.
    # To speed up, we compute the delta R for a flip in O(N).
    
    # Pre-allocate delta array
    delta = np.zeros(N)
    
    iterations = 0
    stagnant_count = 0
    
    while time.time() < deadline:
        iterations += 1
        
        # Find the index of the current max sidelobe
        # This is the "bottleneck" we try to reduce
        k_max = np.argmax(np.abs(R[1:])) + 1
        
        improved = False
        best_candidate_j = -1
        best_candidate_max = best_max
        best_candidate_R = None
        
        # Try flipping each bit j
        # We can vectorize the "check" part partially, but the dependency on s[j] makes it tricky.
        # However, N=400 is small.
        
        for j in range(N):
            val = s[j]
            
            # Compute Delta R for flipping s[j]
            # R[k] = sum_i s[i] s[i+k]
            # Flipping s[j] changes:
            # 1. Term where i=j: s[j] s[j+k] -> -s[j] s[j+k]. Change: -2 s[j] s[j+k]
            # 2. Term where i+k=j => i=j-k: s[j-k] s[j] -> s[j-k] (-s[j]). Change: -2 s[j] s[j-k]
            
            # We need to compute D[k] for all k.
            # D[k] = -2 * val * ( (s[j+k] if j+k < N else 0) + (s[j-k] if j-k >= 0 else 0) )
            
            # Efficiently compute D
            # Term 1: k in 0..N-1-j
            # Term 2: k in 0..j
            
            # Reset delta to 0
            delta[:] = 0
            
            # Term 1
            if N - 1 - j >= 0:
                delta[0:N-j] -= 2 * val * s[j:N]
            
            # Term 2
            if j >= 0:
                delta[0:j+1] -= 2 * val * s[j::-1]
            
            # New R
            R_new = R + delta
            
            # Check new max sidelobe
            # We only care about k=1..N-1
            new_max = np.max(np.abs(R_new[1:]))
            
            if new_max < best_candidate_max:
                best_candidate_max = new_max
                best_candidate_j = j
                best_candidate_R = R_new.copy()
                
                # If we found a value lower than the global best, we can potentially accept it
                # But we should finish the loop to see if there's an even better flip?
                # Or accept immediately?
                # Accepting immediately is faster (greedy).
                # Let's accept the first improvement found to keep it simple and fast.
                # Actually, finding the *best* flip among all j might be better for escaping local minima.
                # But O(N^2) per iteration is fine.
                
        if best_candidate_j != -1:
            # Apply the best flip
            s[best_candidate_j] = -s[best_candidate_j]
            R = best_candidate_R
            best_max = best_candidate_max
            best_s = s.copy()
            best_R = R.copy()
            improved = True
            stagnant_count = 0
        else:
            stagnant_count += 1
            # If stuck, try a random perturbation or restart
            if stagnant_count > 50:
                # Random restart: flip a few random bits
                for _ in range(5):
                    j = np.random.randint(N)
                    s[j] = -s[j]
                R = calc_autocorr(s)
                current_max = np.max(np.abs(R[1:]))
                if current_max < best_max:
                    best_max = current_max
                    best_s = s.copy()
                    best_R = R.copy()
                stagnant_count = 0
                
    return best_s
# EVOLVE-BLOCK-END
