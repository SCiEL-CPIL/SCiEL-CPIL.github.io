import numpy as np
import time

N = 400
# We want to construct a signed step function with 400 intervals.
# The task mentions "signed step function". Usually this implies values in {-1, 1} or similar.
# However, the seed returns np.ones(N).
# The inspirations mention "signed step heights".
# Let's assume the output should be a sequence of 400 values.
# The score is the upper bound on C3 (autocorrelation).
# Lower is better.
# Seed returns np.ones(N).
# Autocorrelation of ones is triangular with peak N and sidelobes decreasing.
# Max sidelobe for ones is N-1 (at lag 1).
# If we use +/- 1, we can potentially reduce sidelobes.

def _autocorr_sidelobe_max(x):
    """Compute the maximum absolute value of the autocorrelation sidelobes."""
    # x is a 1D array of length N
    # Autocorrelation r[k] = sum_{i=0}^{N-1-k} x[i] * x[i+k]
    # We are interested in k=1..N-1 (sidelobes).
    # Note: Depending on definition, it might be normalized or not.
    # "Upper bound on C3" likely refers to the maximum sidelobe value.
    
    # Efficient computation using FFT
    # Zero-pad to length 2*N to avoid circular convolution effects for linear autocorrelation
    n_fft = 2 * N
    X = np.fft.fft(x, n=n_fft)
    R = np.fft.ifft(X * np.conj(X))
    # R[0] is the peak (sum x^2).
    # The linear autocorrelation lags are R[1:N] and R[N+1:2*N] (conjugates).
    # We take the real part.
    # Sidelobes are indices 1 to N-1 in the linear autocorrelation.
    # In the FFT output R, the lags 1..N-1 are at indices 1..N-1.
    # The lags -1..-(N-1) are at indices N+1..2*N-1.
    # Due to symmetry for real x, we only need 1..N-1.
    
    sidelobes = np.real(R[1:N])
    return np.max(np.abs(sidelobes))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initialize with random phases in frequency domain
    # We want to minimize max sidelobe of autocorrelation.
    # This is equivalent to minimizing the peak sidelobe level (PSL) of the aperiodic autocorrelation.
    # This is a classic problem (Golay sequences, Barker sequences, etc.).
    
    # Strategy:
    # 1. Start with a random sequence of +/- 1.
    # 2. Compute its FFT.
    # 3. Keep the magnitude, randomize the phase? No, that's for constant modulus.
    #    Here we have a fixed magnitude spectrum? No, the magnitude spectrum is determined by the sequence.
    
    # Better Strategy: Phase Retrieval / Spectral Shaping
    # Let X_k be the DFT of the sequence x_n.
    # Autocorrelation is related to |X_k|^2.
    # Specifically, the aperiodic autocorrelation r_m is the inverse DFT of |X_k|^2 (with zero padding).
    # To minimize the peak of r_m (m!=0), we want to shape |X_k|^2 to be as flat as possible.
    # However, x_n must be real (and likely +/- 1 or real values).
    # If we allow real values, we can use continuous optimization.
    # The task says "signed step function". This often implies piecewise constant, i.e., a sequence of values.
    # "Signed" usually implies +/- 1. Let's assume the values should be in {-1, 1} or we can optimize real values and then binarize?
    # The seed returns np.ones(N) which are 1s.
    # If I return a sequence of 1s and -1s, the autocorrelation sidelobes can be much lower.
    
    # Let's try to find a sequence of +/- 1 with low PSL.
    # This is a combinatorial optimization problem.
    # We can use simulated annealing or local search.
    
    # Local Search:
    # 1. Start with random +/- 1 sequence.
    # 2. Calculate PSL.
    # 3. Try flipping a single bit. If PSL decreases, accept.
    # 4. Or use a more sophisticated move.
    
    # Given the time limit, a simple local search might be slow if we recalculate full autocorrelation each time.
    # But N=400 is small. FFT is O(N log N). 400 is tiny.
    # We can do many iterations.
    
    best_seq = None
    best_score = float('inf')
    
    # Initial random sequence
    x = np.random.choice([1, -1], size=N)
    
    # Local search with restarts
    max_iters = 50000
    
    while time.time() < deadline:
        # Perform local search from current x
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            # Calculate current score
            current_score = _autocorr_sidelobe_max(x)
            if current_score < best_score:
                best_score = current_score
                best_seq = x.copy()
                
            # Try flipping a random bit
            idx = np.random.randint(0, N)
            old_val = x[idx]
            x[idx] = -old_val
            
            new_score = _autocorr_sidelobe_max(x)
            
            if new_score < current_score:
                # Accept improvement
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_seq = x.copy()
            else:
                # Revert
                x[idx] = old_val
        
        # If we didn't improve in the last block, restart with a new random sequence
        # Simple check: if best_score hasn't changed in this block, restart
        # For simplicity, just restart every block if no improvement found in the block?
        # Let's track if any improvement was made in this block
        # Actually, the loop above updates best_score. If best_score doesn't change, we are stuck.
        # Let's just restart the sequence periodically or if stuck.
        
        # A simpler restart strategy:
        # If we haven't improved the global best in the last 1000 iterations, restart.
        # But implementing that inside the tight loop is messy.
        # Let's just do a few random restarts.
        
        # Let's try a different initialization: spectral phase retrieval
        # Initialize X_k with random phases and flat magnitude.
        # x = ifft(X)
        # Project x to +/- 1 (sign function)
        # Iterate.
        
        # This is often faster for low PSL.
        
        # 1. Generate random phases
        phases = np.random.uniform(0, 2*np.pi, N)
        # Flat magnitude spectrum (approximation of white noise)
        mags = np.ones(N)
        X = mags * np.exp(1j * phases)
        
        # Ensure Hermitian symmetry for real x?
        # DFT of real sequence is Hermitian: X[k] = conj(X[N-k])
        # If we just take ifft of arbitrary X, x will be complex.
        # We need to enforce Hermitian symmetry or take real part of ifft.
        # Taking real part of ifft is equivalent to projecting onto the real subspace.
        
        for _ in range(50): # 50 iterations of spectral shaping
            if time.time() > deadline:
                break
            x_complex = np.fft.ifft(X)
            x_real = np.real(x_complex)
            # Project to +/- 1
            x_proj = np.sign(x_real)
            # Handle zeros (unlikely with continuous phases, but just in case)
            x_proj[x_proj == 0] = 1
            
            # Calculate score
            score = _autocorr_sidelobe_max(x_proj)
            if score < best_score:
                best_score = score
                best_seq = x_proj.copy()
            
            # Update X
            X_new = np.fft.fft(x_proj)
            # Keep the magnitude of X_new? Or keep the original flat magnitude?
            # Standard phase retrieval: keep magnitude of target, update phase from data.
            # Here target is "low peak sidelobe", which corresponds to flat magnitude.
            # So we keep the flat magnitude (1.0) and update the phase from X_new.
            # But wait, if we keep flat magnitude, we are forcing the spectrum to be flat.
            # This is a strong constraint.
            # Let's try: X = 1.0 * exp(1j * angle(X_new))
            angles = np.angle(X_new)
            X = np.exp(1j * angles)
            
        # Restart with new random phases
        phases = np.random.uniform(0, 2*np.pi, N)
        X = np.exp(1j * phases)
        
    return best_seq if best_seq is not None else np.ones(N)

# EVOLVE-BLOCK-END
