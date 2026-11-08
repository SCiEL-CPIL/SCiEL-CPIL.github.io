import numpy as np
import time

N = 400
DEADLINE = 140.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 is defined as the maximum absolute value of the autoconvolution.
    """
    # signed_step is an array of N values in {-1, 1}
    
    # Use FFT to compute the autoconvolution efficiently
    # We need to pad to at least 2*N to avoid circular convolution effects
    size = 1 << (2 * N - 1).bit_length()  # Next power of 2 >= 2*N
    
    # Build the signal on a fine grid
    # Each interval has width h = 1/N
    # We'll represent the function on a grid of 'size' points
    f = np.zeros(size)
    
    block_size = size // N
    # Vectorized construction for the valid part
    valid_len = N * block_size
    idx_map = np.arange(valid_len) // block_size
    f[:valid_len] = signed_step[idx_map]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real
    
    # Scale by the width of each grid cell
    # The integral is approximated by sum * dx, where dx = 0.5 / size
    # Domain is [-1/4, 1/4] so total length is 0.5.
    dx = 0.5 / size
    autoconv *= dx
    
    # The autoconvolution is defined on [-0.5, 0.5]
    # C3 = max |autoconv|
    c3 = np.max(np.abs(autoconv))
    
    return c3

def construct():
    start_time = time.time()
    
    # Initialize with a random pattern
    np.random.seed(42)
    best_step = np.random.choice([-1, 1], size=N).astype(np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Precompute the full autoconvolution for the initial state to enable delta updates
    # We need to maintain the autoconvolution array A such that A[k] = sum_i f[i] f[i+k]
    # Note: The FFT-based autoconvolution computes circular convolution if not padded, 
    # but here we padded to size. The linear autoconvolution is embedded in the FFT result.
    # However, for delta updates, it is easier to work with the linear autoconvolution array 
    # of length 2N-1.
    
    # Let's re-derive the delta update for linear autoconvolution.
    # Let f be the sequence of length N.
    # A[k] = sum_{i=0}^{N-1-k} f[i] f[i+k] for k >= 0
    # A[-k] = A[k]
    # C3 is max |A[k]| * dx, where dx is the scaling factor.
    # Actually, the problem statement says "autoconvolution upper bound".
    # The seed program used FFT on a padded grid. 
    # Let's stick to the FFT definition to be safe, but we can approximate the delta.
    # Actually, the delta update for the *linear* autoconvolution is exact and much faster.
    # Does the evaluator use linear or circular? 
    # The seed program padded to power of 2 >= 2N-1, which effectively computes linear autoconvolution 
    # (with zero padding). So A_lin[k] is what we want.
    
    # Let's compute A_lin for the current best_step.
    # A_lin[k] for k in [-(N-1), N-1]
    # We can use np.correlate or manual loop. For speed, we'll maintain it.
    
    def get_autoconv_lin(f):
        # Returns array of length 2N-1, centered at 0.
        # Index 0 corresponds to lag -(N-1), index N-1 to lag 0, etc.
        # Or simpler: use np.correlate(f, f, mode='full')
        # np.correlate returns length 2N-1.
        # c[k] = sum_i f[i] conj(f[i+k])
        # For real f, c[k] = sum_i f[i] f[i+k]
        return np.correlate(f, f, mode='full')
    
    current_step = best_step.copy()
    current_A = get_autoconv_lin(current_step)
    # The scaling factor dx from the seed program:
    size = 1 << (2 * N - 1).bit_length()
    dx = 0.5 / size
    # Note: The seed program's autoconv was scaled by dx.
    # The max value is max(|current_A|) * dx.
    current_c3 = np.max(np.abs(current_A)) * dx
    
    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 0.001
    alpha = 0.9995 # Cooling rate
    
    # We will perform many iterations.
    # Each iteration:
    # 1. Pick a random index i to flip.
    # 2. Compute delta A.
    # 3. Update A.
    # 4. Compute new C3.
    # 5. Accept/Reject.
    
    # To compute delta A efficiently:
    # If we flip f[i] to -f[i], the change in A[k] is:
    # A_new[k] = A_old[k] - 2 * f[i] * (f[i+k] + f[i-k] - 2*f[i]*delta_{k,0})? 
    # Let's derive:
    # A[k] = sum_j f[j] f[j+k]
    # Terms involving f[i]:
    # 1. j=i: f[i] f[i+k]
    # 2. j+k=i => j=i-k: f[i-k] f[i]
    # So A[k] contains f[i]f[i+k] + f[i-k]f[i] (if indices valid).
    # If we flip f[i] -> -f[i], the new terms are -f[i]f[i+k] - f[i-k]f[i].
    # Delta A[k] = New - Old = -2 * f[i] * (f[i+k] + f[i-k])
    # Wait, if k=0, the term is f[i]f[i] = 1. New is (-f[i])^2 = 1. Delta is 0?
    # Let's check:
    # Old: f[i]^2 = 1
    # New: (-f[i])^2 = 1
    # Delta = 0.
    # Formula: -2 * f[i] * (f[i] + f[i]) = -4 f[i]^2 = -4.
    # So the formula needs a correction for k=0.
    # Actually, for k=0, the term f[i]f[i] appears only once in the sum?
    # np.correlate definition: c[k] = sum_n x[n] y[n+k]
    # For k=0, c[0] = sum x[n]^2.
    # If we flip x[i], x[i]^2 becomes (-x[i])^2 = x[i]^2. So c[0] does NOT change.
    # My derivation above assumed distinct terms.
    
    # Let's use a more robust delta update logic.
    # A[k] = sum_{j} f[j] f[j+k]
    # Flipping f[i]:
    # For a given lag k:
    # The term f[i] f[i+k] is present if i+k < N.
    # The term f[i-k] f[i] is present if i-k >= 0.
    # Note: If k=0, these are the same term f[i]^2.
    
    # Let's precompute the "influence" of each index i on each lag k.
    # This is essentially the structure of the autoconvolution.
    # Instead of complex delta logic, let's use the property that 
    # flipping f[i] changes A[k] by -2 f[i] (f[i+k] + f[i-k]) for k != 0,
    # and 0 for k=0?
    # Let's test with small example.
    # f = [1, 1]
    # A = [1, 2, 1] (lags -1, 0, 1)
    # Flip f[0] -> [-1, 1]
    # A_new = [-1, 1, -1]
    # Delta for lag 0: 1 -> 1 (change 0).
    # Delta for lag 1: 2 -> -1 (change -3).
    # Formula for k=1: -2 * f[0] * (f[0+1] + f[0-1])? 
    # f[0-1] is out of bounds.
    # Valid terms for k=1 in old A: f[0]f[1] = 1.
    # In new A: f[0]f[1] = -1.
    # Change: -2.
    # Wait, A[1] in np.correlate [1,1] is 1?
    # np.correlate([1,1], [1,1], 'full') -> [1, 2, 1]
    # Index 0: lag -1: f[1]f[0] = 1
    # Index 1: lag 0: f[0]^2 + f[1]^2 = 2
    # Index 2: lag 1: f[0]f[1] = 1
    # So A[2] (lag 1) is 1.
    # New f: [-1, 1]
    # np.correlate([-1,1], [-1,1], 'full') -> [1, 2, -1]
    # A_new[2] (lag 1) is -1.
    # Delta = -2.
    # Formula: -2 * f[0] * (f[1]) = -2 * 1 * 1 = -2. Correct.
    # What about lag -1 (index 0)?
    # Old A[0] = 1. New A[0] = 1. Delta 0.
    # Formula: -2 * f[0] * (f[-1])? f[-1] invalid.
    # Term in A[-1] is f[1]f[0].
    # Flip f[0]: f[1](-f[0]) = -f[1]f[0].
    # Change: -2 f[1]f[0].
    # Formula should be -2 * f[i] * f[i-k] for the term where i is the second index?
    # Let's generalize:
    # A[k] = sum_j f[j] f[j+k]
    # Flipping f[i]:
    # Term 1: j=i -> f[i] f[i+k]. Change: -2 f[i] f[i+k]
    # Term 2: j+k=i -> j=i-k -> f[i-k] f[i]. Change: -2 f[i-k] f[i]
    # If k=0, Term 1 and Term 2 are the same. Change is -2 f[i]^2?
    # But we know change is 0.
    # Why? Because f[i]^2 is always 1.
    # So for k=0, delta is 0.
    # For k != 0:
    # Delta A[k] = -2 f[i] ( f[i+k] + f[i-k] )
    # Where out-of-bounds terms are 0.
    
    # We can vectorize this delta calculation for a given i.
    # For a fixed i, the lags k that change are those where i+k or i-k is in [0, N-1].
    # k ranges from -N+1 to N-1.
    # We can create a vector delta_A of length 2N-1.
    
    # Let's implement the SA loop.
    
    T = T_start
    no_improve = 0
    patience = 1000
    
    # Preallocate delta array
    delta_A = np.zeros(2*N-1)
    
    while time.time() - start_time < DEADLINE - 2.0:
        # Pick random index
        i = np.random.randint(N)
        
        # Calculate delta_A
        # Reset delta_A to 0
        delta_A[:] = 0
        
        # Term 1: j=i -> affects lag k such that i+k is valid.
        # k = j' - i where j' = i+k.
        # Actually, let's just iterate over valid neighbors.
        # This is O(N) per step, which is fine.
        
        # Vectorized approach for delta:
        # We want to add -2*f[i]*f[j] to A[lag] where lag = j-i or i-j?
        # A[k] = sum f[j] f[j+k].
        # If we flip f[i], it affects:
        # 1. f[i] * f[i+k] for all k where i+k in [0, N-1].
        #    Lag k = (i+k) - i. So lag = j - i where j is the other index.
        # 2. f[i-k] * f[i] for all k where i-k in [0, N-1].
        #    Lag k = i - (i-k) = k. So lag = i - j where j is the other index.
        
        # Let's use the explicit formula:
        # For each valid j in [0, N-1]:
        # If j == i: delta = 0 (as derived)
        # Else:
        #   k = j - i.
        #   This contributes to A[k] via term f[j]f[i]? No, A[k] = sum f[m] f[m+k].
        #   If m=i, term is f[i] f[i+k] = f[i] f[j].
        #   If m+k=i, term is f[i-k] f[i] = f[j] f[i].
        #   So for pair (i, j) with i != j, it appears in A[j-i] and A[i-j].
        #   Wait, A is symmetric. A[k] = A[-k].
        #   np.correlate returns full array.
        #   Let's just use the direct update on the array.
        
        # Simpler:
        # Create a mask of indices j where j != i.
        # This is still O(N).
        
        # Let's try a different vectorization.
        # The change in A is -2 * f[i] * (f shifted by i + f shifted by -i) ?
        # Let g be a vector with g[i] = 1, else 0.
        # The interaction is 2 * f[i] * (f * g_convolved_with_shift)?
        
        # Let's just do the O(N) loop in NumPy vectorized way.
        # Indices j: 0..N-1
        # Lags k: j - i
        # For each j != i:
        #   A[j-i] changes by -2 f[i] f[j]
        #   A[i-j] changes by -2 f[i] f[j] (since A is symmetric, and the term is the same)
        
        # Wait, does A[k] and A[-k] both change?
        # A[k] = sum f[m] f[m+k].
        # Term f[i] f[i+k] is in A[k].
        # Term f[i-k] f[i] is in A[k].
        # If we flip f[i], both terms change.
        # So A[k] changes by -2 f[i] (f[i+k] + f[i-k]).
        # This is for a specific k.
        
        # Let's compute delta_A for all k at once.
        # delta_A[k] = -2 * f[i] * (f[i+k] + f[i-k])
        
        # We can create an array of lags k from -(N-1) to N-1.
        # k_vals = np.arange(-(N-1), N)
        # idx1 = i + k_vals
        # idx2 = i - k_vals
        # valid1 = (idx1 >= 0) & (idx1 < N)
        # valid2 = (idx2 >= 0) & (idx2 < N)
        # terms1 = np.where(valid1, current_step[idx1], 0)
        # terms2 = np.where(valid2, current_step[idx2], 0)
        # delta_A = -2 * current_step[i] * (terms1 + terms2)
        # But for k=0, terms1 and terms2 are both f[i].
        # delta_A[0] = -2 f[i] (2 f[i]) = -4 f[i]^2 = -4.
        # But we know delta_A[0] should be 0.
        # So we set delta_A[k=0] = 0.
        
        k_vals = np.arange(-(N-1), N)
        idx1 = i + k_vals
        idx2 = i - k_vals
        
        valid1 = (idx1 >= 0) & (idx1 < N)
        valid2 = (idx2 >= 0) & (idx2 < N)
        
        terms1 = np.where(valid1, current_step[idx1], 0)
        terms2 = np.where(valid2, current_step[idx2], 0)
        
        delta_A = -2 * current_step[i] * (terms1 + terms2)
        
        # Correct k=0
        # k=0 corresponds to index N-1 in k_vals (since k_vals starts at -(N-1))
        idx_zero_k = N - 1
        delta_A[idx_zero_k] = 0
        
        # Apply delta to current_A
        new_A = current_A + delta_A
        
        # Calculate new C3
        new_c3 = np.max(np.abs(new_A)) * dx
        
        # Acceptance criterion
        if new_c3 < current_c3:
            # Always accept
            current_A = new_A
            current_step[i] *= -1
            current_c3 = new_c3
            no_improve = 0
            
            if current_c3 < best_c3:
                best_c3 = current_c3
                best_step = current_step.copy()
        else:
            # Accept with probability exp(-(new_c3 - current_c3) / T)
            # Note: C3 is positive.
            delta_cost = new_c3 - current_c3
            if np.random.rand() < np.exp(-delta_cost / T):
                current_A = new_A
                current_step[i] *= -1
                current_c3 = new_c3
            else:
                # Reject, no change
                pass
        
        # Cooling
        T *= alpha
        if T < T_end:
            T = T_end
            
        # Stagnation check
        # If no improvement for a long time, we might restart or just continue cooling.
        # With SA, we just keep going.
        
        # To prevent infinite loop if T is very small and we never accept,
        # we can check time.
        
    return best_step
