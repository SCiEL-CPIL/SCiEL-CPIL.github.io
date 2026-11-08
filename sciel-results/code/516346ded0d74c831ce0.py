"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _autoconv_max_abs(h):
    """Compute max absolute value of autoconvolution of h."""
    # Use FFT for efficient convolution
    n = len(h)
    size = 2 * n
    fft_h = np.fft.fft(h, n=size)
    conv = np.fft.ifft(fft_h * np.conj(fft_h)).real
    # The linear convolution of h with h has length 2n-1
    # The FFT of size 2n gives the circular convolution, which equals linear conv if padded
    # The result is in conv[0:n-1] and conv[n:size-1]
    # We take the max abs over the full linear convolution range
    # For linear conv of length 2n-1, indices 0 to 2n-2
    # In the zero-padded FFT result of size 2n, the linear conv corresponds to indices 0 to 2n-2
    # Note: index 2n-1 is the wrap-around which should be 0 for linear conv if padded correctly?
    # Actually, for h of length n, padded to 2n, the linear conv is exactly conv[0:2n-1]
    # conv[2n-1] is 0 (since h[2n-1] is 0 in the padded array? No, h is length n, so h[n:2n] is 0)
    # So conv[2n-1] is indeed 0.
    return np.max(np.abs(conv[:2*n-1]))

def _objective(h):
    """Compute the objective function: max |autoconv| / (sum h)^2."""
    s = np.sum(h)
    if s == 0:
        return float('inf')
    return _autoconv_max_abs(h) / (s * s)

def _hill_climb(h, max_iter=10000, time_limit=10.0):
    """Local search by flipping signs."""
    best_h = h.copy()
    best_obj = _objective(best_h)
    start_time = time.time()
    n = len(best_h)
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random index to flip
        idx = random.randint(0, n - 1)
        new_h = best_h.copy()
        new_h[idx] = -new_h[idx]
        
        new_obj = _objective(new_h)
        
        if new_obj < best_obj:
            best_h = new_h
            best_obj = new_obj
            
    return best_h, best_obj

def _generate_rudin_shapiro(n):
    """Generate Rudin-Shapiro sequence of length n."""
    # Rudin-Shapiro sequence: r(0)=1, r(1)=1, r(2n)=r(n), r(2n+1)=-r(n+1)
    # This is for the infinite sequence. For finite length, we generate up to n.
    seq = []
    # We need to generate enough terms
    # Let's use the iterative construction
    # Actually, the standard RS sequence is defined recursively.
    # Let's use a simpler approach: generate the sequence using the recursion.
    # r(0) = 1, r(1) = 1
    # r(2k) = r(k), r(2k+1) = -r(k+1)
    
    # We need to compute up to index n-1.
    # This requires knowing r(k) for k < n.
    # Let's build it iteratively.
    # This is tricky because of the dependency.
    # Let's use the known property that RS sequences have low auto-correlation.
    # For now, let's use a random initialization and rely on hill climbing.
    # Or, we can use a simple pattern like alternating signs.
    # Let's try a few different initializations.
    
    # Initialization 1: Random
    h1 = np.random.choice([-1, 1], size=n)
    
    # Initialization 2: Alternating
    h2 = np.ones(n)
    h2[1::2] = -1
    
    # Initialization 3: Block pattern (e.g., 2 +, 2 -)
    h3 = np.ones(n)
    h3[2::4] = -1
    h3[3::4] = -1
    
    # Initialization 4: Rudin-Shapiro like (using the recursion properly)
    # Let's implement the recursion properly.
    # We need a list to store the values.
    # The recursion r(2k) = r(k), r(2k+1) = -r(k+1)
    # To compute r(n), we need r(k) for k < n.
    # Let's compute up to n.
    # This is not straightforward to do in O(n) without a stack or memoization.
    # Let's use a simple heuristic instead.
    
    # Let's just return the best of these initializations.
    best_h = None
    best_obj = float('inf')
    
    for h_init in [h1, h2, h3]:
        obj = _objective(h_init)
        if obj < best_obj:
            best_obj = obj
            best_h = h_init.copy()
            
    return best_h

def construct():
    """Construct the sequence by searching for a good sign pattern."""
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Generate initial candidates
    candidates = []
    
    # Random initializations
    for _ in range(5):
        h = np.random.choice([-1, 1], size=N)
        candidates.append(h)
        
    # Pattern initializations
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    candidates.append(h_alt)
    
    h_block = np.ones(N)
    h_block[2::4] = -1
    h_block[3::4] = -1
    candidates.append(h_block)
    
    # Evaluate all candidates
    best_h = None
    best_obj = float('inf')
    
    for h in candidates:
        obj = _objective(h)
        if obj < best_obj:
            best_obj = obj
            best_h = h.copy()
            
    # Hill climb from the best candidate
    if time.time() < deadline:
        best_h, best_obj = _hill_climb(best_h, max_iter=50000, time_limit=deadline - time.time())
        
    # If we have time, try hill climbing from other good candidates
    if time.time() < deadline:
        # Sort candidates by objective value
        candidate_objs = [( _objective(h), h) for h in candidates]
        candidate_objs.sort(key=lambda x: x[0])
        
        # Try the next few candidates
        for _, h in candidate_objs[1:3]:
            if time.time() >= deadline:
                break
            h_climbed, obj_climbed = _hill_climb(h, max_iter=20000, time_limit=deadline - time.time())
            if obj_climbed < best_obj:
                best_h = h_climbed
                best_obj = obj_climbed
                
    return best_h
# EVOLVE-BLOCK-END
