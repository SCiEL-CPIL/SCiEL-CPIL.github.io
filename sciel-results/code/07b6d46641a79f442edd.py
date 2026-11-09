import numpy as np
import time

N = 400

def construct():
    """
    Constructs a signed step function (sequence of +/- 1) of length N
    that minimizes the peak sidelobe level of its autocorrelation.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with Thue-Morse sequence for a good starting point
    # Thue-Morse has low autocorrelation properties
    h = _thue_morse(N)
    
    # Compute initial autocorrelation vector
    # R[k] = sum_{i=0}^{N-1-k} h[i] h[i+k]
    R = _compute_autocorr(h)
    current_psl = np.max(np.abs(R[1:]))
    
    best_h = h.copy()
    best_psl = current_psl
    
    # Simulated Annealing Parameters
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling = 0.999  # Cooling factor
    
    # We will run until time limit
    # Each step is O(N) for updating R and O(N) for finding max PSL
    # N=400, so ~800 operations per step.
    # 100,000 steps ~ 80,000,000 ops. Python can do ~10-50M simple ops/sec.
    # So ~1-8 seconds for 100k steps. We can do many more.
    
    while time.time() < deadline:
        # Select a random index to flip
        i = np.random.randint(N)
        
        # Calculate change in autocorrelation if we flip h[i]
        # Delta R[k] = -2 * h[i] * (h[i+k] + h[i-k]) for valid indices
        # We need to update R in place or calculate new R.
        # To avoid allocation overhead, we can update R directly.
        
        old_h_i = h[i]
        
        # Store old R values to revert if needed? 
        # Or just compute new PSL and revert if rejected.
        # Reverting requires storing the changes or recomputing.
        # Storing R is cheap (400 floats).
        R_old = R.copy()
        
        # Update R
        # For k from 0 to N-1:
        # Term 1: h[i]h[i+k] exists if i+k < N. Contribution change: -2 h[i] h[i+k]
        # Term 2: h[i-k]h[i] exists if i-k >= 0. Contribution change: -2 h[i-k] h[i]
        
        # Vectorized update might be tricky with boundaries.
        # Loop is safer and fast enough for N=400.
        for k in range(N):
            delta = 0.0
            if i + k < N:
                delta -= 2.0 * old_h_i * h[i+k]
            if i - k >= 0:
                delta -= 2.0 * h[i-k] * old_h_i
            R[k] += delta
            
        # Flip h[i]
        h[i] = -old_h_i
        
        # Compute new PSL
        new_psl = np.max(np.abs(R[1:]))
        
        # Acceptance criterion
        if new_psl < current_psl:
            # Always accept improvement
            current_psl = new_psl
            if new_psl < best_psl:
                best_psl = new_psl
                best_h = h.copy()
        else:
            # Check Boltzmann acceptance
            delta_psl = new_psl - current_psl
            if T > 0 and np.random.rand() < np.exp(-delta_psl / T):
                current_psl = new_psl
            else:
                # Reject: revert h and R
                h[i] = old_h_i
                R[:] = R_old
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_h

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n (+/- 1)."""
    seq = np.ones(n, dtype=np.int8)
    # Standard Thue-Morse is 0,1,1,0,0,1,1,0...
    # We map 0 -> 1, 1 -> -1 or similar.
    # Let's generate bits first.
    bits = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        bits[i] = bits[i >> 1] ^ (i & 1)
    # Map bits to +/- 1
    # 0 -> 1, 1 -> -1
    seq = 1 - 2 * bits
    return seq

def _compute_autocorr(h):
    """
    Computes the aperiodic autocorrelation vector R.
    R[k] = sum_{i=0}^{N-1-k} h[i] h[i+k]
    """
    R = np.zeros(N)
    for k in range(N):
        R[k] = np.dot(h[:N-k], h[k:N])
    return R
