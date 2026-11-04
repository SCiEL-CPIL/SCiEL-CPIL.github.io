"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_autocorr(s):
    """Compute linear autocorrelation of s (length N)."""
    n = len(s)
    # Use FFT for initial computation, or direct? 
    # Direct is O(N^2) = 160k, which is fast enough for init.
    # But FFT is faster. Let's use FFT for init to be safe, then direct updates.
    # Actually, direct is simpler to verify.
    A = np.zeros(n)
    for k in range(n):
        A[k] = np.dot(s[:n-k], s[k:])
    return A


def _objective(A):
    """Compute the objective value (L3 norm of autocorrelation)."""
    return np.sum(np.abs(A) ** 3)


def _construct_initial():
    """Generate initial random sequence."""
    rng = np.random.RandomState(42)
    s = rng.randint(0, 2, size=N).astype(np.float64)
    s = 2.0 * s - 1.0 # Map to {-1, 1}
    return s


def _local_search_fast(s, time_limit=120.0):
    """
    Fast local search using O(N) updates for autocorrelation.
    Minimizes sum(|A[k]|^3).
    """
    n = len(s)
    A = _compute_autocorr(s)
    best_obj = _objective(A)
    start_time = time.time()
    
    # Precompute powers for speed? No, just compute delta.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for j in range(n):
            # Calculate delta A for flipping s[j]
            # s[j] is currently val. New val is -val.
            # Delta A[k] = -2 * s[j] * s[j+k] (if j+k < n) - 2 * s[j] * s[j-k] (if j-k >= 0)
            # Note: for k=0, delta is 0.
            
            # We need to update A and check if objective improves.
            # To avoid modifying A if it doesn't improve, we can compute the delta in objective directly.
            
            # Affected lags k:
            # 1. k such that j+k < n => k < n-j
            # 2. k such that j-k >= 0 => k <= j
            # Union is k in [0, n-1].
            
            # Let's compute the new A values for affected k and the change in objective.
            # To be efficient, we can just update A, compute new obj, and revert if worse.
            # Since N=400, updating A is 400 ops. Computing obj is 400 ops.
            # Total 800 ops per flip. 400 flips/pass => 320k ops/pass.
            # 120s / 320k ops ~ 3750 passes. This is plenty.
            
            old_A = A.copy() # Copy is 400 ops.
            
            # Update A
            for k in range(n):
                delta = 0.0
                if j + k < n:
                    delta -= 2.0 * s[j] * s[j+k]
                if j - k >= 0:
                    delta -= 2.0 * s[j] * s[j-k]
                if delta != 0.0:
                    A[k] += delta
            
            new_obj = _objective(A)
            
            if new_obj < best_obj:
                best_obj = new_obj
                improved = True
                # Accept change: s[j] is already flipped? No, we didn't flip s in the loop above.
                # We need to flip s[j] to reflect the change in A.
                s[j] = -s[j]
            else:
                # Revert A
                A[:] = old_A
                
    return s, best_obj


def construct():
    deadline = time.time() + 140.0
    
    # Start with random uniform distribution
    s = _construct_initial()
    
    # Run fast local search
    if time.time() < deadline - 5.0:
        s, _ = _local_search_fast(s, time_limit=130.0)
        
    return s
# EVOLVE-BLOCK-END
