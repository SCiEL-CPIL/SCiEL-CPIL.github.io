import numpy as np

N = 400
h = 1.0 / (2 * N)

def autocorr_peak(f):
    """Compute max |autocorrelation| for step function f on [-1/4, 1/4]."""
    R = np.correlate(f, f, mode='full') * h
    return np.max(np.abs(R))

def generate_legendre(p):
    """Generate a Legendre sequence of length p (prime)."""
    seq = np.zeros(p)
    for i in range(1, p):
        # Legendre symbol
        if pow(i, (p-1)//2, p) == 1:
            seq[i] = 1.0
        else:
            seq[i] = -1.0
    seq[0] = 1.0 # Convention for 0, doesn't matter much
    return seq

def construct():
    # Initialize with a Legendre sequence of length 397 (prime)
    p = 397
    leg = generate_legendre(p)
    # Pad to N=400
    f = np.zeros(N)
    f[:p] = leg
    # Pad the remaining 3 spots with 0s or continue pattern? 
    # 0s are neutral for correlation if we treat them as 0. 
    # But the task says "signed step heights". Usually implies +/-1.
    # If 0s are allowed, it reduces energy. If only +/-1, we should pad with +/-1.
    # The prompt says "signed step function". Usually implies non-zero. 
    # However, "signed" often allows 0. But "step heights" in signal processing often implies binary.
    # The seed used [-1, 1]. Let's stick to [-1, 1] to be safe, or allow 0?
    # "Choose a signed step function". 
    # If I use 0s, the peak might be lower because total energy is lower.
    # Let's try +/-1 padding first, as it's standard for "signed binary".
    # Actually, if 0 is allowed, the optimal solution might have 0s.
    # Let's assume {-1, 1} for now as per seed, but the hill climb can try 0 if we expand the space.
    # Let's stick to {-1, 1} to match the "signed" binary intuition of the seed.
    f[p:] = 1.0 # Pad with 1s
    
    best_peak = autocorr_peak(f)
    
    # Hill Climbing
    # We will do several passes.
    # To speed up, we can stop early if no improvement in a full pass.
    max_passes = 50
    for _ in range(max_passes):
        improved = False
        # Shuffle order to avoid bias? Or just sequential.
        # Sequential is faster.
        for i in range(N):
            old_val = f[i]
            new_val = -old_val
            f[i] = new_val
            new_peak = autocorr_peak(f)
            if new_peak < best_peak:
                best_peak = new_peak
                improved = True
            else:
                f[i] = old_val
        
        if not improved:
            break
            
    return f
