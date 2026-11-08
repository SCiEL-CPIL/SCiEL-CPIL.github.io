import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initialize with all 1s
    x = np.ones(N, dtype=np.int8)
    
    def get_score(x):
        s = np.sum(x)
        if s == 0:
            return np.inf
        c = np.correlate(x, x, mode='full')
        # We only care about lags 0 to N-1 for max abs?
        # Autoconvolution is symmetric. c[k] for k=0..N-1 corresponds to lags 0..N-1.
        # np.correlate(x, x, 'full') has length 2N-1.
        # The central part is lags -(N-1) to N-1.
        # We want max abs of the whole thing?
        # "maximum absolute autoconvolution" usually implies over all lags.
        # Since it's symmetric, max over k=0..N-1 is enough.
        max_c = np.max(np.abs(c[N-1:])) # lags 0 to N-1
        return max_c / (s * s)
    
    current_score = get_score(x)
    best_x = x.copy()
    best_score = current_score
    
    # Local Search
    # We will try to improve by flipping bits
    # To avoid O(N^2) per iteration, we can just do a few passes.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            # Try flipping x[i]
            old_val = x[i]
            x[i] = -old_val
            
            new_score = get_score(x)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
                improved = True
            else:
                x[i] = old_val
                
    return best_x

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 400

def _compute_score(x):
    s = np.sum(x)
    if s == 0:
        return np.inf
    # Efficient autocorrelation using FFT
    # Zero pad to 2N
    n = len(x)
    size = 1
    while size < 2 * n:
        size *= 2
        # Ensure power of 2 for speed, though numpy fft is fast for any size
        # Actually, just use 2*n or next power of 2
    # Simpler: use np.correlate for N=400, it's fast enough
    c = np.correlate(x, x, mode='full')
    # c is length 2N-1. Center is N-1.
    # Lags 0..N-1 are c[N-1 : 2*N-1]
    max_c = np.max(np.abs(c[N-1:]))
    return max_c / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with all ones
    x = np.ones(N, dtype=np.int8)
    current_score = _compute_score(x)
    best_x = x.copy()
    best_score = current_score
    
    # Local search with restarts
    # We will do a greedy hill climb
    # To escape local minima, we might add some randomization or restarts
    
    for restart in range(50): # Max restarts
        if time.time() > deadline:
            break
            
        # Initialize
        if restart == 0:
            x = np.ones(N, dtype=np.int8)
        else:
            # Random initialization biased towards +1 to keep S large?
            # Or just random?
            # Random binary might have small S, leading to huge score.
            # Let's try random but ensure S is not too small.
            x = np.random.choice([1, -1], size=N).astype(np.int8)
            s = np.sum(x)
            if abs(s) < N * 0.2:
                # Force some +1s
                x = np.ones(N, dtype=np.int8)
                # Flip 20% randomly
                indices = np.random.choice(N, size=int(0.2*N), replace=False)
                x[indices] = -1
        
        current_score = _compute_score(x)
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle order to avoid bias?
            # Or just iterate.
            for i in range(N):
                if time.time() > deadline:
                    break
                old_val = x[i]
                x[i] = -old_val
                new_score = _compute_score(x)
                
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_x = x.copy()
                    improved = True
                else:
                    x[i] = old_val

    return best_x

# EVOLVE-BLOCK-END
