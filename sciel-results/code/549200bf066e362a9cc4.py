import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Generate Thue-Morse sequence
    # t_n = parity of number of 1s in binary representation of n
    # Map 0 -> 1, 1 -> -1 (or vice versa)
    def thue_morse(n):
        # Efficient generation for first n terms
        # t_0 = 0
        # t_{2n} = t_n
        # t_{2n+1} = 1 - t_n
        t = np.zeros(n, dtype=int)
        # Use a simple loop or vectorized trick?
        # Vectorized trick: count bits in np.arange(n)
        # This might be slow for large N, but N=400 is small.
        indices = np.arange(n)
        # Count bits using a lookup or iterative doubling
        # Simple method:
        # t = np.array([bin(i).count('1') % 2 for i in range(n)])
        # Let's use the recursive property for speed and correctness
        if n == 0:
            return np.array([], dtype=int)
        t[0] = 0
        i = 1
        while i < n:
            # Fill t[i] based on t[i//2]
            # This is O(N)
            t[i] = t[i // 2] if i % 2 == 0 else 1 - t[i // 2]
            i += 1
        # Convert to signed steps: 0 -> 1, 1 -> -1
        return 1 - 2 * t

    # Initialize with Thue-Morse
    f = thue_morse(N)
    
    # We can also try the complement (multiply by -1), but autocorrelation magnitude is same.
    # We can also try reversing, etc. Thue-Morse is symmetric in a way.
    
    def compute_autocorr(f):
        # Aperiodic autocorrelation
        # R[k] = sum_{i=0}^{N-1-|k|} f[i] f[i+|k|]
        # Using convolution: conv(f, f[::-1])
        conv = np.convolve(f, f[::-1])
        # Center is at N-1
        # We want max abs of all lags except 0
        R = conv.copy()
        R[N-1] = 0.0
        return R

    R = compute_autocorr(f)
    current_max_abs = np.max(np.abs(R))
    best_f = f.copy()
    best_max_abs = current_max_abs

    # Local Search: Greedy Bit Flip
    # Try flipping each bit and see if it reduces the global max autocorrelation.
    # To speed up, we can update R incrementally, but N=400 is small enough
    # that recomputing conv (O(N^2) or O(N log N) via FFT) might be okay if we do it smartly.
    # np.convolve is O(N^2) for N=400 -> 160,000 ops. Very fast.
    # Let's do a full recompute for each candidate flip in a greedy step.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_flip_idx = -1
        best_new_max = current_max_abs
        
        for i in range(N):
            # Flip f[i]
            f[i] *= -1
            R_new = compute_autocorr(f)
            max_new = np.max(np.abs(R_new))
            
            if max_new < best_new_max - 1e-9:
                best_new_max = max_new
                best_flip_idx = i
                best_f_candidate = f.copy()
                best_R_candidate = R_new.copy()
            
            # Undo flip
            f[i] *= -1
            
        if best_flip_idx != -1:
            f = best_f_candidate
            R = best_R_candidate
            current_max_abs = best_new_max
            best_f = f.copy()
            best_max_abs = current_max_abs
            improved = True
        else:
            # Local minimum reached.
            # Try some random flips to escape?
            # Or just break. Thue-Morse is often a global or near-global minimum for this metric.
            # Given time constraints, breaking is safer.
            # Let's try a few random perturbations if we have time.
            if time.time() < deadline - 5.0:
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    idx = np.random.randint(N)
                    f[idx] *= -1
                    R_new = compute_autocorr(f)
                    max_new = np.max(np.abs(R_new))
                    if max_new < best_max_abs:
                        best_f = f.copy()
                        best_max_abs = max_new
                        f = best_f.copy()
                        improved = True
                        break
                    else:
                        f[idx] *= -1 # Undo

    return best_f
