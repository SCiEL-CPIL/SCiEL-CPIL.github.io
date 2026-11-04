import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    def compute_R_positive(f):
        """Compute R[L] for L=1..N-1 efficiently using FFT."""
        # Aperiodic autocorrelation: R[L] = sum_{i=0}^{N-1-L} f[i] * f[i+L]
        # This is the first half of the linear autocorrelation.
        # We can compute the full linear autocorrelation via FFT.
        # Let's use the standard trick:
        # Linear autocorrelation c[k] = sum_{n} x[n] x[n+k]
        # We can compute this via: ifft(fft(x, 2N) * conj(fft(x, 2N)))
        # But we need to be careful with the indexing.
        
        # A simpler way for N=400 is just a loop, which is 400 * 400 / 2 = 80,000 ops.
        # This is very fast in NumPy if vectorized, or even in a pure Python loop with small N.
        # Let's try a vectorized approach first.
        
        R = np.zeros(N)
        # For each lag L, R[L] = dot(f[:N-L], f[L:])
        # We can do this in a loop.
        for L in range(1, N):
            R[L] = np.dot(f[:N-L], f[L:])
        return R

    def max_abs_autocorr(f):
        R = compute_R_positive(f)
        return np.max(np.abs(R[1:]))

    rng = np.random.RandomState(42)
    
    # Initial best
    best_f = None
    best_val = float('inf')
    
    # Try a few random starts
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        f = rng.choice([-1.0, 1.0], size=N)
        val = max_abs_autocorr(f)
        if val < best_val:
            best_val = val
            best_f = f.copy()
    
    if best_f is None:
        best_f = rng.choice([-1.0, 1.0], size=N)
        best_val = max_abs_autocorr(best_f)
    
    current_f = best_f.copy()
    current_val = best_val
    current_R = compute_R_positive(current_f)
    
    # Simulated Annealing with incremental updates
    # When flipping bit k, the change in R[L] is:
    # R[L] = sum f[i] f[i+L]
    # If we flip f[k], then:
    # - For each L, the term f[k] f[k+L] changes if k+L < N.
    # - The term f[k-L] f[k] changes if k-L >= 0.
    # So, for a fixed k, R[L] changes by:
    # - 2 * f[k] * f[k+L] (for the term where k is the first index, i.e., i=k)
    # - 2 * f[k] * f[k-L] (for the term where k is the second index, i.e., i+L=k => i=k-L)
    # Note: The sign of the change depends on the old value of f[k].
    # Let old_f_k = f[k]. New f[k] = -old_f_k.
    # Change in term f[k] f[k+L] is (-old_f_k - old_f_k) * f[k+L] = -2 * old_f_k * f[k+L].
    # Change in term f[k-L] f[k] is f[k-L] * (-old_f_k - old_f_k) = -2 * old_f_k * f[k-L].
    # So total change in R[L] is -2 * old_f_k * (f[k+L] + f[k-L]) (with appropriate bounds).
    
    # We will implement a local search that tries to flip the bit that most reduces the peak.
    # Since N is small, we can afford to check all bits in each iteration.
    
    max_iterations = 20000
    iteration = 0
    
    while time.time() < deadline - 5 and iteration < max_iterations:
        iteration += 1
        
        # Find the lag with the maximum absolute autocorrelation
        # We only care about L=1..N-1
        L_max = np.argmax(np.abs(current_R[1:])) + 1
        R_max = current_R[L_max]
        
        if np.abs(R_max) < 1e-9:
            break
            
        # Try flipping each bit and see which one gives the best improvement
        # To speed this up, we can estimate the new max for each flip.
        # However, updating R for each flip is O(N), and checking all N flips is O(N^2).
        # N=400, so N^2 = 160,000 ops per iteration.
        # 20,000 iterations * 160,000 ops = 3.2 * 10^9 ops. This might be too slow in Python.
        
        # Let's optimize:
        # Instead of checking all bits, we can pick a random bit or a bit near the peak.
        # Or, we can use a greedy approach: pick the bit that most reduces R[L_max].
        
        # Let's try a hybrid:
        # 1. Identify the top few lags with high absolute values.
        # 2. For each of these lags, find the bit that most reduces it.
        # 3. Pick the best among these candidates.
        
        # Actually, a simpler and effective method for this problem is to use
        # a "random walk with restarts" or "simulated annealing" with a single flip.
        # Let's stick with simulated annealing but with a better acceptance criterion.
        
        # Pick a random bit
        k = rng.randint(N)
        
        # Compute the change in R for this flip
        old_f_k = current_f[k]
        
        # We need to update R for all L.
        # Let's create a copy of R to test the flip.
        R_new = current_R.copy()
        
        for L in range(1, N):
            delta = 0
            if k + L < N:
                delta -= 2 * old_f_k * current_f[k+L]
            if k - L >= 0:
                delta -= 2 * old_f_k * current_f[k-L]
            R_new[L] += delta
        
        new_val = np.max(np.abs(R_new[1:]))
        
        # Acceptance criterion: accept if better, or with some probability if worse
        # To escape local minima, we use a simple rule:
        # Accept if new_val < current_val
        # Or, with probability 0.1, accept if new_val < current_val * 1.1 (allow small worsening)
        
        if new_val < current_val:
            current_f[k] *= -1
            current_R = R_new
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_f = current_f.copy()
        else:
            # Revert (do nothing)
            pass
        
        # Occasionally, do a random restart to escape local minima
        if iteration % 1000 == 0 and time.time() < deadline - 20:
            # With some probability, restart from a random point
            if rng.rand() < 0.1:
                f_new = rng.choice([-1.0, 1.0], size=N)
                val_new = max_abs_autocorr(f_new)
                if val_new < current_val:
                    current_f = f_new
                    current_val = val_new
                    current_R = compute_R_positive(current_f)
                    if current_val < best_val:
                        best_val = current_val
                        best_f = current_f.copy()
    
    return best_f
