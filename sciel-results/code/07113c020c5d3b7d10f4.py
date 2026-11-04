import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    def calc_score(x):
        """
        Calculate the score: max |C[k]| for k != 0 divided by S^2.
        x is a binary vector of +/-1.
        """
        S = np.sum(x)
        if S == 0:
            return np.inf
        
        # Linear autocorrelation via FFT
        # C[k] = sum x[i] x[i+k]
        # Using np.correlate is O(N^2). FFT is O(N log N).
        # For N=400, np.correlate is fast enough, but let's use FFT for safety in loops.
        size = 1
        while size < 2 * N:
            size *= 2
        
        X = np.fft.rfft(x, size)
        # The power spectrum is |X|^2. The inverse FFT gives the circular autocorrelation.
        # Since x is zero-padded to size >= 2N-1, the circular autocorrelation 
        # matches the linear autocorrelation for lags -N+1 to N-1.
        C_full = np.fft.irfft(X * X, size)
        
        # C_full[0] is lag 0 (S^2)
        # C_full[1:N] are lags 1 to N-1
        # C_full[N+1:size] are negative lags (symmetric to positive lags for real signals)
        # We only need max absolute value for non-zero lags.
        # Due to symmetry, C[k] = C[-k].
        max_abs_corr = np.max(np.abs(C_full[1:N]))
        
        return max_abs_corr / (S * S)

    def objective_continuous(x):
        """
        Objective for continuous optimization.
        Minimizes the L2 norm of the off-diagonal autocorrelation.
        This acts as a smooth surrogate for the max absolute autocorrelation.
        """
        size = 1
        while size < 2 * N:
            size *= 2
        
        X = np.fft.rfft(x, size)
        C_full = np.fft.irfft(X * X, size)
        
        # Sum of squares of off-diagonal lags
        # Lags 1 to N-1 are at indices 1 to N-1
        # Due to symmetry, we can just sum 1 to N-1 and multiply by 2, or sum all non-zero.
        # Let's sum all non-zero lags in the valid range.
        off_diag = C_full[1:N]
        # Also include the negative lags if they are distinct in the FFT output?
        # irfft returns real values. C[k] = C[-k].
        # The sum of squares over all k != 0 is 2 * sum_{k=1}^{N-1} C[k]^2.
        return np.sum(off_diag * off_diag)

    def local_search(x, max_iters=5000):
        """
        Greedy local search on binary sequence.
        """
        x = x.copy()
        current_score = calc_score(x)
        S = np.sum(x)
        
        # Precompute FFT size
        size = 1
        while size < 2 * N:
            size *= 2
            
        # We will use a simple greedy approach: try flipping each bit, keep if better.
        # To speed up, we can track the current C array and update it incrementally.
        # However, for N=400, recalculating FFT is ~10us. 5000 iters * 400 flips = 2M flips.
        # 2M * 10us = 20s. Too slow for multiple restarts.
        # Incremental update is better.
        
        # C_full = irfft(|fft(x)|^2)
        # Flipping x[i] from 1 to -1 (or vice versa) changes x by delta = -2 * x[i] * e_i
        # This is complex to update in frequency domain efficiently for real FFT without recomputing.
        # Let's stick to a fast local search with a smaller number of iterations or better pruning.
        # Or, use the fact that N is small.
        
        # Actually, let's use a randomized hill climbing with restarts within the local search phase.
        # Or just a few passes of greedy.
        
        improved = True
        passes = 0
        while improved and passes < 10:
            improved = False
            passes += 1
            for i in range(N):
                old_val = x[i]
                x[i] = -old_val
                new_score = calc_score(x)
                
                if new_score < current_score:
                    current_score = new_score
                    S = S - 2 * old_val
                    improved = True
                else:
                    x[i] = old_val
                    
                if time.time() - start_time > time_limit - 2.0:
                    return x, current_score
                    
        return x, current_score

    best_score = np.inf
    best_x = None
    
    # Number of restarts
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # 1. Continuous Optimization
        # Initialize with random normal
        x0 = np.random.randn(N)
        
        # Optimize
        res = minimize(objective_continuous, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-6})
        
        # 2. Project to Binary
        x_bin = np.sign(res.x)
        # Ensure no zeros (sign(0)=0)
        x_bin[x_bin == 0] = 1
        
        # 3. Local Search
        x_refined, score_refined = local_search(x_bin)
        
        if score_refined < best_score:
            best_score = score_refined
            best_x = x_refined.copy()
            
            # If we find a very good score, we can reduce the time spent on future restarts
            # or increase the local search intensity.
            
    # If we still have time, do a more intensive local search on the best result
    if best_x is not None and time.time() - start_time < time_limit - 5.0:
        # Try a simulated annealing or more aggressive local search
        # For now, just run local_search again with more passes if time permits
        # The previous local_search had max 10 passes.
        # Let's just run it again, it will stop if no improvement.
        x_final, score_final = local_search(best_x, max_iters=10000)
        if score_final < best_score:
            best_score = score_final
            best_x = x_final

    return best_x
