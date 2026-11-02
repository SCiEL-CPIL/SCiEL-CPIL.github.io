"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Minimizes the Peak Third-Order Autocorrelation.
    C3(k1, k2) = sum_i f(i) f(i+k1) f(i+k2)
    Metric: max |C3(k1, k2)| for k1, k2 not both 0.
    """
    np.random.seed(42)
    
    def calculate_c3_max(seq):
        """
        Calculate the maximum absolute value of the aperiodic third-order autocorrelation.
        Uses a loop over k1 and vectorized correlation for each k1.
        """
        max_val = 0.0
        # k1 ranges from -(N-1) to N-1
        for k1 in range(-(N-1), N):
            # Compute y[i] = seq[i] * seq[i+k1]
            # Handle boundaries: valid i are such that 0 <= i < N and 0 <= i+k1 < N
            start = max(0, -k1)
            end = min(N, N - k1)
            if start >= end:
                continue
            
            # y is the element-wise product of seq[start:end] and seq[start+k1:end+k1]
            # Note: seq[i+k1] corresponds to indices start+k1 to end+k1
            y = seq[start:end] * seq[start+k1:end+k1]
            
            # Now compute correlation of y with seq
            # C3(k1, k2) = sum_i y[i] seq[i+k2]
            # This is the cross-correlation of y and seq at lag k2.
            # y has length L = end - start.
            # We need the aperiodic correlation.
            # np.correlate(y, seq, mode='full')
            # The result length is L + N - 1.
            # The lag k2 corresponds to index ...
            # Let's use direct summation via np.einsum or simple dot products for speed?
            # N=400, L~400. 400*400 = 160k ops per k1.
            # 800 k1 values. 128M ops total.
            # This might be slow in Python loop.
            
            # Vectorized approach for this k1:
            # We want R[k2] = sum_i y[i] seq[i+k2]
            # This is the convolution/correlation.
            # Using FFT is faster for large N, but N=400 is small.
            # np.correlate is C-optimized.
            corr = np.correlate(y, seq, mode='full')
            
            # The correlation lags k2 range from -(N-1) to N-1.
            # In 'full' mode, index 0 is lag -(N-1) (if y is shorter? No, full length is Ny+Ny-1).
            # Actually, np.correlate(a, v, mode='full')
            # c_k = sum_n a_{n+k} * v_n
            # Here a=y, v=seq.
            # We want sum_i y[i] seq[i+k2].
            # This matches c_{-k2} if we define c_k = sum y[i+k] seq[i]?
            # Let's just take the max abs of the whole array, as we want max over all valid k2.
            # The valid k2 are those where the overlap is non-zero.
            # np.correlate handles the aperiodic case correctly with zero padding implicitly?
            # No, 'full' mode assumes zero padding outside the array bounds.
            # So corr contains the aperiodic correlations.
            
            local_max = np.max(np.abs(corr))
            if local_max > max_val:
                max_val = local_max
                
        return max_val

    # Initial random sequence
    f = np.random.choice([-1, 1], size=N)
    current_cost = calculate_c3_max(f)
    
    start_time = time.time()
    max_time = 100.0
    
    # Simple Hill Climbing
    # Since cost evaluation is ~20-50ms, we can do ~2000-5000 evals.
    # We will do a few passes.
    
    improved = True
    pass_count = 0
    while improved and (time.time() - start_time) < max_time:
        improved = False
        pass_count += 1
        indices = np.random.permutation(N)
        
        for i in indices:
            if (time.time() - start_time) > max_time:
                break
            f[i] = -f[i]
            new_cost = calculate_c3_max(f)
            
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
            else:
                f[i] = -f[i]
                
        if not improved:
            # Restart if stuck and time allows
            if current_cost > 10 and (time.time() - start_time) < max_time * 0.8:
                f = np.random.choice([-1, 1], size=N)
                current_cost = calculate_c3_max(f)
            else:
                break

    return f
# EVOLVE-BLOCK-END
