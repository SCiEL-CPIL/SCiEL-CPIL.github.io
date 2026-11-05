"""
Optimized solver for the Third Autocorrelation Inequality task.
Uses Simulated Annealing with incremental autoconvolution updates.
"""
import numpy as np
import time

N = 400
DX = 0.5 / N  # Length of each interval

def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    
    # Efficient linear autoconvolution using FFT
    # Pad to length 2n-1 (or next power of 2 for FFT efficiency)
    fft_len = 1
    while fft_len < 2 * n - 1:
        fft_len *= 2
        
    h_fft = np.fft.fft(h, n=fft_len)
    conv = np.fft.ifft(h_fft * np.conj(h_fft)).real
    conv = conv[:2*n-1]
    
    max_abs = np.max(np.abs(conv))
    c3_bound = DX * max_abs
    return c3_bound

def _init_conv(h):
    """
    Initialize the autoconvolution array C for a given h.
    C[k] = sum_{i} h[i] h[i+k] for k in -(N-1)..(N-1)
    We store C in an array of size 2N-1, indexed by k+N-1.
    """
    n = len(h)
    fft_len = 1
    while fft_len < 2 * n - 1:
        fft_len *= 2
        
    h_fft = np.fft.fft(h, n=fft_len)
    conv = np.fft.ifft(h_fft * np.conj(h_fft)).real
    return conv[:2*n-1]

def _simulate_annealing(h_init, time_budget):
    """
    Run Simulated Annealing to minimize the peak autoconvolution.
    """
    n = N
    h = h_init.copy()
    C = _init_conv(h)
    
    # Objective function: max absolute value of C
    current_peak = np.max(np.abs(C))
    best_h = h.copy()
    best_peak = current_peak
    
    # Schedule parameters
    # Start high enough to accept most moves, end low to freeze
    T_start = 0.1
    T_end = 0.001
    # We don't know exact number of iterations in advance due to time check
    # But we can estimate based on speed. Let's just use a cooling rate per step.
    # A typical SA needs ~1000-10000 steps for N=400 to converge.
    # Let's aim for a dynamic number of steps based on time.
    
    start_time = time.time()
    step = 0
    
    # Precompute indices for faster access?
    # The update for flipping index i:
    # C_new = C_old - 2 * h[i] * (h shifted)
    # Specifically, C[k] depends on h[i] and h[i+k].
    # Flipping h[i] to -h[i]:
    # New term for pair (i, j): (-h[i]) * h[j] = - h[i] h[j]
    # Old term: h[i] h[j]
    # Change: -2 h[i] h[j]
    # This affects C at lag j-i.
    # So C[k] for all k is affected? No.
    # C[k] = sum_i h[i] h[i+k].
    # If we flip h[i], only terms involving h[i] change.
    # Terms are (i, i+k) and (i-k, i).
    # So C[k] changes by -2 h[i] h[i+k] (from term i)
    # And C[-k] changes by -2 h[i-k] h[i] (from term i-k).
    # Actually, it's easier to think:
    # C_new = C_old - 2 h[i] (h * e_i) 
    # Where (h * e_i)[k] = h[i-k] ? No, correlation.
    # Let's derive:
    # C[k] = sum_j h[j] h[j+k]
    # If h[i] -> -h[i]:
    # 1. Term j=i: h[i]h[i+k] becomes -h[i]h[i+k]. Change: -2 h[i] h[i+k].
    # 2. Term j=i-k: h[i-k]h[i] becomes -h[i-k]h[i]. Change: -2 h[i-k] h[i].
    # So C[k] -= 2 h[i] h[i+k]
    # And C[-k] -= 2 h[i-k] h[i]
    # Note that C[-k] is at index (N-1-k). C[k] is at index (N-1+k).
    # h[i+k] is only valid if 0 <= i+k < N.
    # h[i-k] is only valid if 0 <= i-k < N.
    
    # Optimization: We can update C in O(N) by slicing.
    # Let v = h[i]
    # For lags k such that i+k is valid: C[k] -= 2*v*h[i+k]
    # For lags k such that i-k is valid: C[-k] -= 2*v*h[i-k]
    
    # Let's implement the update carefully.
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Cooling: exponential decay
        # We need a total number of steps to define cooling.
        # Let's estimate max steps: ~100 steps/ms * 1000ms * budget_s ?
        # Let's just use a fixed cooling rate per step, assuming ~10000 steps.
        # If we run fewer steps, T stays higher, which is fine (more exploration).
        # If we run more, T drops.
        # Let's assume ~20000 steps total for the budget.
        # T = T_start * (T_end/T_start)^(step / total_steps_est)
        total_steps_est = 20000
        if step >= total_steps_est:
            # If we exceed estimate, keep T at T_end
            T = T_end
        else:
            T = T_start * (T_end / T_start) ** (step / total_steps_est)
        
        # Pick a random index to flip
        i = np.random.randint(n)
        
        # Calculate delta E
        # We need to update C and find new max.
        # To avoid full max scan O(N), we can track current max.
        # But max can decrease or increase.
        # Simple approach: Update C, compute new max. O(N) for update, O(N) for max.
        # Total O(N) per step.
        
        old_val = h[i]
        # Flip
        h[i] = -old_val
        
        # Update C
        # 1. Terms where h[i] is the first element: j=i.
    #    C[k] involves h[i]h[i+k]. Change -2 h[i] h[i+k].
    #    Valid k: -i <= k <= n-1-i.
    #    Indices in C array: k + n - 1.
    #    Range: n-1-i to 2n-2-i.
    #    Values to subtract: 2 * old_val * h[i : i + (n-1-i) + 1] -> h[i:n]
    #    Wait, h[i+k] for k in [0, n-1-i].
    #    So we subtract 2*old_val*h[i:n] from C[n-1-i : n-1-i + (n-i)]
        
    #    2. Terms where h[i] is the second element: j=i-k.
    #    C[k] involves h[i-k]h[i]. Change -2 h[i-k] h[i].
    #    Valid k: -(i) <= k <= 0 ? No.
    #    j = i-k >= 0 => k <= i.
    #    j = i-k < n => k > i-n.
    #    So k in [i-n+1, i].
    #    Indices in C: i-n+1+n-1 to i+n-1 => i to i+n-1.
    #    Values: h[i-k] for k in [i-n+1, i].
    #    Let m = i-k. m goes from n-1 down to 0.
    #    So we subtract 2*old_val*h[0:i+1] (reversed? No)
    #    C[k] for k=i-m. Index k+n-1 = i-m+n-1.
    #    We want to subtract 2*old_val*h[m] from C[i-m+n-1].
    #    As m goes 0..i, index goes i+n-1 down to n-1.
    #    So C[n-1 : i+n] corresponds to m=i..0.
    #    Values h[0:i+1] reversed?
    #    m=0 -> h[0] at index i+n-1.
    #    m=i -> h[i] at index n-1.
    #    So we subtract 2*old_val * h[0:i+1][::-1] from C[n-1 : i+n].
        
        # Let's implement this vectorized update.
        
        # Part 1: j=i
        # k from 0 to n-1-i
        # C indices: n-1 to n-1 + (n-1-i) = 2n-2-i
        # Values: h[i : n]
        start_idx1 = n - 1
        end_idx1 = 2 * n - 2 - i + 1
        C[start_idx1:end_idx1] -= 2.0 * old_val * h[i:n]
        
        # Part 2: j=i-k (k from -(i) to ... wait)
        # k such that i-k is valid index.
        # k = i - m. m in 0..i.
        # k in i-i=0 ... i-0=i.
        # Wait, k can be negative?
        # j = i-k. If k is negative, j > i.
        # In Part 1, we covered j=i. The term is h[i]h[i+k].
        # In Part 2, we cover terms where the second factor is h[i].
        # Term is h[j]h[j+k] = h[i-k]h[i].
        # This term exists in C[k].
        # j = i-k.
        # If k > 0, j < i.
        # If k < 0, j > i.
        # If k = 0, j = i. (Covered in Part 1? Yes, h[i]h[i].)
        # So Part 2 should cover k != 0?
        # Actually, C[k] = sum_j h[j]h[j+k].
        # The term h[i] appears in:
        # 1. j=i: h[i]h[i+k]. (Part 1)
        # 2. j+k=i => j=i-k: h[i-k]h[i]. (Part 2)
        
        # For Part 2:
        # k ranges such that 0 <= i-k < N.
        # i-N < k <= i.
        # Since k is integer, k in [i-N+1, i].
        # But we already handled k=0 in Part 1?
        # In Part 1, k=0 term is h[i]h[i].
        # In Part 2, k=0 term is h[i]h[i].
        # So if we include k=0 in both, we subtract 2*2*h[i]^2 = 4.
        # But the total change is -2 h[i] h[i] - 2 h[i] h[i] = -4?
        # Original: h[i]^2. New: (-h[i])^2 = h[i]^2. Change 0?
        # No, wait.
        # C[0] = sum h[j]^2 = N.
        # Flipping h[i] does not change h[i]^2.
        # So C[0] should NOT change.
        # My formula:
        # Part 1 k=0: -2 h[i] h[i] = -2.
        # Part 2 k=0: -2 h[i] h[i] = -2.
        # Total change -4.
        # But C[0] is sum of squares. It doesn't change.
        # So I must exclude k=0 from one of them, or handle it carefully.
        # Actually, the term h[i]h[i] is counted once in the sum for C[0].
        # When we flip h[i], h[i]^2 is invariant.
        # So the change for C[0] is 0.
        # My derivation:
        # C_new[0] = sum_{j!=i} h[j]^2 + (-h[i])^2 = sum_{j!=i} h[j]^2 + h[i]^2 = C_old[0].
        # So C[0] is unchanged.
        # Therefore, I should NOT update C[0] (index N-1).
        
        # Let's adjust ranges to exclude k=0.
        
        # Part 1: k > 0.
        # k in 1 .. n-1-i.
        # Indices: n .. 2n-2-i.
        # If i = n-1, range is empty.
        if i < n - 1:
            start_idx1 = n
            end_idx1 = 2 * n - 2 - i + 1
            C[start_idx1:end_idx1] -= 2.0 * old_val * h[i+1:n] # h[i+k] for k>=1
            
        # Part 2: k < 0.
        # k in i-n+1 .. -1.
        # Indices: i .. n-2.
        # Values: h[i-k] for k in [i-n+1, -1].
        # Let m = i-k. k=-1 => m=i+1. k=i-n+1 => m=n-1.
        # So m in i+1 .. n-1.
        # C index for k: k+n-1.
        # k=-1 => n-2.
        # k=i-n+1 => i.
        # So C[i : n-1] corresponds to m=n-1 down to i+1.
        # Values h[m] for m=n-1..i+1.
        # So we subtract 2*old_val * h[i+1:n][::-1] from C[i : n-1].
        
        if i > 0:
            start_idx2 = i
            end_idx2 = n - 1
            # h[i+1:n] is h[i+1], ..., h[n-1].
            # We need h[n-1], ..., h[i+1].
            C[start_idx2:end_idx2] -= 2.0 * old_val * h[i+1:n][::-1]
            
        # Calculate new peak
        new_peak = np.max(np.abs(C))
        delta_E = new_peak - current_peak
        
        # Acceptance criterion
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            # Accept
            current_peak = new_peak
            if new_peak < best_peak:
                best_peak = new_peak
                best_h = h.copy()
        else:
            # Reject: revert h and C
            h[i] = old_val
            # Revert C
            if i < n - 1:
                start_idx1 = n
                end_idx1 = 2 * n - 2 - i + 1
                C[start_idx1:end_idx1] += 2.0 * old_val * h[i+1:n]
            if i > 0:
                start_idx2 = i
                end_idx2 = n - 1
                C[start_idx2:end_idx2] += 2.0 * old_val * h[i+1:n][::-1]
            # current_peak remains old_peak
        
        step += 1
        
    return best_h, best_peak

def _local_search_polish(h_init, time_budget):
    """
    Deterministic local search to polish the result.
    """
    h = h_init.copy()
    C = _init_conv(h)
    current_peak = np.max(np.abs(C))
    
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_budget:
        improved = False
        best_gain = 0
        best_i = -1
        
        # Sample a subset of indices to try, or all if fast enough
        # N=400, O(N) per check. 400*400 = 160k ops per pass. Very fast.
        # Let's try all indices.
        for i in range(N):
            old_val = h[i]
            h[i] = -old_val
            
            # Update C
            if i < N - 1:
                C[N : 2*N-2-i+1] -= 2.0 * old_val * h[i+1:N]
            if i > 0:
                C[i : N-1] -= 2.0 * old_val * h[i+1:N][::-1]
            
            new_peak = np.max(np.abs(C))
            gain = current_peak - new_peak
            
            # Revert for now to check others
            h[i] = old_val
            if i < N - 1:
                C[N : 2*N-2-i+1] += 2.0 * old_val * h[i+1:N]
            if i > 0:
                C[i : N-1] += 2.0 * old_val * h[i+1:N][::-1]
                
            if gain > best_gain:
                best_gain = gain
                best_i = i
        
        if best_i >= 0:
            # Apply best move
            old_val = h[best_i]
            h[best_i] = -old_val
            if best_i < N - 1:
                C[N : 2*N-2-best_i+1] -= 2.0 * old_val * h[best_i+1:N]
            if best_i > 0:
                C[best_i : N-1] -= 2.0 * old_val * h[best_i+1:N][::-1]
            
            current_peak -= best_gain
            improved = True
            
    return h, current_peak

def construct():
    start_time = time.time()
    total_budget = 145  # Safety margin
    
    best_h = None
    best_bound = np.inf
    
    # Strategy: Multiple SA runs with different seeds
    # Estimate time per SA run: ~10-20s
    # We can do ~5-7 runs.
    
    num_runs = 0
    while time.time() - start_time < total_budget - 10:
        # New random initialization
        np.random.seed(42 + num_runs)
        h_init = np.random.choice([-1.0, 1.0], size=N)
        
        # Reserve time for polish
        polish_time = 5
        sa_time = min(20, total_budget - (time.time() - start_time) - polish_time)
        
        if sa_time < 1:
            break
            
        h_sa, peak_sa = _simulate_annealing(h_init, sa_time)
        
        # Polish
        h_final, peak_final = _local_search_polish(h_sa, polish_time)
        
        if peak_final < best_bound:
            best_bound = peak_final
            best_h = h_final
            
        num_runs += 1
        
    # If no SA runs completed, fallback to random
    if best_h is None:
        best_h = np.random.choice([-1.0, 1.0], size=N)
        best_bound = _compute_c3_bound(best_h)
        
    # Final check and return
    # Ensure we return the best found
    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    print(f"C3 bound: {_compute_c3_bound(h)}")
    print(f"Score: {1.0 / _compute_c3_bound(h)}")
