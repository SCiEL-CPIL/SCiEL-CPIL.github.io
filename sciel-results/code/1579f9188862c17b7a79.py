import numpy as np
import time

N = 400
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    The bound is dx * max_k |conv[k]| where conv is the linear autoconvolution.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n
    
    # Pad for linear convolution
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h
    
    # FFT size for linear convolution
    fft_size = 1
    while fft_size < 2 * (2 * n - 1):
        fft_size *= 2
        
    f_fft = np.fft.fft(h_pad, n=fft_size)
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real
    
    max_abs = np.max(np.abs(conv))
    c3_bound = dx * max_abs
    
    return c3_bound

def _generate_rule90(seed, steps, length):
    """
    Generate a binary sequence using Rule 90 cellular automaton.
    Start with a single seed bit, evolve for 'steps' iterations.
    Map the final state to [-1, 1] and truncate/pad to 'length'.
    """
    # Rule 90: new_bit = left_bit XOR right_bit
    # We maintain an array of bits.
    # To get a sequence of length 'length', we need enough CA steps.
    # The width expands by 2 each step.
    # After k steps, width is 2*k + 1.
    # We need 2*steps + 1 >= length, so steps >= (length-1)/2.
    # However, the prompt suggests running for k steps from a single seed.
    # Let's interpret this as: start with a row of length L, or start with 1 bit and let it grow.
    # "run for k steps from a single seed" usually implies the growth pattern.
    
    # To get a specific length N, we can simulate a larger field and take the center.
    # Or simply simulate enough steps such that the width covers N.
    
    # Let's simulate on a large enough grid.
    # Max width needed is N.
    # Steps needed: ceil((N-1)/2).
    # But we can also choose 'steps' as a parameter to vary the complexity.
    # The prompt says "run for k steps". It doesn't fix k.
    # We will try different k values.
    
    # Initialize CA state
    # To be deterministic and reproducible for a given seed index:
    # We'll use the seed to pick an initial pattern or just use the standard single-bit start.
    # "from a single seed" -> single bit start.
    
    # We need to generate a sequence of length N.
    # Let's simulate for enough steps to cover width N.
    # Let width = N.
    # We need to run for enough steps so that the signal has propagated.
    # Actually, Rule 90 from a single bit creates a Sierpinski triangle.
    # After t steps, the non-zero bits are at positions that are sums of distinct powers of 2? No, that's not quite right for the XOR structure.
    # The state at time t is determined by the binomial coefficients mod 2.
    # bit(i, t) = C(t, i) mod 2 (if we start with bit at 0).
    # We can compute this directly or simulate.
    # Simulation is O(steps * width).
    
    # Let's simulate.
    # We need a width that covers N.
    # Let's use a width of N + 2*steps to be safe, or just N if we center it.
    # Let's just simulate for 'steps' on a width of N.
    # But if steps is small, the pattern might be sparse.
    # If steps is large, it fills up.
    
    # Strategy:
    # 1. Choose a number of steps k.
    # 2. Initialize an array of zeros of size N.
    # 3. Place the seed at the center (N//2).
    # 4. Evolve for k steps.
    # 5. Map 0->-1, 1->1 (or similar).
    
    # However, if k is small, most of the array is 0.
    # This creates a lot of -1s (if 0 maps to -1).
    # This might not be a good starting point for optimization.
    
    # Alternative interpretation:
    # Maybe the "seed" is a random initial state, and we run Rule 90 for k steps?
    # "from a single seed" strongly implies a single bit.
    
    # Let's try generating the Sierpinski triangle explicitly.
    # The triangle has height k.
    # We need to map it to length N.
    # If we take the bottom row of a triangle of height k, it has length 2k+1.
    # We can pick k such that 2k+1 is close to N, or pad/truncate.
    
    # Let's generate a sequence of length N based on Rule 90 evolution.
    # We'll simulate k steps on a width of N.
    # To ensure the "fractal" structure is visible, k should be reasonably large.
    # Let's try k = N // 2.
    
    state = np.zeros(N, dtype=np.uint8)
    state[N // 2] = 1
    
    for _ in range(steps):
        # Rule 90: new[i] = state[i-1] XOR state[i+1]
        # Boundaries: treat as 0 (or wrap? No, standard CA usually has boundaries).
        # Let's use zero padding.
        new_state = np.zeros(N, dtype=np.uint8)
        new_state[1:] = state[:-2] ^ state[2:]
        state = new_state
        
    # Map bits to signs
    # 0 -> -1, 1 -> 1
    signs = 2.0 * state - 1.0
    return signs

def _local_search(h, max_iter=2000, deadline=None):
    """
    Greedy local search: try flipping individual bits to reduce the C3 bound.
    Uses efficient delta computation if possible, otherwise recomputes.
    For N=400, recomputing FFT is fast enough for a few thousand iterations.
    """
    current_bound = _compute_c3_bound(h)
    improved = True
    iters = 0
    
    while improved and iters < max_iter:
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        iters += 1
        best_gain = 0
        best_idx = -1
        
        # Try flipping each position
        # Optimization: We can compute the change in autoconvolution more efficiently,
        # but for simplicity and robustness, we recompute.
        # N=400, FFT is ~microseconds. 400 flips * 2000 iters = 800k FFTs.
        # Might be too slow. Let's estimate: 1 FFT ~ 10us? 800k * 10us = 8s.
        # This is borderline. We should limit iterations or use a smarter method.
        
        # Let's use a smarter method: compute the gradient/delta for all i at once?
        # Flipping h[i] changes the autoconvolution.
        # Let h' = h with h[i] flipped.
        # h' = h - 2*h[i]*e_i
        # Autoconvolution C' = (h') * (h') = (h - 2h_i e_i) * (h - 2h_i e_i)
        # = h*h - 4 h_i (h * e_i) + 4 h_i^2 e_i * e_i
        # h * e_i is just h shifted.
        # e_i * e_i is a delta at 2i? No, convolution of impulse with impulse is impulse.
        # Wait, autoconvolution is h * h.
        # If we flip index i, h_new = h_old except h_new[i] = -h_old[i].
        # Let d = h_new - h_old. d has -2*h_old[i] at i, 0 elsewhere.
        # C_new = (h+d)*(h+d) = h*h + 2*h*d + d*d.
        # h*d = h * (-2 h_i e_i) = -2 h_i * (h shifted by i).
        # d*d = (-2 h_i e_i) * (-2 h_i e_i) = 4 h_i^2 e_i * e_i = 4 e_i * e_i.
        # e_i * e_i is an impulse at position 2i?
        # Convolution of delta_i and delta_i is delta_{2i}.
        # So C_new = C_old - 4 h_i * (h shifted by i) + 4 delta_{2i}.
        
        # This allows O(N) update per flip? No, "h shifted by i" is the whole vector h shifted.
        # But we can precompute the shifts?
        # Actually, we just need to update the max.
        # This is complex to implement correctly for the max norm.
        
        # Let's stick to recomputation but reduce iterations or use random sampling of flips.
        # Or, just do a few full passes.
        
        # Let's try a random subset of flips per iteration to speed up.
        # Or just accept the cost and limit max_iter to 500.
        
        for i in range(N):
            old_val = h[i]
            h[i] = -old_val
            new_bound = _compute_c3_bound(h)
            h[i] = old_val
            
            gain = current_bound - new_bound
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx >= 0 and best_gain > 1e-9:
            h[best_idx] = -h[best_idx]
            current_bound -= best_gain
            improved = True
        else:
            improved = False
            
    return h, current_bound

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_h = None
    best_bound = float('inf')
    
    # Strategy 1: Rule 90 based initialization
    # Try different numbers of steps for Rule 90
    # The prompt suggests leveraging the fractal structure.
    # We'll try a few values of 'steps'.
    
    rule90_steps_options = [N // 4, N // 2, N // 3, 10, 20, 50]
    
    for steps in rule90_steps_options:
        if time.time() > deadline - 10:
            break
            
        h = _generate_rule90(0, steps, N)
        # Randomize signs slightly? No, keep it deterministic first.
        # Maybe flip a random subset to break symmetry?
        # Let's just use it as is.
        
        bound = _compute_c3_bound(h)
        
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
            
    # Strategy 2: Random initialization
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        h = np.random.choice([-1.0, 1.0], size=N)
        bound = _compute_c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()

    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        best_bound = _compute_c3_bound(best_h)

    # Refinement: Local Search
    # We have ~100s left.
    # Local search is expensive. Let's run it on the best candidate.
    # We might run it on multiple candidates if time permits.
    
    candidates = []
    # Add the best one
    candidates.append(best_h.copy())
    
    # Add a few random ones for diversity
    for _ in range(3):
        if time.time() > deadline - 20:
            break
        h = np.random.choice([-1.0, 1.0], size=N)
        candidates.append(h)
        
    # Also add a few Rule 90 variants
    for steps in [N//2, N//4]:
        if time.time() > deadline - 20:
            break
        h = _generate_rule90(0, steps, N)
        candidates.append(h)

    # Run local search on each candidate
    # Limit iterations per candidate to ensure we finish.
    # Let's say 200 iterations per candidate.
    # 4 candidates * 200 iters * 400 flips * 10us/FFT = 32s.
    # This is tight. Let's reduce to 100 iterations.
    
    for i, h in enumerate(candidates):
        if time.time() > deadline - 5:
            break
            
        # Adjust max_iter based on remaining time
        remaining = deadline - 5 - time.time()
        # Estimate time per iteration: ~400 flips * 10us = 4ms
        # 100 iters = 0.4s
        # So we can afford many more.
        # Let's try 500 iterations.
        h_opt, bound_opt = _local_search(h, max_iter=500, deadline=deadline - 5)
        
        if bound_opt < best_bound:
            best_bound = bound_opt
            best_h = h_opt.copy()

    # Final check
    final_bound = _compute_c3_bound(best_h)
    if final_bound < best_bound:
        best_bound = final_bound

    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    print(f"C3 bound: {_compute_c3_bound(h)}")
