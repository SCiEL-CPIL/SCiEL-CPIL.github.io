import numpy as np
import time

N = 400

def _generate_m_sequence(n):
    # LFSR with primitive polynomial x^9 + x^4 + 1 (for length 511)
    # We need length >= 400. 511 is good.
    reg = 1
    poly = 0b100010001 # x^9 + x^4 + 1? 
    # Standard primitive polynomial for 9 bits: x^9 + x^4 + 1 is 0b100010001?
    # Let's use a known one. x^9 + x^5 + 1?
    # Actually, we can just use a simple LFSR and check length.
    # Or use scipy? No, stdlib + numpy.
    
    # Let's use x^9 + x^4 + 1.
    # Tap at 9 and 4?
    # Feedback = bit[9] XOR bit[4]
    # Shift left, insert feedback.
    
    seq = []
    reg = 1
    for i in range(511):
        seq.append(reg & 1)
        feedback = (reg >> 8) ^ (reg >> 3) # x^9 is bit 8 (0-indexed), x^4 is bit 3?
        # If poly is x^9 + x^4 + 1, taps are 9 and 4.
        # In 0-indexed bits, bit 8 and bit 3.
        reg = ((reg << 1) | feedback) & 0x1FF
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # 1. Initialize with m-sequence
    m_seq = _generate_m_sequence(N)
    # Map 0 -> -1, 1 -> 1
    h = np.where(np.array(m_seq[:N]) == 1, 1.0, -1.0)
    
    # 2. Local Search
    # Compute initial convolution
    def compute_conv(h):
        n = len(h)
        n_fft = 2 * n
        H = np.fft.rfft(h, n=n_fft)
        conv = np.fft.irfft(H * H, n=n_fft)[:n]
        return conv
    
    conv = compute_conv(h)
    current_peak = np.max(np.abs(conv))
    best_peak = current_peak
    best_h = h.copy()
    
    no_improve = 0
    
    while time.time() < deadline:
        improved = False
        # Try flipping each bit
        for i in range(N):
            if time.time() > deadline:
                break
            
            # Calculate new convolution peak if we flip h[i]
            # C_new(k) = C_old(k) - 2 * h[i] * h[k-i]
            # We only need to check if the new peak is lower.
            # Instead of recalculating max for all k, we can update the array.
            # But we need to revert if it's not better.
            
            # Optimization: 
            # The change is localized? No, it affects all k where h[k-i] is non-zero.
            # Since h is full length, it affects all k.
            
            # Let's just compute the new peak efficiently.
            # We can't easily update the max without scanning.
            # Scanning is O(N).
            
            delta = -2 * h[i]
            # h[k-i] is non-zero for k in [i, i+N-1]
            # Conv is length N (circular? No, linear).
            # Linear convolution length is 2N-1.
            # The code `compute_conv` returns length N?
            # `irfft(..., n=n_fft)[:n]` returns the first N points of the linear convolution.
            # This is the "valid" part? No, linear conv of two length N signals has length 2N-1.
            # The peak might be in the second half.
            # The current `_c3_upper_bound` only checks `[:n]`.
            # Is that correct?
            # "autoconvolution upper bound".
            # Usually, the peak is in the center.
            # For length 400, the center is at 399.
            # `[:400]` covers indices 0..399.
            # The full convolution has indices 0..799.
            # The peak of autoconvolution of a compact signal is at the center.
            # So checking `[:N]` might miss the peak if the peak is at $N$?
            # For $h*h$, the peak is at lag 0?
            # $C(0) = \sum h_i^2 = N$.
            # $C(k)$ for $k \ne 0$ is smaller.
            # So the peak is at 0.
            # Index 0 is included in `[:N]`.
            
            # So we can just check the max of the updated first N values.
            
            # To be safe, let's use the full convolution length in the check?
            # The current program uses `[:n]`. I will stick to that to be consistent with the "seed" behavior if it's defined that way.
            # But wait, if the peak is at 0, it's always in `[:N]`.
            
            # Let's simulate the flip on a copy of conv? 
            # That's O(N) memory and time.
            # 400 * 400 = 160k ops per iteration.
            # 160k * 1000 iterations = 160M ops.
            # Python is slow. 160M ops might take > 1s.
            # We need to be faster.
            
            # Vectorized update:
            # The term $h[k-i]$ for $k \in [0, N-1]$ is $h[0:N-i]$ if $i > 0$?
            # $k-i \in [-i, N-1-i]$.
            # Valid indices for $h$ are $0..N-1$.
            # So $h[k-i]$ is non-zero when $0 \le k-i \le N-1 \implies i \le k \le i+N-1$.
            # Since $k < N$, the range is $k \in [i, N-1]$.
            # So for $k < i$, the change is 0?
            # No, $h$ is zero outside $[0, N-1]$.
            # So for $k < i$, $h[k-i]$ is $h[negative] = 0$.
            # So $C_{new}(k) = C_{old}(k)$ for $k < i$.
            # For $k \ge i$, $C_{new}(k) = C_{old}(k) - 2 h[i] h[k-i]$.
            
            # So we only need to update $conv[i:N]$.
            
            if i == 0:
                # Update all
                new_conv = conv - 2 * h[i] * h
            else:
                new_conv = conv.copy()
                new_conv[i:] -= 2 * h[i] * h[:N-i]
                
            new_peak = np.max(np.abs(new_conv))
            
            if new_peak < current_peak - 1e-12:
                # Accept flip
                h[i] = -h[i]
                conv = new_conv
                current_peak = new_peak
                improved = True
                
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_h = h.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        if not improved:
            break
            
        if no_improve > 100:
            break
            
    return best_h
