import numpy as np
import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    n = 24
    sqrt_n = np.sqrt(n)
    
    # Precompute grid
    num_points = 1024
    thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    z = np.exp(1j * thetas)
    
    # Precompute powers matrix Z (num_points x n)
    Z = np.ones((num_points, n), dtype=np.complex128)
    for k in range(1, n):
        Z[:, k] = Z[:, k-1] * z
        
    # Function to refine peak
    def refine_peak(coeffs):
        # coeffs is a list or array of 1s and -1s
        def f(theta):
            z_point = np.exp(1j * theta)
            p_val = 0.0
            zp = 1.0 + 0.0j
            for c in coeffs:
                p_val += c * zp
                zp *= z_point
            return np.abs(p_val)
        
        # Find coarse max index
        vals = Z @ coeffs
        abs_vals = np.abs(vals)
        idx = np.argmax(abs_vals)
        theta_max = thetas[idx]
        
        # Golden section
        delta = 2 * np.pi / num_points
        a = theta_max - delta
        b = theta_max + delta
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = f(c)
        fd = f(d)
        
        for _ in range(20):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
        
        return f((a + b) / 2)

    # Keep track of top K candidates
    K = 100
    # Heap stores (max_val, mask)
    # We want min-heap of max_vals, so we store -max_val?
    # Or just keep a list and sort at the end?
    # 16M items is too big for a list of tuples in memory if we store all.
    # Use a heap of size K.
    
    # To speed up coeffs generation, we can precompute the coeffs for all masks?
    # No, memory.
    
    # We can iterate and update the heap.
    # best_heap = [] # stores (-max_val, mask)
    
    # Optimization: 
    # Instead of generating coeffs array every time, we can compute the dot product differently?
    # Z @ coeffs.
    # coeffs[k] = 1 - 2*bit_k.
    # Sum Z[:,k] * (1 - 2*bit_k) = Sum Z[:,k] - 2 * Sum (bit_k * Z[:,k])
    # Let S = Sum Z[:,k] for all k (vector of length num_points).
    # Then P(z) = S - 2 * Sum_{k: bit_k=1} Z[:,k].
    # This allows us to precompute S.
    # And we can maintain a running sum?
    # As we iterate masks from 0 to 2^24-1, the bits change.
    # This is like a Gray code or just standard binary.
    # In standard binary, many bits change.
    
    # Let's try the direct Z @ coeffs first. It's simple.
    
    # To make it faster, we can use a smaller grid for the main loop and refine only top K.
    # Let's use 256 points for the main loop.
    
    num_points_coarse = 256
    thetas_coarse = np.linspace(0, 2 * np.pi, num_points_coarse, endpoint=False)
    z_coarse = np.exp(1j * thetas_coarse)
    Z_coarse = np.ones((num_points_coarse, n), dtype=np.complex128)
    for k in range(1, n):
        Z_coarse[:, k] = Z_coarse[:, k-1] * z_coarse
        
    S = np.sum(Z_coarse, axis=1) # Sum of all columns
    
    # We want to minimize max |P|.
    # P = S - 2 * Sum_{k in OnBits} Z_coarse[:, k]
    
    # We will iterate masks.
    # To speed up, we can precompute the contribution of each bit?
    # No, we just loop.
    
    best_heap = [] # (max_val, mask)
    
    # Check time budget
    # 16M iterations.
    # Each iteration:
    # 1. Identify bits set.
    # 2. Sum the corresponding columns of Z_coarse.
    # 3. Compute P = S - 2 * Sum.
    # 4. Compute Max.
    # 5. Push to heap.
    
    # Step 2 is the bottleneck. Summing 24 complex vectors of length 256.
    # We can precompute the sum for each bit?
    # Sum_bits = np.zeros((n, num_points_coarse), dtype=np.complex128)
    # Sum_bits[k] = Z_coarse[:, k]
    # Then for a mask, we sum Sum_bits[k] for all k where mask has bit k.
    
    # We can use a lookup table for 12-bit chunks?
    # Precompute sum for all 2^12 masks for the first 12 bits and last 12 bits.
    # Sum_first12[mask12] = Sum_{k in mask12} Z_coarse[:, k]
    # Sum_last12[mask12] = Sum_{k in mask12} Z_coarse[:, 12+k]
    
    # This reduces the sum to 2 lookups and 1 addition.
    
    half_n = n // 2
    size_half = 1 << half_n
    
    Sum_first = np.zeros((size_half, num_points_coarse), dtype=np.complex128)
    Sum_last = np.zeros((size_half, num_points_coarse), dtype=np.complex128)
    
    # Fill Sum_first
    # Sum_first[0] = 0
    # For m in 1..size_half-1:
    #   lsb = m & -m
    #   idx = bit_length(lsb) - 1
    #   prev = m ^ lsb
    #   Sum_first[m] = Sum_first[prev] + Z_coarse[:, idx]
    
    for m in range(1, size_half):
        lsb = m & -m
        idx = lsb.bit_length() - 1
        prev = m ^ lsb
        Sum_first[m] = Sum_first[prev] + Z_coarse[:, idx]
        
    for m in range(1, size_half):
        lsb = m & -m
        idx = lsb.bit_length() - 1
        prev = m ^ lsb
        Sum_last[m] = Sum_last[prev] + Z_coarse[:, half_n + idx]

    # Now iterate all masks
    # mask = m1 | (m2 << half_n)
    
    for m1 in range(size_half):
        s1 = Sum_first[m1]
        for m2 in range(size_half):
            if time.time() > deadline:
                break
            s2 = Sum_last[m2]
            # P = S - 2 * (s1 + s2)
            p_vals = S - 2.0 * (s1 + s2)
            max_val = np.max(np.abs(p_vals))
            
            # Push to heap
            if len(best_heap) < K:
                heapq.heappush(best_heap, (max_val, (m1, m2)))
            elif max_val < best_heap[0][0]:
                heapq.heapreplace(best_heap, (max_val, (m1, m2)))
    
    if time.time() > deadline:
        # If we broke out early, we might have missed the best.
        # But we have a heap of the best seen so far.
        pass

    # Refine the top K candidates
    best_val = float('inf')
    best_mask = 0
    
    # Sort heap by value
    candidates = sorted(best_heap, key=lambda x: x[0])
    
    for val, (m1, m2) in candidates:
        if time.time() > deadline - 2.0:
            break
            
        mask = m1 | (m2 << half_n)
        # Reconstruct coeffs
        coeffs = np.ones(n, dtype=np.float64)
        for k in range(n):
            if (mask >> k) & 1:
                coeffs[k] = -1.0
        
        refined_val = refine_peak(coeffs)
        
        if refined_val < best_val:
            best_val = refined_val
            best_mask = mask

    # Return coeffs for best_mask
    final_coeffs = np.ones(n, dtype=int)
    for k in range(n):
        if (best_mask >> k) & 1:
            final_coeffs[k] = -1
            
    return final_coeffs.tolist()
