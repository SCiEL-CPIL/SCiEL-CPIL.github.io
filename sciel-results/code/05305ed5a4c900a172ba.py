import numpy as np
import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    n = 24
    half_n = n // 2
    size_half = 1 << half_n  # 4096
    
    # --- Stage 1: Coarse exhaustive search (Vectorized) ---
    # Use a slightly larger grid since vectorization is fast
    num_coarse = 512 
    thetas_coarse = np.linspace(0, 2 * np.pi, num_coarse, endpoint=False)
    z_coarse = np.exp(1j * thetas_coarse)
    Z_coarse = np.ones((num_coarse, n), dtype=np.complex128)
    for k in range(1, n):
        Z_coarse[:, k] = Z_coarse[:, k-1] * z_coarse
    
    S = np.sum(Z_coarse, axis=1)
    
    # Precompute lookup tables
    Sum_first = np.zeros((size_half, num_coarse), dtype=np.complex128)
    Sum_last = np.zeros((size_half, num_coarse), dtype=np.complex128)
    
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
    
    # Exhaustive search with coarse grid
    K1 = 2000
    best_heap = []  # (max_val, m1, m2)
    
    t0 = time.time()
    for m1 in range(size_half):
        if time.time() > deadline - 30.0:
            break
        s1 = Sum_first[m1]
        # Vectorized computation for all m2
        # p[m2, :] = S - 2.0 * (s1 + Sum_last[m2, :])
        p_all = S[None, :] - 2.0 * (s1[None, :] + Sum_last)
        max_vals = np.abs(p_all).max(axis=1)
        
        if len(best_heap) < K1:
            # Push top candidates from this row
            # Sort indices by max_val
            idxs = np.argsort(max_vals)[:min(K1, size_half)]
            for idx in idxs:
                heapq.heappush(best_heap, (max_vals[idx], m1, idx))
        else:
            threshold = best_heap[0][0]
            # Find indices where max_val < threshold
            mask = max_vals < threshold
            if np.any(mask):
                # Get the best few from this row
                idxs_all = np.where(mask)[0]
                # Sort them to get the best first
                idxs_sorted = idxs_all[np.argsort(max_vals[idxs_all])]
                for idx in idxs_sorted:
                    heapq.heappush(best_heap, (max_vals[idx], m1, idx))
                    # Update threshold? The heap handles it, but we can break if we push enough
                    # To avoid pushing too many, we can limit to e.g. 5 per row
                    if len(idxs_sorted) > 5:
                         # Just push the best 5
                         break
                    # Actually, if we push, the heap size grows.
                    # We should use heapreplace if size is full?
                    # The logic above pushes. If heap is full, we should replace.
                    # Let's refine this logic:
                    pass

    # The previous logic had a flaw: if heap is full, we should replace, not push.
    # Let's restart the loop logic with correct heap management.
    
    best_heap = []
    for m1 in range(size_half):
        if time.time() > deadline - 30.0:
            break
        s1 = Sum_first[m1]
        p_all = S[None, :] - 2.0 * (s1[None, :] + Sum_last)
        max_vals = np.abs(p_all).max(axis=1)
        
        if len(best_heap) < K1:
            idxs = np.argsort(max_vals)[:min(K1, size_half)]
            for idx in idxs:
                heapq.heappush(best_heap, (max_vals[idx], m1, idx))
        else:
            threshold = best_heap[0][0]
            mask = max_vals < threshold
            if np.any(mask):
                idxs_all = np.where(mask)[0]
                idxs_sorted = idxs_all[np.argsort(max_vals[idxs_all])]
                # Push the best ones that are better than threshold
                # We need to maintain heap size K1.
                # If we push, we must pop the worst.
                # Simplest: heapreplace for the best one?
                # Or push and pop?
                # Let's just take the single best one from this row that is better than threshold.
                # This might miss some, but it's fast.
                # Given the vectorization speed, we can afford to check more.
                best_idx_in_row = idxs_sorted[0]
                if max_vals[best_idx_in_row] < threshold:
                    heapq.heapreplace(best_heap, (max_vals[best_idx_in_row], m1, best_idx_in_row))
    
    # --- Stage 2: Medium-resolution re-ranking (Vectorized) ---
    num_medium = 4096
    thetas_medium = np.linspace(0, 2 * np.pi, num_medium, endpoint=False)
    z_medium = np.exp(1j * thetas_medium)
    Z_medium = np.ones((num_medium, n), dtype=np.complex128)
    for k in range(1, n):
        Z_medium[:, k] = Z_medium[:, k-1] * z_medium
    
    S_med = np.sum(Z_medium, axis=1)
    
    Sum_first_med = np.zeros((size_half, num_medium), dtype=np.complex128)
    Sum_last_med = np.zeros((size_half, num_medium), dtype=np.complex128)
    
    for m in range(1, size_half):
        lsb = m & -m
        idx = lsb.bit_length() - 1
        prev = m ^ lsb
        Sum_first_med[m] = Sum_first_med[prev] + Z_medium[:, idx]
    
    for m in range(1, size_half):
        lsb = m & -m
        idx = lsb.bit_length() - 1
        prev = m ^ lsb
        Sum_last_med[m] = Sum_last_med[prev] + Z_medium[:, half_n + idx]
    
    # Re-rank the top K1 candidates with medium grid
    candidates = sorted(best_heap, key=lambda x: x[0])
    K2 = 200
    medium_heap = []  # (max_val, m1, m2)
    
    for max_val, m1, m2 in candidates:
        if time.time() > deadline - 10.0:
            break
        s1 = Sum_first_med[m1]
        s2 = Sum_last_med[m2]
        p = S_med - 2.0 * (s1 + s2)
        abs_p = np.abs(p)
        max_val_med = abs_p.max()
        
        if len(medium_heap) < K2:
            heapq.heappush(medium_heap, (max_val_med, m1, m2))
        elif max_val_med < medium_heap[0][0]:
            heapq.heapreplace(medium_heap, (max_val_med, m1, m2))
    
    # --- Stage 3: Golden-section refinement of top candidates ---
    def refine_peak(coeffs):
        # Precompute Z for fine grid
        num_fine = 4096
        thetas_fine = np.linspace(0, 2 * np.pi, num_fine, endpoint=False)
        z_fine = np.exp(1j * thetas_fine)
        Z_fine = np.ones((num_fine, n), dtype=np.complex128)
        for k in range(1, n):
            Z_fine[:, k] = Z_fine[:, k-1] * z_fine
        
        def f(theta):
            z_point = np.exp(1j * theta)
            # Use Horner's method or direct sum?
            # Direct sum with precomputed powers is faster for small n
            # But we can just use polyval if we reverse coeffs?
            # P(z) = sum c_k z^k.
            # np.polyval expects highest degree first.
            # Let's just do a loop, it's only 24 iterations.
            p_val = 0.0
            zp = 1.0 + 0.0j
            for c in coeffs:
                p_val += c * zp
                zp *= z_point
            return np.abs(p_val)
        
        # Evaluate on fine grid
        vals = Z_fine @ coeffs
        abs_vals = np.abs(vals)
        
        # Find top 3 peaks
        # Sort indices by abs_vals
        idxs = np.argsort(abs_vals)[::-1]
        top_idxs = idxs[:3]
        
        max_refined = 0.0
        for idx in top_idxs:
            theta_max = thetas_fine[idx]
            
            # Golden section
            delta = 2 * np.pi / num_fine
            a = theta_max - delta
            b = theta_max + delta
            gr = (np.sqrt(5) - 1) / 2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            fc = f(c)
            fd = f(d)
            
            for _ in range(30):
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
            
            val = f((a + b) / 2)
            if val > max_refined:
                max_refined = val
        
        return max_refined
    
    # Refine top candidates
    best_val = float('inf')
    best_mask = 0
    
    medium_candidates = sorted(medium_heap, key=lambda x: x[0])
    
    for max_val_med, m1, m2 in medium_candidates:
        if time.time() > deadline - 2.0:
            break
        if max_val_med >= best_val and best_val < float('inf'):
            break
        
        mask = m1 | (m2 << half_n)
        coeffs = np.ones(n, dtype=np.float64)
        for k in range(n):
            if (mask >> k) & 1:
                coeffs[k] = -1.0
        
        refined_val = refine_peak(coeffs)
        
        if refined_val < best_val:
            best_val = refined_val
            best_mask = mask
    
    # Return final coefficients
    final_coeffs = np.ones(n, dtype=int)
    for k in range(n):
        if (best_mask >> k) & 1:
            final_coeffs[k] = -1
            
    return final_coeffs.tolist()
