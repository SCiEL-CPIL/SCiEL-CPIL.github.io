"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    random.seed(12345)
    np.random.seed(12345)
    
    N = 256
    # Precompute all 4-term APs as index tuples (0-indexed)
    aps = []
    for d in range(1, 86):
        max_start = N - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps.append((i0, i1, i2, i3))
    
    n_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (n_aps, 4)
    
    # Precompute for each position, which APs it belongs to and in which slot
    # ap_membership[pos] = list of (ap_index, slot)
    ap_membership = [[] for _ in range(N)]
    for ap_idx in range(n_aps):
        for slot in range(4):
            pos = aps_arr[ap_idx, slot]
            ap_membership[pos].append((ap_idx, slot))
    
    def count_mono(coloring):
        """Count monochromatic 4-APs. coloring is array of 0/1, shape (N,)"""
        c = coloring[aps_arr]  # shape (n_aps, 4)
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(mono.sum())
    
    def calc_all_deltas(coloring):
        """Calculate delta for flipping each position. Returns array of shape (N,)."""
        # For each position, we need to know the effect on all APs containing it
        # An AP is monochromatic iff all 4 colors are equal
        # Flipping position p changes color at p from c[p] to 1-c[p]
        # For an AP containing p in slot j:
        #   Before: mono if all 4 equal
        #   After:  mono if all 4 equal with c[p] flipped
        # 
        # Let's compute this efficiently:
        # For each AP, we can precompute the "mono status" and then for each position
        # in that AP, compute the delta.
        
        # Strategy: for each AP, determine if it's currently mono.
        # Then for each position in the AP, flipping it will:
        #   - If AP was mono: after flip, it's NOT mono (since one color differs)
        #   - If AP was not mono: after flip, it MIGHT become mono
        #
        # For an AP with colors (c0, c1, c2, c3):
        #   Currently mono: c0==c1==c2==c3
        #   After flipping slot j: new colors have 1-cj at position j
        #   New is mono iff all other 3 are equal to 1-cj
        #   i.e., the 3 non-flipped positions all equal 1-cj
        #
        # So for AP with colors (c0,c1,c2,c3), not mono:
        #   Flip slot 0: becomes mono iff c1==c2==c3==1-c0
        #   Flip slot 1: becomes mono iff c0==c2==c3==1-c1
        #   etc.
        
        deltas = np.zeros(N, dtype=np.int32)
        
        c = coloring[aps_arr]  # (n_aps, 4)
        
        # For each AP, check current mono status
        is_mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        
        # For non-mono APs, check if flipping any single position would make it mono
        # For slot j, the other 3 positions must all equal 1-c[j]
        for j in range(4):
            # Get the 3 other positions
            other_slots = [s for s in range(4) if s != j]
            # Check if all 3 others are equal to 1-c[j]
            others_equal = np.ones(n_aps, dtype=bool)
            for s in other_slots:
                others_equal &= (c[:, s] == 1 - c[:, j])
            # This AP would become mono if we flip slot j, and it's currently not mono
            would_become_mono = others_equal & ~is_mono
            # For each such AP, add +1 to delta of the position in slot j
            for ap_idx in np.where(would_become_mono)[0]:
                pos = aps_arr[ap_idx, j]
                deltas[pos] += 1
        
        # For mono APs, flipping any position makes it non-mono
        # So for each mono AP, subtract 1 from delta of each of its 4 positions
        for ap_idx in np.where(is_mono)[0]:
            for j in range(4):
                pos = aps_arr[ap_idx, j]
                deltas[pos] -= 1
        
        return deltas
    
    def spectral_refine(coloring, max_iters=20):
        """
        Use FFT-based approach to refine coloring.
        Map colors to ±1: s[i] = 1 if coloring[i]==0, -1 if coloring[i]==1
        The number of monochromatic 4-APs can be related to the 4th-order correlation.
        For a ±1 signal, a 4-AP (i, i+d, i+2d, i+3d) is monochromatic iff
        s[i]*s[i+d]*s[i+2d]*s[i+3d] = 1.
        
        We want to minimize the number of such APs, which is equivalent to
        minimizing the sum over all 4-APs of (1 + s[i]*s[i+d]*s[i+2d]*s[i+3d])/2
        = n_aps/2 + (1/2) * sum of s[i]*s[i+d]*s[i+2d]*s[i+3d]
        
        So we want to minimize the 4th-order autocorrelation sum.
        """
        s = 1 - 2 * coloring.astype(np.float64)  # ±1 signal
        best_s = s.copy()
        best_obj = None
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute objective: sum over all 4-APs of product of 4 values
            obj = 0.0
            for ap_idx in range(n_aps):
                i0, i1, i2, i3 = aps_arr[ap_idx]
                obj += s[i0] * s[i1] * s[i2] * s[i3]
            
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_s = s.copy()
            
            # Compute gradient: for each position p, the gradient of obj w.r.t. s[p]
            # d(obj)/d(s[p]) = sum over APs containing p of product of other 3
            grad = np.zeros(N, dtype=np.float64)
            for ap_idx in range(n_aps):
                i0, i1, i2, i3 = aps_arr[ap_idx]
                prod_all = s[i0] * s[i1] * s[i2] * s[i3]
                # For each position, gradient = prod_all / s[pos] = product of other 3
                grad[i0] += prod_all / s[i0]
                grad[i1] += prod_all / s[i1]
                grad[i2] += prod_all / s[i2]
                grad[i3] += prod_all / s[i3]
            
            # Gradient descent step with small learning rate
            lr = 0.1
            s_new = s - lr * grad
            # Project back to ±1 by sign
            s_new = np.sign(s_new)
            s_new[s_new == 0] = 1  # Handle zeros
            
            s = s_new
        
        # Convert back to 0/1 coloring
        coloring_new = ((1 - best_s) / 2).astype(np.int32)
        return coloring_new
    
    best_coloring = None
    best_count = float('inf')
    
    # Try multiple initializations
    initializations = []
    
    # 1. Random
    for _ in range(5):
        initializations.append(np.random.randint(0, 2, N))
    
    # 2. Alternating blocks of various sizes
    for block_size in [1, 2, 3, 4, 5, 8, 16, 32]:
        coloring = np.array([(i // block_size) % 2 for i in range(N)])
        initializations.append(coloring)
    
    # 3. Based on binary representation
    for bit in [0, 1, 2, 3, 4]:
        coloring = np.array([(i >> bit) & 1 for i in range(N)])
        initializations.append(coloring)
    
    # 4. Random with bias
    for p in [0.3, 0.7]:
        initializations.append((np.random.random(N) < p).astype(np.int32))
    
    # 5. Van der Corput / low-discrepancy style
    coloring = np.array([(i ^ (i >> 1)) & 1 for i in range(N)])
    initializations.append(coloring)
    
    for init in initializations:
        if time.time() > deadline - 15:
            break
        
        coloring = init.copy()
        current_count = count_mono(coloring)
        
        if current_count < best_count:
            best_count = current_count
            best_coloring = coloring.copy()
        
        # Fast local search using precomputed deltas
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            deltas = calc_all_deltas(coloring)
            # Find positions with negative delta (flipping improves)
            improve_mask = deltas < 0
            if not improve_mask.any():
                break
            # Flip all improving positions at once
            flip_positions = np.where(improve_mask)[0]
            coloring[flip_positions] = 1 - coloring[flip_positions]
            current_count = count_mono(coloring)
            improved = True
            if current_count < best_count:
                best_count = current_count
                best_coloring = coloring.copy()
        
        # Spectral refinement
        if time.time() < deadline - 20:
            spectral_coloring = spectral_refine(coloring, max_iters=30)
            spectral_count = count_mono(spectral_coloring)
            if spectral_count < best_count:
                best_count = spectral_count
                best_coloring = spectral_coloring.copy()
        
        # Local search on best found so far
        if time.time() < deadline - 15:
            coloring = best_coloring.copy()
            current_count = best_count
            improved = True
            while improved and time.time() < deadline - 15:
                improved = False
                deltas = calc_all_deltas(coloring)
                improve_mask = deltas < 0
                if not improve_mask.any():
                    break
                flip_positions = np.where(improve_mask)[0]
                coloring[flip_positions] = 1 - coloring[flip_positions]
                current_count = count_mono(coloring)
                improved = True
                if current_count < best_count:
                    best_count = current_count
                    best_coloring = coloring.copy()
    
    # Final local search pass
    if time.time() < deadline - 5:
        coloring = best_coloring.copy()
        current_count = best_count
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            deltas = calc_all_deltas(coloring)
            improve_mask = deltas < 0
            if not improve_mask.any():
                break
            flip_positions = np.where(improve_mask)[0]
            coloring[flip_positions] = 1 - coloring[flip_positions]
            current_count = count_mono(coloring)
            improved = True
            if current_count < best_count:
                best_count = current_count
                best_coloring = coloring.copy()
    
    # Return as list of 0/1
    return best_coloring.tolist()
# EVOLVE-BLOCK-END
