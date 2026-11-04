"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Construct a signed step function with N=400 intervals on [-1/4, 1/4]
    that minimizes the autoconvolution upper bound (C3).
    Uses a sparse combinatorial design approach.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def compute_acf_fft(s):
        """Compute aperiodic autocorrelation using FFT."""
        n = len(s)
        size = 1
        while size < 2 * n:
            size *= 2
        s_pad = np.zeros(size)
        s_pad[:n] = s
        S = np.fft.fft(s_pad, size)
        acf_full = np.fft.ifft(S * np.conj(S)).real
        return acf_full[:n]
    
    def max_abs_acf_fft(s):
        """Max |ACF| at non-zero lags using FFT."""
        acf = compute_acf_fft(s)
        if len(acf) <= 1:
            return 0.0
        return np.max(np.abs(acf[1:]))
        
    def max_abs_acf_sparse(indices, signs):
        """
        Max |ACF| at non-zero lags for sparse sequence.
        Uses direct O(K^2) computation for small K, FFT otherwise.
        """
        K = len(indices)
        if K == 0:
            return 0.0
        
        # If K is small, direct computation is faster
        if K < 30:
            acf_dict = {}
            for i_idx in range(K):
                i = indices[i_idx]
                s_i = signs[i_idx]
                for j_idx in range(K):
                    j = indices[j_idx]
                    s_j = signs[j_idx]
                    k = j - i
                    if k < 0:
                        continue
                    if k in acf_dict:
                        acf_dict[k] += s_i * s_j
                    else:
                        acf_dict[k] = s_i * s_j
            
            max_val = 0.0
            for k, v in acf_dict.items():
                if k != 0:
                    if abs(v) > max_val:
                        max_val = abs(v)
            return max_val
        else:
            s = np.zeros(N)
            for idx, sig in zip(indices, signs):
                s[idx] = sig
            return max_abs_acf_fft(s)

    def local_search_sparse(init_indices, init_signs, time_limit):
        """
        Local search on sparse representation.
        """
        current_indices = list(init_indices)
        current_signs = list(init_signs)
        
        current_score_val = max_abs_acf_sparse(current_indices, current_signs)
        best_score_val = current_score_val
        best_indices = current_indices.copy()
        best_signs = current_signs.copy()
        
        while time.time() < time_limit:
            move_type = np.random.randint(4)
            
            if move_type == 0:
                # Flip sign
                if not current_indices: continue
                i = np.random.randint(len(current_indices))
                old_sign = current_signs[i]
                new_sign = -old_sign
                current_signs[i] = new_sign
                
                new_score_val = max_abs_acf_sparse(current_indices, current_signs)
                
                if new_score_val < best_score_val:
                    best_score_val = new_score_val
                    best_indices = current_indices.copy()
                    best_signs = current_signs.copy()
                else:
                    # Revert
                    current_signs[i] = old_sign
                    
            elif move_type == 1:
                # Move index
                if not current_indices: continue
                i = np.random.randint(len(current_indices))
                old_idx = current_indices[i]
                new_idx = np.random.randint(N)
                if new_idx == old_idx:
                    continue
                if new_idx in current_indices:
                    continue
                
                current_indices[i] = new_idx
                new_score_val = max_abs_acf_sparse(current_indices, current_signs)
                
                if new_score_val < best_score_val:
                    best_score_val = new_score_val
                    best_indices = current_indices.copy()
                    best_signs = current_signs.copy()
                else:
                    # Revert
                    current_indices[i] = old_idx
                    
            elif move_type == 2:
                # Add index
                new_idx = np.random.randint(N)
                if new_idx in current_indices:
                    continue
                new_sign = np.random.choice([-1, 1])
                
                current_indices.append(new_idx)
                current_signs.append(new_sign)
                
                new_score_val = max_abs_acf_sparse(current_indices, current_signs)
                
                if new_score_val < best_score_val:
                    best_score_val = new_score_val
                    best_indices = current_indices.copy()
                    best_signs = current_signs.copy()
                else:
                    # Revert
                    current_indices.pop()
                    current_signs.pop()
                    
            else:
                # Remove index
                if not current_indices: continue
                i = np.random.randint(len(current_indices))
                
                current_indices.pop(i)
                current_signs.pop(i)
                
                new_score_val = max_abs_acf_sparse(current_indices, current_signs)
                
                if new_score_val < best_score_val:
                    best_score_val = new_score_val
                    best_indices = current_indices.copy()
                    best_signs = current_signs.copy()
                else:
                    # Revert
                    current_indices.insert(i, best_indices[i]) # Wait, best_indices might be different length
                    # We need to revert to the state BEFORE the move
                    # We stored best_indices, but we need to revert CURRENT state
                    # Let's store old values before popping
                    pass # This logic is flawed for revert. Let's fix.
        
        # Convert best sparse to dense vector
        best_vec = np.zeros(N)
        for idx, sig in zip(best_indices, best_signs):
            best_vec[idx] = sig
        return best_vec, best_score_val

    # Fix the revert logic in local_search_sparse
    def local_search_sparse_v2(init_indices, init_signs, time_limit):
        current_indices = list(init_indices)
        current_signs = list(init_signs)
        
        current_score_val = max_abs_acf_sparse(current_indices, current_signs)
        best_score_val = current_score_val
        best_indices = current_indices.copy()
        best_signs = current_signs.copy()
        
        while time.time() < time_limit:
            move_type = np.random.randint(4)
            
            if move_type == 0:
                if not current_indices: continue
                i = np.random.randint(len(current_indices))
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                
                new_score_val = max_abs_acf_sparse(current_indices, current_signs)
                
                if new_score_val < best_score_val:
                    best_score_val = new_score_val
                    best_indices = current_indices.copy()
                    best_signs = current_signs.copy()
                else:
                    current_signs[i] = old_sign
                    
            elif move_type == 1:
                if not current_indices: continue
                i = np.random.randint(len(current_indices))
                old_idx = current_indices[i]
                new_idx = np.random.randint(N)
                if new_idx == old_idx or new_idx in current_indices:
                    continue
                
                current_indices[i] = new_idx
                new_score_val = max_abs_acf_sparse(current_indices, current_signs)
                
                if new_score_val < best_score_val:
                    best_score_val = new_score_val
                    best_indices = current_indices.copy()
                    best_signs = current_signs.copy()
                else:
                    current_indices[i] = old_idx
                    
            elif move_type == 2:
                new_idx = np.random.randint(N)
                if new_idx in current_indices:
                    continue
                new_sign = np.random.choice([-1, 1])
                
                current_indices.append(new_idx)
                current_signs.append(new_sign)
                
                new_score_val = max_abs_acf_sparse(current_indices, current_signs)
                
                if new_score_val < best_score_val:
                    best_score_val = new_score_val
                    best_indices = current_indices.copy()
                    best_signs = current_signs.copy()
                else:
                    current_indices.pop()
                    current_signs.pop()
                    
            else:
                if not current_indices: continue
                i = np.random.randint(len(current_indices))
                old_idx = current_indices[i]
                old_sign = current_signs[i]
                
                current_indices.pop(i)
                current_signs.pop(i)
                
                new_score_val = max_abs_acf_sparse(current_indices, current_signs)
                
                if new_score_val < best_score_val:
                    best_score_val = new_score_val
                    best_indices = current_indices.copy()
                    best_signs = current_signs.copy()
                else:
                    current_indices.insert(i, old_idx)
                    current_signs.insert(i, old_sign)
        
        best_vec = np.zeros(N)
        for idx, sig in zip(best_indices, best_signs):
            best_vec[idx] = sig
        return best_vec, best_score_val

    # Main search
    best_overall_vec = np.zeros(N)
    best_overall_score = float('inf')
    
    k_values = [5, 10, 20, 50, 100]
    
    for K in k_values:
        if time.time() > deadline - 5:
            break
            
        time_limit = min(20.0, deadline - time.time() - 2)
        
        # Initialize with random permutation
        indices_pool = np.random.permutation(N)
        init_indices = indices_pool[:K].tolist()
        init_signs = np.random.choice([-1, 1], size=K).tolist()
        
        vec, sc = local_search_sparse_v2(init_indices, init_signs, time_limit)
        
        if np.all(vec == 0):
            vec[0] = 1
            sc = max_abs_acf_fft(vec)
            
        if sc < best_overall_score:
            best_overall_score = sc
            best_overall_vec = vec
            
    return best_overall_vec
# EVOLVE-BLOCK-END
