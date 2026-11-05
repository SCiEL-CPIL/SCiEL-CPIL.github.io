import numpy as np
import time

def construct():
    N = 17
    MIN_D = 6
    deadline = time.time() + 55

    # 1. Generate Base Code [17, 8, 6] - shortened extended quadratic residue code
    f1 = [1, 0, 0, 1, 1, 1, 0, 0, 1]
    basis = []
    for i in range(9):
        vec = [0] * 17
        for j in range(len(f1)):
            if i + j < 17:
                vec[i+j] = f1[j]
        basis.append(vec)
        
    words_17 = []
    for mask in range(512):
        word = [0] * 17
        for i in range(9):
            if mask & (1 << i):
                for j in range(17):
                    word[j] ^= basis[i][j]
        words_17.append(word)
        
    words_18 = []
    for w in words_17:
        parity = sum(w) % 2
        words_18.append(w + [parity])
        
    base_codewords = []
    for w18 in words_18:
        if w18[0] == 0:
            new_word = w18[1:]
            val = 0
            for i, bit in enumerate(new_word):
                if bit:
                    val |= (1 << i)
            base_codewords.append(val)
            
    # 2. Pre-compute bits for all 2^17 candidates
    all_vals = np.arange(2**N, dtype=np.int64)
    bits_all = np.array([[(w >> i) & 1 for i in range(N)] for w in all_vals], dtype=np.uint8)
    all_weights = np.sum(bits_all, axis=1)
    
    # 3. Pre-compute bits for base code
    base_bits = np.array([[(w >> i) & 1 for i in range(N)] for w in base_codewords], dtype=np.uint8)
    base_weights = np.sum(base_bits, axis=1)
    
    # 4. Filter candidates: must be dist >= 6 from ALL base codewords
    dots = bits_all @ base_bits.T
    dists = all_weights[:, None] + base_weights[None, :] - 2 * dots
    min_dists_to_base = np.min(dists, axis=1)
    
    safe_indices = np.where(min_dists_to_base >= MIN_D)[0]
    
    if len(safe_indices) == 0:
        return base_codewords

    M = len(safe_indices)
    safe_bits = bits_all[safe_indices]
    safe_vals = all_vals[safe_indices]
    safe_weights = np.sum(safe_bits, axis=1)
    
    # 5. Greedy with vectorized pool filtering and multiple restarts
    best_code = list(base_codewords)
    best_size = len(base_codewords)
    
    rng = np.random.default_rng(42)
    
    # Pre-compute pairwise distances among safe candidates for fast pool updates
    # This is M x M which might be large, so we compute on-the-fly in batches
    
    for restart in range(200):
        if time.time() > deadline - 3:
            break
            
        current_code_vals = list(base_codewords)
        current_code_bits = base_bits.copy()
        current_code_weights = base_weights.copy()
        
        # Shuffle
        perm = rng.permutation(M)
        
        # Active pool as boolean mask
        active = np.ones(M, dtype=bool)
        active_count = M
        
        for idx in perm:
            if active_count == 0:
                break
            if not active[idx]:
                continue
                
            cand_bits = safe_bits[idx]
            cand_w = safe_weights[idx]
            
            # Check against current code
            dots_c = current_code_bits @ cand_bits
            dists_c = current_code_weights + cand_w - 2 * dots_c
            if np.min(dists_c) < MIN_D:
                continue
            
            # Add this word
            current_code_vals.append(safe_vals[idx])
            current_code_bits = np.vstack([current_code_bits, cand_bits[None, :]])
            current_code_weights = np.append(current_code_weights, cand_w)
            
            # Update pool: remove candidates that are too close to the new word
            dots_p = safe_bits @ cand_bits
            dists_p = safe_weights + cand_w - 2 * dots_p
            active[dists_p < MIN_D] = False
            active[idx] = False
            active_count = np.sum(active)
            
            if len(current_code_vals) > best_size:
                best_size = len(current_code_vals)
                best_code = list(current_code_vals)
    
    # 6. Final exhaustive scan: try to add any remaining valid word
    # Build final code bits
    final_code_bits = np.array([[(w >> i) & 1 for i in range(N)] for w in best_code], dtype=np.uint8)
    final_weights = np.sum(final_code_bits, axis=1)
    
    # Check all candidates against final code
    dots_final = bits_all @ final_code_bits.T
    dists_final = all_weights[:, None] + final_weights[None, :] - 2 * dots_final
    min_dists_final = np.min(dists_final, axis=1)
    
    # Find any word not already in code that is at distance >= 6 from all
    code_set = set(best_code)
    candidates_to_add = np.where(min_dists_final >= MIN_D)[0]
    
    added = 0
    for cidx in candidates_to_add:
        if time.time() > deadline - 1:
            break
        cv = int(candidates_to_add[cidx])
        if cv in code_set:
            continue
        # Double-check (should be redundant but safe)
        cbits = bits_all[cv]
        d = np.sum(final_code_bits ^ cbits[None, :], axis=1)
        if np.min(d) >= MIN_D:
            best_code.append(cv)
            code_set.add(cv)
            final_code_bits = np.vstack([final_code_bits, cbits[None, :]])
            final_weights = np.append(final_weights, np.sum(cbits))
            added += 1
    
    return best_code
