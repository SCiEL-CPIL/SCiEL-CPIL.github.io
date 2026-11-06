import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 17
    MIN_DIST = 6
    
    # 1. Construct the [17, 9, 6] Quadratic Residue Code (256 words)
    residues = [0, 1, 2, 4, 8, 9, 13, 15, 16]
    
    gen_row = [0] * N
    for r in residues:
        gen_row[r] = 1
        
    g_ints = []
    for i in range(9):
        val = 0
        for j in range(N):
            if gen_row[(j - i) % N]:
                val |= (1 << j)
        g_ints.append(val)
        
    words = []
    for mask in range(2**9):
        w = 0
        for i in range(9):
            if (mask >> i) & 1:
                w ^= g_ints[i]
        words.append(w)
        
    min_w = N
    for w in words:
        if w == 0:
            continue
        weight = bin(w).count('1')
        if weight < min_w:
            min_w = weight
            
    if min_w < MIN_DIST:
        # Fallback
        words = []
        np.random.seed(42)
        while len(words) < 100 and time.time() < deadline:
            cand = np.random.randint(0, 2**N)
            valid = True
            for w in words:
                if (cand ^ w).bit_count() < MIN_DIST:
                    valid = False
                    break
            if valid:
                words.append(cand)

    code_bits = np.array([[ (w >> j) & 1 for j in range(N) ] for w in words], dtype=np.uint8)
    K = len(words)
    
    while time.time() < deadline:
        num_cands = 5000
        cands = []
        
        n_rand = num_cands // 2
        rand_cands = np.random.randint(0, 2, size=(n_rand, N), dtype=np.uint8)
        cands.append(rand_cands)
        
        n_pert = num_cands - n_rand
        idxs = np.random.randint(0, K, size=n_pert)
        pert_cands = code_bits[idxs].copy()
        for _ in range(3):
            flip_pos = np.random.randint(0, N, size=n_pert)
            pert_cands[range(n_pert), flip_pos] ^= 1
            
        cands = np.vstack(cands)
        
        # Distance check
        diff = code_bits[:, None, :] != cands[None, :, :]
        dists = np.sum(diff, axis=2)
        min_dists = np.min(dists, axis=0)
        
        valid_indices = np.where(min_dists >= MIN_DIST)[0]
        
        if len(valid_indices) > 0:
            valid_cands = cands[valid_indices]
            V = len(valid_cands)
            
            # Check mutual distances
            # If V is small, do pairwise. If V is large, just take a few.
            # V is usually small.
            selected = []
            # Greedy selection of mutually distant candidates
            remaining = list(range(V))
            while remaining:
                # Pick the first one
                i = remaining[0]
                selected.append(valid_cands[i])
                remaining.pop(0)
                # Remove those too close to valid_cands[i]
                new_remaining = []
                for j in remaining:
                    d = np.sum(valid_cands[i] != valid_cands[j])
                    if d >= MIN_DIST:
                        new_remaining.append(j)
                remaining = new_remaining
                
            # Add selected candidates
            if selected:
                sel_array = np.array(selected, dtype=np.uint8)
                code_bits = np.vstack([code_bits, sel_array])
                K += len(selected)
        
    final_words = []
    for row in code_bits:
        w = 0
        for j in range(N):
            if row[j]:
                w |= (1 << j)
        final_words.append(w)
        
    return final_words
