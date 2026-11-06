import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 17
    MIN_DIST = 6
    
    def time_left():
        return deadline - time.time()
    
    # Generate all candidate words of weight 8 and 9 (optimal weight classes for d=6)
    # Weight 8: C(17,8) = 24310
    # Weight 9: C(17,9) = 24310
    # Total: 48620 candidates
    
    # Precompute all weight-8 words
    weight8_words = []
    for positions in combinations(range(N), 8):
        w = 0
        for p in positions:
            w |= (1 << p)
        weight8_words.append(w)
    
    # Precompute all weight-9 words
    weight9_words = []
    for positions in combinations(range(N), 9):
        w = 0
        for p in positions:
            w |= (1 << p)
        weight9_words.append(w)
    
    all_candidates = weight8_words + weight9_words
    n_cand = len(all_candidates)
    
    # Build compatibility matrix using numpy
    # compatible[i,j] = 1 if dist(cand_i, cand_j) >= 6
    cand_arr = np.array(all_candidates, dtype=np.uint32)
    
    # Popcount lookup table
    popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    
    def popcount_vec(arr):
        b0 = arr & 0xFF
        b1 = (arr >> 8) & 0xFF
        b2 = (arr >> 16) & 0xFF
        b3 = (arr >> 24) & 0xFF
        return popcount_table[b0] + popcount_table[b1] + popcount_table[b2] + popcount_table[b3]
    
    # Build adjacency matrix in blocks to save memory
    # We'll use a list of bitsets for neighbors
    # For each vertex, store which other vertices are compatible (distance >= 6)
    
    # Memory-efficient approach: process in chunks
    # For each candidate, compute its compatibility with all others
    
    # Use a boolean matrix stored as bytes
    # n_cand x n_cand = 48620^2 ~ 2.36 billion bits ~ 295 MB as bits
    # As bytes: 48620 x 48620 / 8 ~ 295 MB - too much
    
    # Alternative: use sparse representation or process greedily with on-the-fly checks
    
    # Strategy: Greedy construction with local search
    # 1. Start with a good initial set (QR code)
    # 2. Use greedy to add compatible candidates
    # 3. Use local search (swap) to improve
    
    # First, construct the QR code as initial seed
    PRIM_POLY = 0x11D
    
    def gf_mul(a, b):
        res = 0
        for i in range(8):
            if (b >> i) & 1:
                res ^= a
            a <<= 1
            if a & 0x100:
                a ^= PRIM_POLY
        return res

    def gf_pow(a, p):
        res = 1
        for _ in range(p):
            res = gf_mul(res, a)
        return res

    alpha = None
    for a in range(2, 256):
        if gf_pow(a, 17) == 1 and gf_pow(a, 1) != 1:
            alpha = a
            break
    
    if alpha is None:
        return greedy_construct(deadline)

    Q = {1, 2, 4, 8, 9, 13, 15, 16}
    g_coeffs = [1]
    
    for q in Q:
        root = gf_pow(alpha, q)
        new_g = [0] * (len(g_coeffs) + 1)
        for i in range(len(g_coeffs)):
            new_g[i] ^= gf_mul(g_coeffs[i], root)
            new_g[i+1] ^= g_coeffs[i]
        g_coeffs = new_g
        
    bin_g = [1 if c == 1 else 0 for c in g_coeffs]
    while len(bin_g) < 9:
        bin_g.append(0)
        
    g_int = 0
    for i in range(9):
        if bin_g[i]:
            g_int |= (1 << i)
            
    base_code = []
    for m in range(512):
        c = 0
        for i in range(9):
            if (m >> i) & 1:
                c ^= (g_int << i)
        if c.bit_count() % 2 == 0:
            base_code.append(c)
    
    # Map QR code words to candidate indices
    cand_to_idx = {c: i for i, c in enumerate(all_candidates)}
    initial_indices = []
    for c in base_code:
        if c in cand_to_idx:
            initial_indices.append(cand_to_idx[c])
    
    # Start with the QR code
    selected = set(initial_indices)
    selected_words = [all_candidates[i] for i in selected]
    
    # Precompute which candidates are in weight 8 or 9 (all of them are)
    # For fast distance checking, use numpy vectorization
    
    # Greedy addition: try to add candidates that are compatible with all selected
    # Process candidates in random order, add if compatible
    
    rng = random.Random(42)
    
    # Create a list of candidate indices not in selected
    remaining = [i for i in range(n_cand) if i not in selected]
    rng.shuffle(remaining)
    
    # For efficiency, maintain a numpy array of selected words
    selected_arr = np.array(selected_words, dtype=np.uint32)
    
    def can_add(candidate_idx):
        """Check if candidate can be added to current selection."""
        if len(selected_arr) == 0:
            return True
        cand = np.array([all_candidates[candidate_idx]], dtype=np.uint32)
        xors = selected_arr ^ cand[0]
        dists = popcount_vec(xors)
        return np.all(dists >= MIN_DIST)
    
    # Greedy pass
    for idx in remaining:
        if time_left() < 3.0:
            break
        if can_add(idx):
            selected.add(idx)
            selected_words.append(all_candidates[idx])
            selected_arr = np.append(selected_arr, np.array([all_candidates[idx]], dtype=np.uint32))
    
    # Local search: 1-for-1 swap
    # Try removing one selected and adding two new ones
    local_rng = random.Random(123)
    improved = True
    while improved and time_left() > 3.0:
        improved = False
        n_sel = len(selected)
        
        # Try random swaps
        for _ in range(min(2000, n_sel * 3)):
            if time_left() < 3.0:
                break
            
            # Pick a random selected index to remove
            sel_list = list(selected)
            remove_idx = local_rng.choice(sel_list)
            
            # Find candidates that are compatible with all selected except remove_idx
            # and also compatible with each other
            
            # Try random pairs of unselected candidates
            unselected = [i for i in range(n_cand) if i not in selected]
            if len(unselected) < 2:
                break
            
            found = False
            for _ in range(100):
                if time_left() < 3.0:
                    break
                
                i1 = local_rng.choice(unselected)
                i2 = local_rng.choice(unselected)
                if i1 == i2:
                    continue
                
                # Check if i1 and i2 are compatible with each other
                if (all_candidates[i1] ^ all_candidates[i2]).bit_count() < MIN_DIST:
                    continue
                
                # Check if i1 is compatible with all selected except remove_idx
                ok1 = True
                for s in selected:
                    if s == remove_idx:
                        continue
                    if (all_candidates[i1] ^ all_candidates[s]).bit_count() < MIN_DIST:
                        ok1 = False
                        break
                if not ok1:
                    continue
                
                # Check if i2 is compatible with all selected except remove_idx
                ok2 = True
                for s in selected:
                    if s == remove_idx:
                        continue
                    if (all_candidates[i2] ^ all_candidates[s]).bit_count() < MIN_DIST:
                        ok2 = False
                        break
                if not ok2:
                    continue
                
                # Perform swap
                selected.remove(remove_idx)
                selected_words.remove(all_candidates[remove_idx])
                selected.add(i1)
                selected_words.append(all_candidates[i1])
                selected.add(i2)
                selected_words.append(all_candidates[i2])
                
                selected_arr = np.array(selected_words, dtype=np.uint32)
                improved = True
                found = True
                break
            
            if found:
                break
    
    # Final greedy pass after local search
    remaining = [i for i in range(n_cand) if i not in selected]
    rng.shuffle(remaining)
    selected_arr = np.array(selected_words, dtype=np.uint32)
    
    for idx in remaining:
        if time_left() < 2.0:
            break
        if can_add(idx):
            selected.add(idx)
            selected_words.append(all_candidates[idx])
            selected_arr = np.append(selected_arr, np.array([all_candidates[idx]], dtype=np.uint32))
    
    return selected_words

def greedy_construct(deadline):
    # Fallback if QR construction fails
    code = [0]
    code_set = {0}
    rng = random.Random(42)
    while time.time() < deadline:
        w = rng.randint(3, 14)
        positions = rng.sample(range(17), w)
        c = 0
        for p in positions:
            c |= (1 << p)
        if c not in code_set:
            ok = True
            for w2 in code:
                if (c ^ w2).bit_count() < 6:
                    ok = False
                    break
            if ok:
                code.append(c)
                code_set.add(c)
    return code
