"""Construct largest 17-bit binary code with min Hamming distance 6."""
import numpy as np
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _build_linear_code():
    """Construct the 256-word [17,8,6] code from the [18,9,6] extended QR code."""
    poly = 0x1D
    
    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            b >>= 1
            if a & 0x80:
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
        return res
    
    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res
    
    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break
    
    if alpha is None:
        return [0]
    
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    g_coeffs = [1]
    
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j + 1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g
    
    G = np.zeros((9, 17), dtype=np.int64)
    for k in range(9):
        for j in range(9):
            if g_coeffs[j]:
                pos = (j + k) % 17
                G[k, pos] = 1
    
    words_17 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.int64)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        words_17.append(word)
    
    words_18 = []
    for w in words_17:
        parity = int(np.sum(w) % 2)
        word_ext = np.append(w, parity)
        words_18.append(word_ext)
    
    words_short = []
    for w in words_18:
        if w[17] == 0:
            w_short = w[:17]
            val = 0
            for i in range(17):
                if w_short[i]:
                    val |= (1 << i)
            words_short.append(val)
    
    return words_short


def _generate_distance_k_words(N, k):
    """Generate all N-bit words with Hamming weight exactly k."""
    words = []
    for positions in combinations(range(N), k):
        val = 0
        for p in positions:
            val |= (1 << p)
        words.append(val)
    return np.array(words, dtype=np.int64)


def construct():
    """Construct the largest set of 17-bit words with pairwise Hamming distance at least 6."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 17
    MIN_DIST = 6
    NUM_WORDS = 2**N
    
    # Build popcount table for all 17-bit values
    _PC17 = np.array([i.bit_count() for i in range(NUM_WORDS)], dtype=np.int8)
    
    # Step 1: Construct the 256-word linear code
    try:
        words = _build_linear_code()
    except Exception:
        words = [0]
    
    if not words:
        return [0]
    
    best_words = list(words)
    best_size = len(best_words)
    code_arr = np.array(best_words, dtype=np.int64)
    code_set = set(best_words)
    
    # Step 2: Generate candidate pool
    all_candidates_list = []
    for k in [6, 7, 8, 9, 10, 11]:
        if time.time() > deadline - 30.0:
            break
        cw = _generate_distance_k_words(N, k)
        mask = np.array([w not in code_set for w in cw], dtype=bool)
        all_candidates_list.append(cw[mask])
    
    if all_candidates_list:
        all_candidates = np.concatenate(all_candidates_list)
    else:
        all_candidates = np.array([], dtype=np.int64)
    
    if len(all_candidates) == 0:
        return best_words
    
    n_cands = len(all_candidates)
    
    # Precompute conflict structure: for each candidate, which codewords are within distance < MIN_DIST
    # Use the "conflict count" approach for efficient greedy
    
    # Precompute: for each candidate, the list of initial codeword indices it conflicts with
    # This is expensive upfront but saves per-restart cost
    
    # Build a mapping from codeword value to index
    cw_to_idx = {int(w): i for i, w in enumerate(code_arr)}
    
    # For each candidate, precompute which initial codewords it's too close to
    # conflicts[i] = set of codeword indices that conflict with candidate i
    # Use a list of sets for O(1) membership and fast iteration
    
    # Actually, let's use a more memory-efficient approach:
    # For each candidate, store the count of conflicts and the list of conflicting cw indices
    # When we add a new word, we only need to update candidates that conflict with it
    
    # Precompute conflict lists between candidates and initial code
    # conflicts_init[cand_idx] = list of cw indices in code_arr that are too close
    conflicts_init = []
    for i in range(n_cands):
        c = all_candidates[i]
        clist = []
        for j in range(len(code_arr)):
            if _PC17[c ^ code_arr[j]] < MIN_DIST:
                clist.append(j)
        conflicts_init.append(clist)
    
    # Also precompute conflicts between candidates
    # This is O(n_cands^2) which could be large. Let's check size.
    # C(17,6)+C(17,7)+...+C(17,11) = 12376+19448+24310+24310+19448+12376 = ~112k
    # 112k^2 is too large. We'll compute candidate-candidate conflicts on the fly.
    
    # Greedy with conflict counting
    # State: 
    #   conflict_count[cand_idx] = number of selected words (code + added) that conflict with candidate
    #   A candidate is valid iff conflict_count[cand_idx] == 0
    #   When we select candidate j, we increment conflict_count for all candidates that conflict with j
    
    # Precompute candidate-candidate conflicts lazily in batches
    
    def greedy_extend(ordered_cand_indices, deadline):
        """Greedy extension using conflict counting."""
        selected = list(best_words)
        selected_set = set(best_words)
        n_sel = len(selected)
        
        # conflict_count[i] = number of selected words conflicting with candidate i
        conflict_count = np.array([len(conflicts_init[i]) for i in range(n_cands)], dtype=np.int32)
        
        # Track which candidates are still "alive" (conflict_count == 0)
        # We'll maintain a list of alive candidate indices
        alive = [i for i in range(n_cands) if conflict_count[i] == 0]
        
        # Map from candidate value to its index in all_candidates for quick lookup
        # Actually we need: when we select candidate j, find all alive candidates k that conflict with j
        # Precompute: for each candidate, the set of other candidates it conflicts with
        # This is O(n_cands^2 * cost) which is ~112k^2 * 1 = 12.5 billion - too slow
        
        # Alternative: when selecting candidate j, scan all alive candidates and check distance
        # This is O(|alive|) per selection, which is acceptable if |alive| shrinks fast
        
        added_count = 0
        
        for ci in ordered_cand_indices:
            if time.time() > deadline - 3.0:
                break
            
            if conflict_count[ci] > 0:
                continue
            
            # Candidate ci is valid - select it
            w = int(all_candidates[ci])
            selected.append(w)
            selected_set.add(w)
            added_count += 1
            
            # Update conflict counts for all other alive candidates
            # Check each alive candidate against w
            new_alive = []
            for k in alive:
                if k == ci:
                    continue
                if _PC17[int(all_candidates[k]) ^ w] < MIN_DIST:
                    conflict_count[k] += 1
                if conflict_count[k] == 0:
                    new_alive.append(k)
            
            # Remove ci from alive (it's now selected)
            alive = new_alive
            
            if not alive:
                break
        
        return selected, added_count
    
    # Try multiple orderings
    rng = np.random.default_rng(seed=42)
    
    best_added = 0
    best_final_list = best_words
    
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Use different ordering strategies
        if restart == 0:
            ordered = list(range(n_cands))
        elif restart % 2 == 1:
            ordered = rng.permutation(n_cands).tolist()
        else:
            # Sort by weight (lower weight first tends to be more compatible)
            weights = np.array([_PC17[c] for c in all_candidates])
            ordered = np.argsort(weights, kind='stable').tolist()
            # Add some randomness to ties
            # Actually just use the sorted order
        
        selected, added_count = greedy_extend(ordered, deadline)
        
        if added_count > best_added:
            best_added = added_count
            best_final_list = selected
    
    if len(best_final_list) > best_size:
        best_size = len(best_final_list)
        best_words = best_final_list
    
    return best_words

# EVOLVE-BLOCK-END
