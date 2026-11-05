"""Construct largest 17-bit binary code with min Hamming distance 6."""
import numpy as np
import time

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


def _popcount17_table():
    """Build popcount table for all 17-bit values."""
    NUM = 1 << 17
    table = np.zeros(NUM, dtype=np.int8)
    for i in range(1, NUM):
        table[i] = table[i >> 1] + (i & 1)
    return table


def _forbidden_from_code(code_arr, pc_table):
    """Compute the forbidden set (words within distance 5 of any codeword).
    
    Returns a boolean array of size 2^17 where True means forbidden.
    """
    NUM = 1 << 17
    forbidden = np.zeros(NUM, dtype=bool)
    n = len(code_arr)
    
    # Process codewords in batches to limit memory
    batch_size = 64
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = code_arr[start:end]
        b = end - start
        
        # For each codeword, enumerate all words within distance 5
        # Weight 0: the word itself
        # Weight 1: 17 words
        # Weight 2: C(17,2)=136
        # Weight 3: C(17,3)=680
        # Weight 4: C(17,4)=2380
        # Weight 5: C(17,5)=6188
        # Total per word: 1+17+136+680+2380+6188 = 9402
        
        # Precompute all offset patterns for weights 0..5
        offsets_1 = np.array([1 << i for i in range(17)], dtype=np.int64)
        
        # Weight 0
        forbidden[batch] = True
        
        # Weight 1
        for i in range(17):
            forbidden[batch ^ (1 << i)] = True
        
        # Weight 2
        for i in range(17):
            for j in range(i+1, 17):
                forbidden[batch ^ (1 << i) ^ (1 << j)] = True
        
        # Weight 3
        for i in range(17):
            for j in range(i+1, 17):
                for k in range(j+1, 17):
                    forbidden[batch ^ (1 << i) ^ (1 << j) ^ (1 << k)] = True
        
        # Weight 4
        for i in range(17):
            for j in range(i+1, 17):
                for k in range(j+1, 17):
                    for l in range(k+1, 17):
                        forbidden[batch ^ (1 << i) ^ (1 << j) ^ (1 << k) ^ (1 << l)] = True
        
        # Weight 5
        for i in range(17):
            for j in range(i+1, 17):
                for k in range(j+1, 17):
                    for l in range(k+1, 17):
                        for m in range(l+1, 17):
                            forbidden[batch ^ (1 << i) ^ (1 << j) ^ (1 << k) ^ (1 << l) ^ (1 << m)] = True
    
    return forbidden


def _forbidden_single(word, pc_table):
    """Compute forbidden set for a single word (within distance 5)."""
    NUM = 1 << 17
    forbidden = np.zeros(NUM, dtype=bool)
    
    forbidden[word] = True
    for i in range(17):
        forbidden[word ^ (1 << i)] = True
    for i in range(17):
        for j in range(i+1, 17):
            forbidden[word ^ (1 << i) ^ (1 << j)] = True
    for i in range(17):
        for j in range(i+1, 17):
            for k in range(j+1, 17):
                forbidden[word ^ (1 << i) ^ (1 << j) ^ (1 << k)] = True
    for i in range(17):
        for j in range(i+1, 17):
            for k in range(j+1, 17):
                for l in range(k+1, 17):
                    forbidden[word ^ (1 << i) ^ (1 << j) ^ (1 << k) ^ (1 << l)] = True
    for i in range(17):
        for j in range(i+1, 17):
            for k in range(j+1, 17):
                for l in range(k+1, 17):
                    for m in range(l+1, 17):
                        forbidden[word ^ (1 << i) ^ (1 << j) ^ (1 << k) ^ (1 << l) ^ (1 << m)] = True
    
    return forbidden


def _min_dist_to_code(word, code_arr, pc_table):
    """Compute minimum Hamming distance from word to any codeword."""
    diffs = code_arr ^ word
    return int(np.min(pc_table[diffs]))


def construct():
    """Construct the largest set of 17-bit words with pairwise Hamming distance at least 6."""
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit
    
    N = 17
    MIN_DIST = 6
    NUM_WORDS = 1 << N
    
    # Build popcount table
    pc_table = _popcount17_table()
    
    # Step 1: Construct the 256-word linear code
    try:
        words = _build_linear_code()
    except Exception:
        words = [0]
    
    if not words:
        return [0]
    
    best_words = list(words)
    best_size = len(best_words)
    
    # Step 2: Build forbidden set for the linear code
    code_arr = np.array(best_words, dtype=np.int64)
    forbidden = _forbidden_from_code(code_arr, pc_table)
    
    # Step 3: Greedy addition using forbidden set
    # Find all non-forbidden words and add them greedily
    # Order candidates by weight for better packing
    all_indices = np.arange(NUM_WORDS, dtype=np.int64)
    available = all_indices[~forbidden]
    
    # Sort available words by weight (lighter words first often pack better)
    weights = pc_table[available]
    sort_order = np.argsort(weights, kind='stable')
    available_sorted = available[sort_order]
    
    current_list = list(best_words)
    current_set = set(best_words)
    
    for idx in range(len(available_sorted)):
        if time.time() > deadline - 30.0:
            break
        
        w = int(available_sorted[idx])
        if w in current_set:
            continue
        
        # Check if this word is still available (not forbidden by previously added words)
        # We need to verify against the current forbidden set
        # But updating forbidden set is expensive, so we re-check distance
        # Actually, let's use a smarter approach: maintain forbidden incrementally
        
        # For efficiency, just check min distance to current code
        if len(current_list) > 300:
            # Use vectorized check
            diffs = np.array(current_list, dtype=np.int64) ^ w
            min_d = int(np.min(pc_table[diffs]))
        else:
            min_d = _min_dist_to_code(w, code_arr, pc_table)
        
        if min_d >= MIN_DIST:
            current_list.append(w)
            current_set.add(w)
            code_arr = np.append(code_arr, w)
    
    if len(current_list) > best_size:
        best_size = len(current_list)
        best_words = list(current_list)
    
    # Step 4: Local search - try to add more words and perform swaps
    current_list = list(best_words)
    current_set = set(best_words)
    code_arr = np.array(current_list, dtype=np.int64)
    rng = np.random.default_rng(seed=123)
    
    # Rebuild forbidden set for the current code
    forbidden = _forbidden_from_code(code_arr, pc_table)
    
    # Try to add any remaining available words
    for _ in range(200):
        if time.time() > deadline - 15.0:
            break
        
        # Find available words
        available = np.where(~forbidden)[0]
        if len(available) == 0:
            break
        
        # Try a random available word
        w = int(available[rng.integers(0, len(available))])
        if w in current_set:
            continue
        
        # Verify
        diffs = code_arr ^ w
        min_d = int(np.min(pc_table[diffs]))
        
        if min_d >= MIN_DIST:
            current_list.append(w)
            current_set.add(w)
            code_arr = np.append(code_arr, w)
            # Update forbidden set incrementally
            new_forbidden = _forbidden_single(w, pc_table)
            forbidden |= new_forbidden
    
    if len(current_list) > best_size:
        best_size = len(current_list)
        best_words = list(current_list)
    
    # Step 5: Swap-based local search
    # Try removing one word and adding one or more new words
    current_list = list(best_words)
    current_set = set(best_words)
    code_arr = np.array(current_list, dtype=np.int64)
    
    for iteration in range(500):
        if time.time() > deadline - 8.0:
            break
        
        # Pick a random codeword to potentially remove
        i = int(rng.integers(0, len(current_list)))
        removed = int(current_list[i])
        
        # Temporarily remove it
        temp_list = current_list[:i] + current_list[i+1:]
        temp_set = set(temp_list)
        temp_arr = np.array(temp_list, dtype=np.int64)
        
        # Compute forbidden set for temp code
        temp_forbidden = _forbidden_from_code(temp_arr, pc_table)
        
        # Find available words
        available = np.where(~temp_forbidden)[0]
        if len(available) == 0:
            continue
        
        # Try to add as many as possible
        added = []
        for w in available[:50]:  # Limit candidates
            w = int(w)
            if w in temp_set:
                continue
            diffs = temp_arr ^ w
            min_d = int(np.min(pc_table[diffs]))
            if min_d >= MIN_DIST:
                added.append(w)
                temp_set.add(w)
                temp_arr = np.append(temp_arr, w)
        
        # Only accept if we added more than we removed
        if len(added) >= 2:
            current_list = temp_list + added
            current_set = temp_set
            code_arr = temp_arr
            if len(current_list) > best_size:
                best_size = len(current_list)
                best_words = list(current_list)
    
    return best_words

# EVOLVE-BLOCK-END
