import time
import random
import numpy as np

N = 18
MIN_DIST = 6
TIME_LIMIT = 60.0

def popcount(x):
    return x.bit_count()

def hamming(a, b):
    return popcount(a ^ b)

def vectorized_popcount(arr):
    """
    Compute popcount for an array of integers using NumPy unpackbits.
    """
    if len(arr) == 0:
        return np.array([], dtype=np.int32)
    # Ensure array is uint32 for unpackbits compatibility if needed, but int32 works too
    # unpackbits expects integer types
    bits = np.unpackbits(arr.astype(np.uint32), axis=1, count=32, bitorder='little')
    return bits.sum(axis=1)

def verify_code(words):
    """Verify all pairwise distances using NumPy for speed."""
    if len(words) <= 1:
        return True
    arr = np.array(words, dtype=np.uint32)
    n = len(arr)
    for i in range(n):
        diffs = arr[i+1:] ^ arr[i]
        counts = vectorized_popcount(diffs)
        if np.any(counts < MIN_DIST):
            return False
    return True

def construct_extended_qr_512():
    """
    Constructs the 512-word extended QR code for p=17.
    """
    p = 17
    # Quadratic residues mod 17
    qr = set((i * i) % p for i in range(p))
    
    # Generator matrix for [17, 9, 5] code
    # Rows are cyclic shifts of the indicator vector of QRs
    G = [[0]*p for _ in range((p+1)//2)]
    for i in range((p+1)//2):
        for j in range(p):
            if (j - i) % p in qr:
                G[i][j] = 1
    
    # Generate all 2^9 = 512 codewords of length 17
    codewords_17 = []
    for mask in range(1 << ((p+1)//2)):
        word = 0
        for i in range((p+1)//2):
            if mask & (1 << i):
                for j in range(p):
                    if G[i][j]:
                        word ^= (1 << j)
        codewords_17.append(word)
    
    # Extend each word by an overall parity bit to get length 18
    codewords_18 = []
    for w in codewords_17:
        parity = popcount(w) % 2
        codewords_18.append(w | (parity << p))
        
    return codewords_18

def find_coset_offset(L, time_budget=30.0):
    """
    Searches for a vector v such that the minimum distance from v to any word in L is >= MIN_DIST.
    If found, L U (v+L) is a valid code of size 2*len(L).
    Uses systematic + random search with NumPy acceleration.
    """
    n = len(L)
    L_set = set(L)
    L_arr = np.array(L, dtype=np.uint32)
    
    deadline = time.time() + time_budget
    random.seed(42)
    
    # First try systematic: low-weight vectors (more likely to be far from a code)
    # Try all vectors of weight 0 to 3
    from itertools import combinations
    for weight in range(0, 4):
        if time.time() > deadline:
            break
        for combo in combinations(range(N), weight):
            v = 0
            for pos in combo:
                v |= (1 << pos)
            if v in L_set:
                continue
            # Check distance to all words in L
            diffs = L_arr ^ v
            min_d = np.min(vectorized_popcount(diffs))
            if min_d >= MIN_DIST:
                return v
    
    # Then random search
    attempts = 0
    while time.time() < deadline:
        v = random.randint(0, (1 << N) - 1)
        if v in L_set:
            continue
        
        # Check distance to all words in L using NumPy
        diffs = L_arr ^ v
        min_d = np.min(vectorized_popcount(diffs))
        if min_d >= MIN_DIST:
            return v
        attempts += 1
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 3.0  # Safety margin
    
    # Step 1: Construct the 512-word base code
    base_code = construct_extended_qr_512()
    
    # Verify base code
    if not verify_code(base_code):
        # Fallback to small RM code if QR construction fails (should not happen)
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        # Extend to 18 bits with zeros
        return [w for w in words]
        
    best_words = list(base_code)
    
    # Step 2: Try to find a coset offset to double the size to 1024
    remaining = deadline - time.time()
    if remaining > 5:
        offset = find_coset_offset(base_code, time_budget=remaining - 2.0)
        
        if offset is not None:
            # Construct the full code: L U (offset + L)
            full_code = list(base_code)
            for w in base_code:
                full_code.append(w ^ offset)
            
            # Verify the full code
            if verify_code(full_code):
                return full_code
            else:
                # Should not happen if offset was valid, but safety check
                pass
    
    # Step 3: If no coset found, try greedy extension
    remaining = deadline - time.time()
    if remaining > 2:
        best_words = greedy_extend(best_words, time_budget=remaining)
        
    return best_words

def greedy_extend(words, time_budget=0):
    """
    Greedy extension with NumPy-accelerated distance checks.
    Tries random candidates and low-weight perturbations of existing words.
    """
    deadline = time.time() + time_budget
    words = list(words)
    word_set = set(words)
    words_arr = np.array(words, dtype=np.uint32)
    random.seed(123)
    
    while time.time() < deadline:
        # Try a random candidate
        candidate = random.randint(0, (1 << N) - 1)
        if candidate in word_set:
            continue
        
        # Check distance to all existing words
        diffs = words_arr ^ candidate
        min_d = np.min(vectorized_popcount(diffs))
        
        if min_d >= MIN_DIST:
            words.append(candidate)
            word_set.add(candidate)
            words_arr = np.append(words_arr, candidate)
    
    return words
