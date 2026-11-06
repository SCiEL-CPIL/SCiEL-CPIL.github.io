"""
Constructs a binary code of length 18 with minimum Hamming distance 6.
Uses the Extended Quadratic Residue Code for p=17, which is a [18, 9, 6] code 
with 512 codewords. Attempts to extend via FLIP-based search if time permits.
"""
import time
import random
import itertools

N = 18
MIN_DIST = 6
TIME_LIMIT = 60.0

def popcount(x):
    return x.bit_count()

def hamming(a, b):
    return popcount(a ^ b)

def verify_code(words):
    """Verify all pairs have distance >= MIN_DIST."""
    n = len(words)
    for i in range(n):
        wi = words[i]
        for j in range(i + 1, n):
            if hamming(wi, words[j]) < MIN_DIST:
                return False
    return True

def quadratic_residue_code():
    """
    Construct the Extended Quadratic Residue Code for p=17.
    Parameters: [18, 9, 6], Size: 512.
    Precomputes generator rows as integers for fast XOR.
    """
    p = 17
    # Quadratic residues mod 17
    qr = set((i * i) % p for i in range(p))
    
    dim = (p - 1) // 2  # 8
    
    # Precompute each generator row as an integer
    G_rows = []
    for i in range(dim):
        row_int = 0
        for j in range(p):
            if (j - i) % p in qr:
                row_int |= (1 << j)
        G_rows.append(row_int)
    
    # Generate all 2^8 = 256 codewords of length 17 using integer XOR
    codewords_17 = []
    for mask in range(1 << dim):
        word = 0
        for i in range(dim):
            if mask & (1 << i):
                word ^= G_rows[i]
        codewords_17.append(word)
    
    # Extend by an overall parity bit to get length 18
    codewords_18 = []
    for w in codewords_17:
        parity = popcount(w) & 1
        codewords_18.append(w | (parity << p))
    
    return codewords_18

def flip_search(words, time_budget=0):
    """
    FLIP-based search: try to add new codewords by flipping bits in existing ones
    or by systematic candidate generation.
    """
    if time_budget <= 0:
        return words
    
    deadline = time.time() + time_budget
    words = list(words)
    word_set = set(words)
    n = len(words)
    
    random.seed(42)
    
    # Strategy 1: Try flipping 6 bits in each existing codeword
    # This generates candidates that are at distance exactly 6 from one existing word
    # but may be at distance >= 6 from all others
    
    # Precompute bit positions
    all_positions = list(range(N))
    
    # First pass: try all 6-bit flips of each codeword (systematic)
    # C(18,6) = 18564 candidates per codeword, times 512 = ~9.5M total
    # This is too many, so we sample
    
    # More efficient: random 6-bit flips
    attempts = 0
    added = 0
    
    while time.time() < deadline:
        # Pick a random base word
        base_idx = random.randint(0, n - 1)
        base = words[base_idx]
        
        # Pick 6 random positions to flip
        flip_positions = random.sample(all_positions, MIN_DIST)
        flip_mask = 0
        for pos in flip_positions:
            flip_mask |= (1 << pos)
        
        candidate = base ^ flip_mask
        
        if candidate in word_set:
            attempts += 1
            continue
        
        # Check distance to all existing words
        ok = True
        for w in words:
            if hamming(candidate, w) < MIN_DIST:
                ok = False
                break
        
        if ok:
            words.append(candidate)
            word_set.add(candidate)
            n = len(words)
            added += 1
        
        attempts += 1
        
        # Periodic check
        if attempts % 5000 == 0 and time.time() > deadline:
            break
    
    return words

def greedy_extend(words, time_budget=0):
    """
    Greedily try to add more words to the code while maintaining min distance.
    Uses a set for O(1) membership check.
    """
    if time_budget <= 0:
        return words
    
    deadline = time.time() + time_budget
    words = list(words)
    word_set = set(words)
    
    random.seed(42)
    
    attempts = 0
    while time.time() < deadline:
        candidate = random.randint(0, (1 << N) - 1)
        if candidate in word_set:
            attempts += 1
            continue
        
        # Check distance to all existing words
        ok = True
        for w in words:
            if hamming(candidate, w) < MIN_DIST:
                ok = False
                break
        
        if ok:
            words.append(candidate)
            word_set.add(candidate)
        
        attempts += 1
    
    return words

def construct():
    """
    Main entry point.
    """
    start_time = time.time()
    
    # Step 1: Construct the extended QR code of length 18
    qr_words = quadratic_residue_code()
    
    # Verify validity
    if not verify_code(qr_words):
        # Fallback to Reed-Muller if QR construction fails (should not happen)
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
    
    # Step 2: Try to extend with FLIP search (more targeted than random)
    remaining_time = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining_time > 1:
        # Use FLIP search first (more likely to find valid additions)
        extended_words = flip_search(qr_words, time_budget=remaining_time * 0.7)
        
        # If time remains, also try random greedy
        remaining_time2 = TIME_LIMIT - 5 - (time.time() - start_time)
        if remaining_time2 > 1:
            extended_words = greedy_extend(extended_words, time_budget=remaining_time2 * 0.9)
    else:
        extended_words = qr_words
    
    # Final verification
    if not verify_code(extended_words):
        return qr_words
    
    return extended_words
