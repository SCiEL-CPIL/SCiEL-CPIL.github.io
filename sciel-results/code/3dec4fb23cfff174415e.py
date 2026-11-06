"""
Constructs a binary code of length 18 with minimum Hamming distance 6.
Uses the Extended Quadratic Residue Code for p=17, which is a [18, 9, 6] code 
with 512 codewords. Attempts to extend via local search (bit flips + swaps)
if time permits.
"""
import time
import random

N = 18
MIN_DIST = 6
TIME_LIMIT = 60.0
SAFETY_MARGIN = 5.0

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
    """
    p = 17
    qr = set((i * i) % p for i in range(p))
    
    dim = (p - 1) // 2  # 8
    G = [[0] * p for _ in range(dim)]
    for i in range(dim):
        for j in range(p):
            if (j - i) % p in qr:
                G[i][j] = 1
    
    codewords_17 = []
    for mask in range(1 << dim):
        word = 0
        for i in range(dim):
            if mask & (1 << i):
                row = G[i]
                for j in range(p):
                    if row[j]:
                        word ^= (1 << j)
        codewords_17.append(word)
    
    codewords_18 = []
    for w in codewords_17:
        parity = popcount(w) % 2
        codewords_18.append(w | (parity << p))
    
    return codewords_18

def check_candidate(candidate, words, word_set):
    """Check if candidate can be added to the code."""
    if candidate in word_set:
        return False
    for w in words:
        if hamming(candidate, w) < MIN_DIST:
            return False
    return True

def local_search_extend(words, time_budget):
    """
    Try to extend the code using local search:
    1. Bit-flip neighborhoods of existing codewords
    2. Random sampling as fallback
    """
    if time_budget <= 0:
        return words
    
    deadline = time.time() + time_budget
    words = list(words)
    word_set = set(words)
    best = list(words)
    best_len = len(best)
    
    random.seed(42)
    
    # Phase 1: Try bit-flip neighborhoods of existing codewords
    # For each codeword, try flipping 1 or 2 bits
    attempts = 0
    while time.time() < deadline:
        if attempts > 500000:
            break
        
        # Pick a random existing codeword
        idx = random.randint(0, len(words) - 1)
        base = words[idx]
        
        # Try flipping 1 bit
        for bit in range(N):
            candidate = base ^ (1 << bit)
            if candidate not in word_set:
                ok = True
                for w in words:
                    if hamming(candidate, w) < MIN_DIST:
                        ok = False
                        break
                if ok:
                    words.append(candidate)
                    word_set.add(candidate)
                    best = list(words)
                    best_len = len(best)
        
        # Occasionally try flipping 2 bits
        if attempts % 5 == 0:
            b1 = random.randint(0, N - 1)
            b2 = random.randint(0, N - 1)
            if b1 != b2:
                candidate = base ^ (1 << b1) ^ (1 << b2)
                if candidate not in word_set:
                    ok = True
                    for w in words:
                        if hamming(candidate, w) < MIN_DIST:
                            ok = False
                            break
                    if ok:
                        words.append(candidate)
                        word_set.add(candidate)
                        best = list(words)
                        best_len = len(best)
        
        attempts += 1
    
    # Phase 2: Random sampling
    while time.time() < deadline:
        candidate = random.randint(0, (1 << N) - 1)
        if candidate in word_set:
            continue
        ok = True
        for w in words:
            if hamming(candidate, w) < MIN_DIST:
                ok = False
                break
        if ok:
            words.append(candidate)
            word_set.add(candidate)
            best = list(words)
            best_len = len(best)
    
    return best

def swap_search(words, time_budget):
    """
    Try to improve the code by swapping pairs of codewords.
    For each pair (i, j), try replacing word[i] with a nearby word
    that maintains validity for all other words.
    """
    if time_budget <= 0:
        return words
    
    deadline = time.time() + time_budget
    words = list(words)
    word_set = set(words)
    best = list(words)
    best_len = len(best)
    
    random.seed(123)
    
    n = len(words)
    attempts = 0
    while time.time() < deadline:
        if attempts > 200000:
            break
        
        # Pick a random codeword to potentially replace
        idx = random.randint(0, n - 1)
        old_word = words[idx]
        
        # Try replacing with a nearby word (1 or 2 bit flips)
        for flip_count in [1, 2]:
            if flip_count == 1:
                bits = [random.randint(0, N - 1)]
            else:
                b1 = random.randint(0, N - 1)
                b2 = random.randint(0, N - 1)
                if b1 == b2:
                    continue
                bits = [b1, b2]
            
            candidate = old_word
            for b in bits:
                candidate ^= (1 << b)
            
            if candidate == old_word or candidate in word_set:
                continue
            
            # Check if replacing old_word with candidate maintains validity
            ok = True
            for j, w in enumerate(words):
                if j == idx:
                    continue
                if hamming(candidate, w) < MIN_DIST:
                    ok = False
                    break
            
            if ok:
                # Perform the swap
                word_set.remove(old_word)
                words[idx] = candidate
                word_set.add(candidate)
                best = list(words)
                best_len = len(best)
                break
        
        attempts += 1
    
    return best

def construct():
    """
    Main entry point.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # Step 1: Construct the extended QR code of length 18
    qr_words = quadratic_residue_code()
    
    # Verify validity
    if not verify_code(qr_words):
        # Fallback to Reed-Muller if QR construction fails
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
    
    # Step 2: Try local search to extend
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        # Allocate 70% to local search, 30% to swap search
        local_budget = remaining_time * 0.7
        swap_budget = remaining_time * 0.3
        
        extended_words = local_search_extend(qr_words, local_budget)
        
        # Step 3: Try swap search on the result
        remaining_time = deadline - time.time()
        if remaining_time > 1:
            extended_words = swap_search(extended_words, min(swap_budget, remaining_time))
    
    # Final verification
    if not verify_code(extended_words):
        return qr_words
    
    return extended_words
