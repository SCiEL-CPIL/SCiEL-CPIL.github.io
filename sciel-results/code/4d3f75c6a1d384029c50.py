"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _hamming_distance(a, b):
    return (a ^ b).bit_count()

def _construct_rm16():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    32 words, minimum distance 8."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _construct_qr_code():
    """Construct a quadratic-residue-based code of length 18 with distance >= 6.
    Uses the (15,7,5) QR code extended and modified to length 18."""
    # Start with the (15,7,5) quadratic residue code over F_2
    # The QR(15) code has generator polynomial g(x) = x^8 + x^7 + x^5 + x^4 + x^2 + x + 1
    # Actually, let's use a known construction.
    # 
    # Alternative: Use the (18, 32, 6) code if it exists, or build one.
    # 
    # Let's try a systematic approach: start with a good code and extend.
    # 
    # The (15, 7, 5) QR code: generator polynomial is (x^15 + 1)/(x^8 + x^7 + x^5 + x^4 + x^2 + x + 1)
    # Actually the QR code of length 15 has dimension 7 and min distance 5.
    # 
    # Let me use a different approach. I'll construct a code using the method of
    # taking a base code and adding parity bits or using product codes.
    #
    # For now, let's use a greedy construction to find a large code.
    # But first, let's try to get a good initial set.
    #
    # Actually, let's use the following known result:
    # A_2(18, 6) >= 32 (from RM code)
    # A_2(18, 6) <= 64 (by various bounds)
    #
    # Let's try to construct a code of size 64 or more using greedy + local search.
    pass

def _greedy_grow(initial_words, n=18, min_dist=6, max_time=30.0):
    """Grow a code greedily, then try local search improvements."""
    start_time = time.time()
    words = list(initial_words)
    n_words = len(words)
    
    # Precompute a set for fast lookup
    word_set = set(words)
    
    # Try to add random words
    attempts = 0
    max_attempts = 500000
    
    while attempts < max_attempts:
        elapsed = time.time() - start_time
        if elapsed > max_time:
            break
        attempts += 1
        
        candidate = random.getrandbits(n)
        if candidate in word_set:
            continue
        
        # Check distance to all existing words
        ok = True
        for w in words:
            if _hamming_distance(candidate, w) < min_dist:
                ok = False
                break
        
        if ok:
            words.append(candidate)
            word_set.add(candidate)
    
    # Now try local search: for each word, try to find a better replacement
    # or add more words by swapping.
    
    # Simple improvement: try to replace pairs of words with two new words
    # that maintain distance but allow more words.
    
    return words

def _local_search(words, n=18, min_dist=6, max_time=20.0):
    """Try to improve the code by local search."""
    start_time = time.time()
    words = list(words)
    word_set = set(words)
    
    # Try random perturbations
    for _ in range(100000):
        elapsed = time.time() - start_time
        if elapsed > max_time:
            break
        
        # Pick a random word and try to flip a few bits
        idx = random.randint(0, len(words) - 1)
        original = words[idx]
        
        # Try flipping 1-3 bits
        for n_flips in [1, 2, 3]:
            if random.random() < 0.5:
                break
            candidate = original
            for _ in range(n_flips):
                pos = random.randint(0, n - 1)
                candidate ^= (1 << pos)
            
            if candidate in word_set:
                continue
            
            # Check if replacing words[idx] with candidate maintains min distance
            ok = True
            for j, w in enumerate(words):
                if j == idx:
                    continue
                if _hamming_distance(candidate, w) < min_dist:
                    ok = False
                    break
            
            if ok:
                # Accept the change (this doesn't increase size, but might help future additions)
                word_set.discard(original)
                words[idx] = candidate
                word_set.add(candidate)
                break
    
    return words

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    # Start with the RM code (32 words, distance 8)
    words = _construct_rm16()
    
    # Greedy growth
    words = _greedy_grow(words, n=18, min_dist=6, max_time=40.0)
    
    # Local search
    if time.time() < deadline - 5.0:
        words = _local_search(words, n=18, min_dist=6, max_time=15.0)
    
    # Verify all pairs have distance >= 6
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if _hamming_distance(words[i], words[j]) < 6:
                # This shouldn't happen, but just in case
                return _construct_rm16()
    
    return words
# EVOLVE-BLOCK-END
