"""
Construct a binary error-correcting code of length 18 with minimum distance 6.
Uses the extended quadratic residue code for p=17, then attempts growth via
efficient candidate search and local search.
"""
import time
import random

N = 18
MIN_DIST = 6
TIME_BUDGET = 55.0  # safety margin below 60s limit


def popcount(x):
    return x.bit_count()


def hamming(a, b):
    return (a ^ b).bit_count()


def verify_code(words):
    """Verify all pairs have distance >= MIN_DIST. Returns True if valid."""
    n = len(words)
    ws = list(words)
    for i in range(n):
        wi = ws[i]
        for j in range(i + 1, n):
            if (wi ^ ws[j]).bit_count() < MIN_DIST:
                return False
    return True


def construct_extended_qr17():
    """
    Construct the extended quadratic residue code over F_2 for p=17.
    Returns the [18,10,6] extended QR code with 1024 codewords.
    """
    p = 17
    # Quadratic residues mod 17 (including 0)
    Q = set((i * i) % p for i in range(p))
    # Q = {0, 1, 2, 4, 8, 9, 13, 15, 16} (9 elements)
    
    # Build 9x17 generator for the [17,9,5] QR code
    dim9 = 9
    G_rows_17 = []
    for i in range(dim9):
        row = 0
        for j in range(p):
            if (j - i) % p in Q:
                row |= (1 << j)
        G_rows_17.append(row)
    
    # Generate all 512 codewords of length 17
    codewords_17 = []
    for mask in range(1 << dim9):
        word = 0
        m = mask
        i = 0
        while m:
            if m & 1:
                word ^= G_rows_17[i]
            m >>= 1
            i += 1
        codewords_17.append(word)
    
    # Extend to length 18 with parity bit
    codewords_18 = set()
    for w in codewords_17:
        parity = popcount(w) & 1
        extended = w | (parity << p)
        codewords_18.add(extended)
    
    # For p ≡ 1 (mod 4), the extended QR code has dimension (p+3)/2 = 10.
    # The 10th generator vector is the all-ones vector of length 18.
    # Adding its coset doubles the code to 1024 words.
    all_ones_18 = (1 << 18) - 1
    if all_ones_18 not in codewords_18:
        for w in list(codewords_18):
            codewords_18.add(w ^ all_ones_18)
    
    return sorted(codewords_18)


def efficient_distance_check(candidate, existing_words):
    """Check if candidate has distance >= MIN_DIST from all existing_words."""
    for w in existing_words:
        if (candidate ^ w).bit_count() < MIN_DIST:
            return False
    return True


def grow_code(words, deadline):
    """
    Attempt to grow the code by adding new codewords.
    Uses structured candidate generation with multiple strategies.
    """
    words = list(words)
    existing = set(words)
    rng = random.Random(42)
    
    # Strategy 1: Try random candidates
    attempts = 0
    max_attempts = 500000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        candidate = rng.getrandbits(N)
        
        if candidate in existing:
            continue
        
        # Quick weight-based rejection
        w = popcount(candidate)
        if 0 in existing and w < MIN_DIST:
            continue
        if ((1 << N) - 1) in existing and w > N - MIN_DIST:
            continue
        
        if efficient_distance_check(candidate, words):
            words.append(candidate)
            existing.add(candidate)
    
    # Strategy 2: Try structured candidates near existing codewords
    # Flip 2-3 bits of existing codewords to find new valid positions
    if time.time() < deadline:
        base_words = list(words)
        for w in base_words:
            if time.time() > deadline:
                break
            # Try all pairs of bit flips
            for b1 in range(N):
                if time.time() > deadline:
                    break
                for b2 in range(b1 + 1, N):
                    candidate = w ^ (1 << b1) ^ (1 << b2)
                    if candidate in existing:
                        continue
                    if efficient_distance_check(candidate, words):
                        words.append(candidate)
                        existing.add(candidate)
    
    return words


def local_search_improve(words, deadline):
    """
    Try to improve the code by swapping codewords to enable growth.
    """
    words = list(words)
    existing = set(words)
    rng = random.Random(123)
    n = len(words)
    best_size = len(words)
    best_code = list(words)
    
    while time.time() < deadline - 2:
        # Pick a random codeword to try replacing
        idx = rng.randrange(n)
        orig = words[idx]
        
        # Try single-bit flips
        for bit in range(N):
            if time.time() > deadline:
                break
            flipped = orig ^ (1 << bit)
            if flipped in existing:
                continue
            
            ok = True
            for j in range(n):
                if j == idx:
                    continue
                if (flipped ^ words[j]).bit_count() < MIN_DIST:
                    ok = False
                    break
            
            if ok:
                words[idx] = flipped
                existing.discard(orig)
                existing.add(flipped)
                break
        
        # Try to grow after each swap
        if time.time() < deadline - 2:
            old_len = len(words)
            words = grow_code(words, time.time() + 1.0)
            if len(words) > best_size:
                best_size = len(words)
                best_code = list(words)
                existing = set(words)
    
    return best_code


def construct():
    """
    Main entry point. Returns a list of distinct integers in [0, 2^18)
    representing codewords with pairwise Hamming distance >= 6.
    """
    start_time = time.time()
    deadline = start_time + TIME_BUDGET
    
    # Step 1: Construct the extended QR code
    qr_words = construct_extended_qr17()
    
    # Verify the construction
    if not verify_code(qr_words):
        # Fallback: build from scratch with greedy
        words = [0]
        for candidate in range(1, 1 << N):
            if time.time() > deadline:
                break
            w = popcount(candidate)
            if w < MIN_DIST:
                continue
            if all((candidate ^ x).bit_count() >= MIN_DIST for x in words):
                words.append(candidate)
                if len(words) > 3000:
                    break
        if not verify_code(words):
            valid = [0]
            for w in words:
                if all((w ^ x).bit_count() >= MIN_DIST for x in valid):
                    valid.append(w)
            return valid
        return words
    
    # Step 2: Try to grow the code
    remaining = deadline - time.time()
    if remaining > 5:
        grown_words = grow_code(qr_words, time.time() + remaining * 0.5)
    else:
        grown_words = qr_words
    
    # Step 3: Local search
    remaining = deadline - time.time()
    if remaining > 5:
        final_words = local_search_improve(grown_words, time.time() + remaining * 0.6)
    else:
        final_words = grown_words
    
    # Final verification
    if not verify_code(final_words):
        return qr_words
    
    return final_words


if __name__ == "__main__":
    result = construct()
    print(f"Constructed code with {len(result)} codewords")
    print("Full verification:", verify_code(result))
