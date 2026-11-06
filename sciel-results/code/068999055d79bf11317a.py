"""
Construct a binary error-correcting code of length 18 with minimum distance 6.
Uses the extended quadratic residue code for p=17 (512 words), then grows
to 1024 words via structured search for a second coset.
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
    Returns the [18,9,6] extended QR code with 512 codewords.
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
    
    return sorted(codewords_18)


def find_second_coset(base_code, deadline):
    """
    Find a word v such that the coset {c ^ v : c in base_code} is at
    distance >= MIN_DIST from all words in base_code AND internally consistent.
    Since base_code is linear, the coset is also at distance >= MIN_DIST
    internally. We just need min_dist(base_code, coset) >= MIN_DIST.
    """
    base = list(base_code)
    n_base = len(base)
    rng = random.Random(7)
    
    # Try random candidates for the coset leader
    attempts = 0
    max_attempts = 2000000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        v = rng.getrandbits(N)
        
        # Check distance from v to all base words
        ok = True
        for c in base:
            if (v ^ c).bit_count() < MIN_DIST:
                ok = False
                break
        if ok:
            return v
    
    # Try structured candidates: weight 9 words (middle weight)
    if time.time() < deadline:
        # Generate weight-9 candidates
        import itertools
        for bits in itertools.combinations(range(N), 9):
            if time.time() > deadline:
                break
            v = 0
            for b in bits:
                v |= (1 << b)
            ok = True
            for c in base:
                if (v ^ c).bit_count() < MIN_DIST:
                    ok = False
                    break
            if ok:
                return v
    
    return None


def grow_code(words, deadline):
    """
    Attempt to grow the code by adding new codewords.
    """
    words = list(words)
    existing = set(words)
    rng = random.Random(42)
    
    # Try random candidates
    attempts = 0
    max_attempts = 1000000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        candidate = rng.getrandbits(N)
        
        if candidate in existing:
            continue
        
        if efficient_distance_check(candidate, words):
            words.append(candidate)
            existing.add(candidate)
    
    return words


def efficient_distance_check(candidate, existing_words):
    """Check if candidate has distance >= MIN_DIST from all existing_words."""
    for w in existing_words:
        if (candidate ^ w).bit_count() < MIN_DIST:
            return False
    return True


def construct_reed_muller():
    """
    Construct the Reed-Muller code RM(1, 4) of length 16,
    extended to length 18 by padding with zeros.
    Returns 32 codewords with min distance 8.
    """
    words = set()
    for a in range(16):
        for b in [0, 1]:
            w = 0
            for i in range(16):
                # Affine function: (a & i).bit_count() + b
                bit = ((a & i).bit_count() + b) & 1
                if bit:
                    w |= (1 << i)
            # Pad to 18 bits (bits 16, 17 are 0)
            words.add(w)
    return list(words)

def construct():
    """
    Main entry point. Returns a list of distinct integers in [0, 2^18)
    representing codewords with pairwise Hamming distance >= 6.
    """
    start_time = time.time()
    deadline = start_time + TIME_BUDGET
    
    # Step 1: Construct the extended QR code (512 words)
    qr_words = construct_extended_qr17()
    
    # Step 2: Try to expand to 1024 words
    candidate_words = qr_words
    if time.time() < deadline - 10:
        coset_leader = find_second_coset(qr_words, time.time() + 5)
        if coset_leader is not None:
            second_coset = [w ^ coset_leader for w in qr_words]
            candidate_words = qr_words + second_coset
    
    # Step 3: Verify the largest candidate
    if verify_code(candidate_words):
        return candidate_words
    
    # Step 4: If the expanded code failed, check the base QR code
    if len(candidate_words) > len(qr_words):
        if verify_code(qr_words):
            return qr_words
    
    # Step 5: Fallback to Reed-Muller (32 words) + Growth
    rm_words = construct_reed_muller()
    if verify_code(rm_words):
        # Try to grow the RM code
        if time.time() < deadline - 5:
            grown = grow_code(rm_words, time.time() + (deadline - time.time()) * 0.8)
            if verify_code(grown):
                return grown
        return rm_words
    
    # Last resort: Single word
    return [0]


if __name__ == "__main__":
    result = construct()
    print(f"Constructed code with {len(result)} codewords")
    print("Full verification:", verify_code(result))
