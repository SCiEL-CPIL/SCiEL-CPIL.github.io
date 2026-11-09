import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0 # Safety margin for 60s limit
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET
    
    # Pre-compute bad masks (Hamming weight < MIN_DIST)
    # These are the offsets that define the "blocked" neighborhood
    bad_masks = []
    for m in range(1 << N):
        if m.bit_count() < MIN_DIST:
            bad_masks.append(m)
    
    # Function to build a blocked set from a list of words
    def build_blocked(words):
        blocked = set()
        for w in words:
            for m in bad_masks:
                blocked.add(w ^ m)
        return blocked

    # Function to verify a code
    def verify(words):
        if not words:
            return 0
        min_d = N
        L = len(words)
        # For large L, this is O(L^2). 512^2 is ~260k, fine. 
        # If we get to 1000+, 1M checks, still fine in Python? 
        # 1M bit_count ops is ~0.1s. 
        # We should be careful.
        for i in range(L):
            for j in range(i + 1, L):
                d = (words[i] ^ words[j]).bit_count()
                if d < min_d:
                    min_d = d
                    if min_d < MIN_DIST:
                        return 0
        return min_d

    # 1. Construct the Extended Quadratic Residue Code (512 words)
    # This is a strong baseline.
    def get_qr_code():
        p = 17
        Q = set((x * x) % p for x in range(1, p))
        u = [1 if i in Q else 0 for i in range(p)]
        
        # Find basis via Gaussian Elimination
        basis = []
        for k in range(p):
            temp = [u[(i + k) % p] for i in range(p)]
            # Reduce
            for b in basis:
                # Find pivot in temp
                first_one = -1
                for i in range(p):
                    if temp[i]:
                        first_one = i
                        break
                if first_one == -1:
                    break
                # Find basis vector with pivot at first_one
                pivot_found = False
                for j in range(len(basis)):
                    if basis[j][first_one]:
                        for i in range(p):
                            temp[i] ^= basis[j][i]
                        pivot_found = True
                        break
                if not pivot_found:
                    basis.append(temp)
                    break
        
        # Generate all combinations
        words_17 = []
        num_basis = len(basis)
        for mask in range(1 << num_basis):
            word = [0] * p
            for i in range(num_basis):
                if (mask >> i) & 1:
                    for j in range(p):
                        word[j] ^= basis[i][j]
            words_17.append(word)
            
        # Extend to 18 bits (append 0)
        words_18 = []
        for w in words_17:
            val = 0
            for i, bit in enumerate(w):
                if bit:
                    val |= (1 << i)
            words_18.append(val)
        return words_18

    qr_words = get_qr_code()
    best_words = qr_words[:]
    best_size = len(qr_words)
    
    # Verify QR code just in case
    if verify(best_words) < MIN_DIST:
        # Fallback to empty if QR failed (shouldn't happen)
        best_words = []
        best_size = 0

    # 2. Seeded Greedy: Try to add more words to the QR code
    # We will run this with a few different random orderings of the remaining candidates
    remaining_candidates = [i for i in range(1 << N) if i not in set(qr_words)]
    
    for restart in range(5): # 5 restarts for seeded greedy
        if time.time() > deadline:
            break
            
        words = qr_words[:]
        blocked = build_blocked(words)
        
        # Shuffle remaining candidates
        candidates = remaining_candidates[:]
        random.shuffle(candidates)
        
        for c in candidates:
            if time.time() > deadline:
                break
            if c not in blocked:
                words.append(c)
                # Update blocked
                for m in bad_masks:
                    blocked.add(c ^ m)
        
        if len(words) > best_size:
            # Verify before accepting
            if verify(words) >= MIN_DIST:
                best_words = words[:]
                best_size = len(words)
            else:
                # If verification fails, it means a bug in blocked set logic or QR code
                # But blocked set logic is sound. 
                # We just discard this attempt.
                pass

    # 3. Pure Greedy: Try from scratch with random restarts
    # This might find a different, larger code structure
    max_pure_restarts = 20
    for restart in range(max_pure_restarts):
        if time.time() > deadline:
            break
            
        words = []
        blocked = set()
        
        candidates = list(range(1 << N))
        random.shuffle(candidates)
        
        for c in candidates:
            if time.time() > deadline:
                break
            if c not in blocked:
                words.append(c)
                for m in bad_masks:
                    blocked.add(c ^ m)
        
        if len(words) > best_size:
            if verify(words) >= MIN_DIST:
                best_words = words[:]
                best_size = len(words)

    return best_words

if __name__ == "__main__":
    words = construct()
    print(f"Constructed {len(words)} words.")
    if len(words) > 0:
        min_d = 18
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                d = (words[i] ^ words[j]).bit_count()
                if d < min_d:
                    min_d = d
        print(f"Min distance: {min_d}")
