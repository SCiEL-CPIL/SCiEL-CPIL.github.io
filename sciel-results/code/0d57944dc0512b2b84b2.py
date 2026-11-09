"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    N = 18
    MIN_D = 6
    
    # We will construct the largest known code A_2(18,6).
    # The known maximum is 256 (from the extended quadratic residue code or similar constructions).
    # Actually, A_2(18,6) is known to be 256.
    # Let's try to find a code of size 256.
    
    # One known construction:
    # Take the [16,4,8] Reed-Muller code RM(1,4) which has 16 words? No, RM(1,4) has 2^(1+4)=32 words, min distance 8.
    # Wait, RM(1,4) is linear, dimension 5, so 32 words.
    # We can extend this?
    
    # Let's look at the extended quadratic residue code.
    # The QR code of length 18? No, QR codes are usually length p-1 where p is prime.
    # p=19 is prime, so length 18 QR code exists?
    # The extended quadratic residue code of length 18 (from p=19) is a [18,9,6] code?
    # Let's check parameters for p=19.
    # The QR code over GF(2) of length 18.
    # If p ≡ 1 mod 4, the QR code is self-dual.
    # 19 ≡ 3 mod 4, so not self-dual.
    # The extended QR code of length 18 has parameters [18, 9, 6]?
    # Let's verify: A_2(18,6) >= 256.
    # If there is a linear [18, 8, 6] code, it has 256 words.
    
    # Let's try to construct a [18, 8, 6] code.
    # We can use the fact that the extended Hamming code [16, 11, 4] is not enough.
    # The Reed-Muller code RM(1,4) is [16, 5, 8].
    # We can take the direct sum with a [2, 1, 2] code? No.
    
    # Alternative approach: Use a greedy algorithm or backtracking to find a large code.
    # Given the 60s limit, a full exhaustive search is impossible.
    # But we can use a known good code.
    
    # Known: A_2(18,6) = 256.
    # Construction:
    # 1. Take the [16, 5, 8] Reed-Muller code C1 (32 words).
    # 2. Take the [2, 1, 2] repetition code C2 (2 words: 00, 11).
    # 3. The direct sum C1 x C2 has 32*2 = 64 words, length 18, distance min(8,2)=2. Too small.
    
    # We need distance 6.
    # Let's look at the "extended quadratic residue" code of length 18.
    # According to tables, A_2(18,6) = 256.
    # A linear [18, 8, 6] code exists.
    
    # Let's try to find it via a simple construction or hardcode a known generator matrix.
    # Since I cannot look up the exact generator matrix offline, I will use a heuristic search
    # or a known construction.
    
    # Actually, a simpler approach:
    # The code C = { (u, v, parity) } ...
    
    # Let's try a different known code:
    # The [18, 8, 6] code can be constructed from the [16, 5, 8] Reed-Muller code?
    # No, 2^8 = 256. RM(1,4) is 2^5=32.
    
    # Let's use a greedy set packing algorithm with a good ordering to find a large code.
    # Or, we can use the fact that the problem allows nonlinear codes.
    
    # However, implementing a fast exact solver for MIP in Python within 60s for N=18 is risky.
    # But we can use a simple backtracking with pruning if we fix the first few words.
    
    # Better: Use a known code.
    # I will implement a constructor for the [18, 8, 6] code if I can derive it.
    # Otherwise, I will use a greedy algorithm that is guaranteed to find at least 32 (the seed)
    # and likely much more.
    
    # Let's try a greedy algorithm with randomization or deterministic ordering.
    # To ensure we get 256, we might need a specific structure.
    
    # Let's try to build the code using the "extended quadratic residue" construction logic.
    # For p=19, the quadratic residues mod 19 are:
    # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=6, 6^2=17, 7^2=11, 8^2=7, 9^2=5
    # QR = {1, 4, 5, 6, 7, 9, 11, 16, 17}
    # Non-QR = {2, 3, 8, 10, 12, 13, 14, 15, 18}
    
    # The extended QR code of length 18 is formed by taking the QR code of length 17 (p-1=18? No, p=19 -> length 18)
    # Wait, the QR code of length n=p-1=18.
    # The generator matrix is 9x18?
    
    # Let's try to generate the [18, 9, 6] code? If it exists, it has 512 words.
    # But A_2(18,6) is 256. So the max size is 256.
    # So we need a [18, 8, 6] code (256 words) or a nonlinear code of size 256.
    
    # Let's use a simple, robust greedy algorithm that adds codewords in a deterministic order
    # and checks all pairs. To maximize the size, we should try to add words that are "far" from existing ones.
    # But a simple greedy might get stuck at a small size.
    
    # Given the constraints, I will implement a backtracking search that fixes the first word to 0
    # and then tries to add words in increasing order, pruning branches where no more words can be added
    # (using a simple upper bound or just backtracking).
    # However, 2^18 = 262,144 candidates. Even with pruning, this might be too slow in Python.
    
    # Alternative: Use the known result.
    # I will hardcode a generator matrix for a [18, 8, 6] code if I can find one.
    # Since I don't have it, I will use a heuristic that is very fast:
    # 1. Start with 0.
    # 2. Add words in order of increasing weight, then increasing value.
    # 3. Check distance to all existing words.
    
    # To improve the greedy result, we can use a "local search" or "simulated annealing" style approach,
    # but that is complex.
    
    # Let's try a different strategy:
    # The task says "a 512-word extended quadratic-residue construction is independently verified as a reference."
    # This implies that the reference solution might have 512 words?
    # But A_2(18,6) is 256.
    # Wait, if the reference has 512 words, then A_2(18,6) >= 512.
    # Let me double-check the value of A_2(18,6).
    # According to the online tables (Brouwer), A_2(18,6) = 256.
    # Why does the prompt say "a 512-word ... construction is independently verified"?
    # Maybe it's a [18, 9, 6] code?
    # If A_2(18,6) = 512, then the seed (32) is far from optimal.
    # Let's assume the prompt is correct and A_2(18,6) is at least 512.
    # Actually, for n=18, d=6:
    # If we take the [16, 5, 8] RM code (32 words) and extend it...
    # What if we take the [18, 9, 6] code?
    # If such a code exists, it has 512 words.
    
    # Let's try to construct a [18, 9, 6] code.
    # One way: Take the [16, 6, 4] code? No.
    # Take the [18, 9, 6] extended QR code.
    # For p=19, the QR code is [18, 9, 6].
    # Let's generate it.
    
    # The QR code of length 18 (p=19) is generated by the matrix:
    # Rows are the quadratic residues and their multiples?
    # Actually, the generator matrix G is 9x18.
    # The first row is the all-ones vector (for the extended code).
    # The other rows are based on the QR set.
    
    # Let's define the QR set Q = {1, 4, 5, 6, 7, 9, 11, 16, 17} in Z_19.
    # The code is the span of vectors v_i for i in Q, where v_i has 1s at positions j such that j-i is in Q?
    # This is getting complicated to derive from scratch without a reference.
    
    # Let's use a simpler, guaranteed approach:
    # Use a greedy algorithm with a twist: try all possible first words? No.
    # Fix the first word to 0.
    # Then, use a backtracking search with strong pruning.
    # Pruning: If the number of remaining candidates that are at distance >= 6 from all selected words is less than the current best, prune.
    # This is a standard maximum independent set algorithm.
    
    # To make it fast in Python:
    # 1. Precompute the distance between all pairs? 2^18 * 2^18 is too big.
    # 2. Use bitsets. Represent each codeword as an integer.
    # 3. For a set of selected words S, the set of valid candidates C is { x | dist(x, s) >= 6 for all s in S }.
    # 4. We can maintain C as a bitset of size 2^18 (262,144 bits = 32,768 bytes).
    # 5. When we add a word w to S, we update C by removing all x such that dist(x, w) < 6.
    # 6. We need a fast way to compute the set of x at distance < 6 from w.
    #    This set has size sum_{i=0}^{5} C(18, i) = 1 + 18 + 153 + 816 + 3060 + 8568 = 12616.
    #    So for each added word, we clear ~12k bits in C.
    # 7. The search tree depth is at most 512 (if we find 512 words).
    #    512 * 12k operations = ~6 million bit operations. This is very fast in Python if optimized.
    
    # Algorithm:
    # 1. Initialize C = all 1s (all 2^18 words are candidates).
    # 2. Fix the first word to 0. Remove all words at distance < 6 from 0 from C.
    # 3. Recursively select the next word from C (smallest index first for determinism).
    # 4. For each candidate w in C:
    #    a. Add w to the code.
    #    b. Save C.
    #    c. Update C by removing all x at distance < 6 from w.
    #    d. Recurse.
    #    e. Restore C.
    # 5. Track the maximum size of the code found.
    
    # To speed up the "remove all x at distance < 6 from w" step:
    # Precompute for each w, the list of x at distance < 6? No, that's 2^18 * 12k integers, too big.
    # Instead, for a given w, iterate through all x in C? No, C is a bitset.
    # We can iterate through the set bits of C? No, we need to clear bits in C based on w.
    # We can precompute a list of offsets for each distance d=0..5.
    # For a given w, the words at distance d are w XOR mask, where mask has weight d.
    # We can generate all masks of weight <= 5.
    # There are 12616 such masks.
    # For each mask, x = w XOR mask.
    # We clear bit x in C.
    
    # Implementation details:
    # - Use an integer as a bitset? 2^18 bits = 2^18 integers? No, Python integers are arbitrary precision.
    #   We can represent C as a Python integer where bit i is 1 if i is in C.
    #   Clearing bit x: C &= ~(1 << x).
    #   Checking if x is in C: (C >> x) & 1.
    #   This is efficient for small x, but 1 << 262143 is a huge integer.
    #   Python handles big integers well, but shifting by 200k bits might be slow.
    #   Alternative: Use a list of 64-bit integers or a bytearray.
    #   Let's use a list of integers, each 64 bits wide. Size = 2^18 / 64 = 4096 integers.
    
    # Let's implement with a list of 64-bit chunks.
    
    N = 18
    MIN_D = 6
    NUM_WORDS = 1 << N
    CHUNKS = NUM_WORDS // 64
    
    # Precompute masks of weight <= 5
    # We need all masks m such that m.bit_count() <= 5
    # There are 12616 such masks.
    # Storing them is fine.
    masks = []
    for m in range(NUM_WORDS):
        if m.bit_count() <= 5:
            masks.append(m)
    
    # This precomputation takes a while (2^18 iterations).
    # 262,144 iterations is fast in Python.
    
    def clear_bits_in_range(C, w, masks):
        # C is a list of 64-bit ints.
        # For each mask in masks, x = w ^ mask.
        # Clear bit x in C.
        for mask in masks:
            x = w ^ mask
            chunk_idx = x >> 6
            bit_idx = x & 63
            C[chunk_idx] &= ~(1 << bit_idx)
            
    def count_bits(C):
        count = 0
        for chunk in C:
            count += chunk.bit_count()
        return count

    # Initial C: all 1s
    C = [(1 << 64) - 1] * CHUNKS
    
    # Fix first word to 0
    # Remove all x at distance < 6 from 0
    # These are exactly the masks in our list (since 0 ^ mask = mask)
    for mask in masks:
        x = mask
        chunk_idx = x >> 6
        bit_idx = x & 63
        C[chunk_idx] &= ~(1 << bit_idx)
        
    # Now C contains all words at distance >= 6 from 0.
    # We will search for the maximum independent set in the remaining graph.
    
    best_code = [0]
    best_size = 1
    
    # Recursive backtracking
    # To avoid deep recursion issues, we can limit the depth or use an iterative approach.
    # But 512 depth is fine.
    
    # Optimization: If count_bits(C) + len(current_code) <= best_size, prune.
    
    def search(current_code, C):
        nonlocal best_code, best_size
        
        # Pruning
        remaining = count_bits(C)
        if len(current_code) + remaining <= best_size:
            return
            
        # If no candidates left, update best
        if remaining == 0:
            if len(current_code) > best_size:
                best_size = len(current_code)
                best_code = current_code[:]
            return
        
        # Pick the next candidate
        # We pick the smallest index x in C
        # To find the smallest index, we scan the chunks.
        x = -1
        for i in range(CHUNKS):
            if C[i] != 0:
                # Find the least significant set bit
                # (C[i] & -C[i]).bit_length() - 1
                lsb = C[i] & -C[i]
                bit_pos = lsb.bit_length() - 1
                x = (i << 6) + bit_pos
                break
                
        if x == -1:
            return
            
        # Option 1: Include x
        # Save C
        C_save = C[:]
        
        # Add x to current_code
        current_code.append(x)
        
        # Update C: remove all y at distance < 6 from x
        # This is clear_bits_in_range(C, x, masks)
        # But we must only remove y that are still in C?
        # No, clear_bits_in_range clears bits regardless.
        # This is correct.
        clear_bits_in_range(C, x, masks)
        
        # Recurse
        search(current_code, C)
        
        # Option 2: Exclude x
        # We need to remove x from C (and its neighbors? No, just x for the exclude branch)
        # Wait, the standard Bron-Kerbosch with coloring or simple backtracking:
        # We partition the candidates into those that can be included and those that cannot.
        # Here, we simply try including x, then exclude x and continue with the rest.
        # To exclude x, we just clear bit x in C and recurse.
        
        # Restore C for the exclude branch
        C[:] = C_save
        # Clear x from C
        chunk_idx = x >> 6
        bit_idx = x & 63
        C[chunk_idx] &= ~(1 << bit_idx)
        
        search(current_code, C)
        
        # Restore C for the caller
        # Not needed if we pass copies, but we are modifying in place.
        # We restored C_save before the exclude branch.
        # After the exclude branch, C is modified.
        # We should restore C_save before returning to the caller?
        # The caller expects C to be as it was when we entered?
        # No, the caller passed C. We modified it.
        # We should restore C to its state at entry.
        C[:] = C_save
        
        current_code.pop()

    # Start search
    search(best_code, C)
    
    return best_code

# EVOLVE-BLOCK-END
