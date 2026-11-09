import time
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Precompute popcount lookup for 9-bit values
    pop9 = np.zeros(512, dtype=np.uint8)
    for i in range(512):
        pop9[i] = i.bit_count()

    def fast_popcount18(arr):
        """Vectorized popcount for numpy array of 18-bit values"""
        low = (arr & 0x1FF).astype(np.intp)
        high = ((arr >> 9) & 0x1FF).astype(np.intp)
        return pop9[low] + pop9[high]

    # Try to find the 512-word extended quadratic residue code
    # This is the known maximum for A_2(18,6)
    
    # The extended quadratic residue code of length 18 is related to the (18, 512, 6) code
    # It can be constructed using the quadratic residues modulo 17
    
    # Quadratic residues mod 17: {1, 4, 9, 16, 8, 2, 15, 13}
    # The quadratic residue code over GF(2) with length 17 can be extended to length 18
    
    # Alternative: use the known construction
    # The (18, 512, 6) code is the extended QR code
    
    # Let me construct it using the generator matrix approach
    # The QR(17) code has a generator polynomial related to the minimal polynomial of a primitive 17th root of unity
    
    # Simpler approach: use the fact that A_2(18,6) = 512
    # and try to construct a code of this size
    
    # The extended quadratic residue code of length 18
    # can be constructed as follows:
    # 1. Start with the QR(17) code
    # 2. Add an overall parity check bit
    
    # The QR(17) code has parameters (17, 256, 5)
    # After extension: (18, 256, 6)
    # But we need 512 words, so we need the double coset or similar construction
    
    # Actually, the (18, 512, 6) code is a nonlinear code
    # It can be constructed using the extended Golay code or similar
    
    # Let me try a different approach: construct the code using
    # the quadratic residues
    
    # For the (18, 512, 6) code, one construction is:
    # Take all quadratic residues mod 19 (since 19 is prime)
    # QR mod 19: {1, 4, 9, 16, 6, 17, 11, 7, 5}
    # That's 9 residues
    
    # Actually, let me use a direct construction
    # The (18, 512, 6) code can be constructed as:
    # C = {c in {0,1}^18 : c is in the extended QR code}
    
    # Let me try to use a known construction
    # The extended quadratic residue code of length 18
    
    # QR(18) is not standard. Let me use the (18, 512, 6) code construction
    # from the literature.
    
    # One approach: use the (16, 256, 6) first-order Reed-Muller code RM(1,4)
    # and extend it to length 18
    
    # RM(1,4) has 2^5 = 32 codewords of length 16 and distance 8
    # We can extend it by adding 2 parity bits
    
    # But 32 < 512, so we need a different construction
    
    # Let me try the greedy approach but with a better ordering
    # Start with 0, then add words in order of increasing weight
    
    words = [0]
    words_set = {0}
    words_arr = np.array(words, dtype=np.uint32)
    
    # Generate all 18-bit numbers
    # Process them in a smart order to find a large code quickly
    
    # Order by weight, then by value
    from collections import defaultdict
    
    by_weight = defaultdict(list)
    for c in range(2**N):
        wt = c.bit_count()
        by_weight[wt].append(c)
    
    # Process weights from 6 to 12 (symmetric, so we can stop at 9)
    # But we need all weights to find 512 words
    
    for w in range(6, N + 1):
        if time.time() > deadline - 1.0:
            break
            
        if len(words) >= 512:
            return words
        
        for c in by_weight[w]:
            if time.time() > deadline - 0.5:
                break
                
            if c in words_set:
                continue
                
            # Check distance to all existing words
            c_val = np.uint32(c)
            xor_vals = np.bitwise_xor(words_arr, c_val).astype(np.uint32)
            bit_counts = fast_popcount18(xor_vals)
            
            if np.min(bit_counts) < MIN_DIST:
                continue
            
            # Add to words
            words.append(c)
            words_set.add(c)
            words_arr = np.append(words_arr, c_val)
            
            if len(words) >= 512:
                return words
    
    return words
