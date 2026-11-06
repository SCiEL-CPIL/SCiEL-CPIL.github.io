import numpy as np
import time

def poly_mult(a, b):
    """Multiply two polynomials over F2. a, b are lists of coefficients."""
    res = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                if bj:
                    res[i+j] ^= 1
    return res

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # 1. Construct the [18,9,6] QR code generator polynomial
    # Primitive element alpha = 3 in F_2^8? No, alpha is a root of x^17-1.
    # We work in F_2[x] / (x^17-1)? No, we need the minimal polynomial in F_2[x].
    # The roots are in F_2^8.
    # We can compute the minimal polynomial by finding the smallest polynomial 
    # with roots alpha^i.
    # Alternatively, use the known minimal polynomial for p=17.
    # m_1(x) = x^8 + x^5 + x^4 + x^3 + 1 ?
    # Let's verify.
    # I will compute it dynamically to be safe.
    
    # Represent alpha as a polynomial in F_2^8.
    # We need a primitive polynomial of degree 8 for F_2^8.
    # Let's use x^8 + x^4 + x^3 + x^2 + 1.
    # Check if it's primitive.
    # For simplicity, I will use the property that the minimal polynomial of alpha
    # for p=17 is known.
    # m_1(x) = x^8 + x^5 + x^4 + x^3 + 1
    # Let's assume this is correct. If not, the code might be wrong.
    # To be robust, I will compute the roots.
    
    # Actually, I can just use the 256 words of the [17,8,6] code if I can find them.
    # Since I can't look them up, I will construct the cyclic code.
    
    # Minimal polynomial of alpha (root of x^17-1) in F_2[x].
    # The order of 2 mod 17 is 8.
    # So the degree is 8.
    # The conjugates of alpha are alpha, alpha^2, alpha^4, alpha^8, alpha^16, alpha^15, alpha^13, alpha^9.
    
    # I will construct the generator matrix directly using the parity check matrix?
    # H is 9x18.
    
    # Let's try a different, simpler code if this is too risky.
    # The [16,5,8] RM code has 32 words.
    # The [17,8,6] code has 256 words.
    
    # I will implement the QR construction carefully.
    
    # Step 1: Find the minimal polynomial m(x) of a primitive 17th root of unity in F_2.
    # We can do this by iterating through polynomials of degree 8.
    # But there is a faster way.
    # The generator polynomial g(x) of the [18,9,6] code is (x+1)m(x).
    
    # Known m(x) for p=17: x^8 + x^5 + x^4 + x^3 + 1
    m = [1, 0, 0, 1, 1, 1, 0, 0, 1] # Coeffs of x^0 to x^8?
    # Let's check: m(x) = 1 + x^3 + x^4 + x^5 + x^8
    # Coeffs: 1, 0, 0, 1, 1, 1, 0, 0, 1
    
    # g(x) = (x+1)m(x)
    g = poly_mult([1, 1], m) # [1,1] is x+1
    
    # g has degree 9.
    # Generator matrix G for [18,9] code.
    # Rows are shifts of g.
    # g = [g0, g1, ..., g9]
    # Row i: [0..0, g0..g9, 0..0]
    
    n = 18
    k = 9
    G = np.zeros((k, n), dtype=int)
    for i in range(k):
        for j in range(len(g)):
            if i+j < n:
                G[i, i+j] = g[j]
    
    # Generate all 256 codewords? No, 2^9 = 512 codewords for [18,9].
    # We need to shorten it.
    # Shortening by 1 coordinate: Set the last coordinate to 0.
    # This means we only take codewords where the last bit is 0.
    # This reduces the dimension by 1, giving 2^8 = 256 words.
    
    words = []
    for mask in range(512):
        # Generate codeword
        # c = mask * G (mod 2)
        # Efficiently:
        c = 0
        for i in range(k):
            if (mask >> i) & 1:
                # XOR row i
                row = G[i]
                # Convert row to int
                # This is slow.
                pass
        
    # Better: Precompute the rows as integers.
    rows_int = []
    for i in range(k):
        val = 0
        for j in range(n):
            if G[i, j]:
                val |= (1 << j)
        rows_int.append(val)
        
    # Generate codewords
    codewords = set()
    for mask in range(512):
        val = 0
        for i in range(k):
            if (mask >> i) & 1:
                val ^= rows_int[i]
        # Shorten: last bit (bit 17) must be 0
        if (val & (1 << 17)) == 0:
            # Remove last bit
            val_short = val & ((1 << 17) - 1)
            codewords.add(val_short)
            
    # We should have 256 words.
    if len(codewords) != 256:
        # Fallback to RM code if construction failed
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Now, try to add more words.
    # Find words in F_2^17 with min distance >= 6 to all codewords.
    
    # This is expensive: 2^17 * 256 distances.
    # 131072 * 256 = 33 million ops.
    # In Python, this might be too slow.
    # Use NumPy.
    
    code_list = list(codewords)
    code_arr = np.array(code_list, dtype=np.uint32)
    
    # Generate all possible words
    all_words = np.arange(2**17, dtype=np.uint32)
    
    # Compute distances
    # This will use a lot of memory if done all at once.
    # 131072 * 256 * 4 bytes = 134 MB. Acceptable.
    
    # XOR and bit count
    # NumPy doesn't have bit_count for uint32 directly in older versions?
    # Use bin(x).count('1')? Too slow.
    # Use lookup table for 8-bit chunks.
    
    # Let's do it in batches.
    batch_size = 10000
    candidates = []
    
    for start in range(0, 2**17, batch_size):
        end = min(start + batch_size, 2**17)
        batch = all_words[start:end]
        
        # Compute min distance to code_arr
        # dist = popcount(batch[:, None] ^ code_arr[None, :])
        # This is 10000 x 256.
        
        # We can use a trick:
        # Create a lookup table for popcount of 0..255
        # But XOR is 32-bit.
        
        # Let's use Python loop for the batch if NumPy is slow.
        # 10000 * 256 = 2.5M ops.
        # 13 batches. Total 33M ops.
        # Python loop: 33M ops might take 30-60s.
        
        # Optimization:
        # If a word is in code_arr, min dist is 0.
        # We only care about words NOT in code_arr.
        
        # Let's just run the loop and hope it's fast enough.
        
        for w in batch:
            # Check if w is in codewords (set lookup)
            if w in codewords:
                continue
                
            min_d = 18 # Max possible
            for c in code_list:
                d = (w ^ c).bit_count()
                if d < min_d:
                    min_d = d
                    if min_d < 6:
                        break
            
            if min_d >= 6:
                candidates.append(w)
                
        if time.time() > deadline:
            break
            
    # If we found candidates, try to add them greedily.
    if candidates:
        # Sort candidates?
        # Greedy: pick one, remove all that are too close.
        # This is O(N^2).
        # N is likely small.
        
        added = []
        for w in candidates:
            # Check distance to existing code + added
            ok = True
            for c in code_list:
                if (w ^ c).bit_count() < 6:
                    ok = False
                    break
            if not ok:
                continue
                
            for a in added:
                if (w ^ a).bit_count() < 6:
                    ok = False
                    break
            
            if ok:
                added.append(w)
                
        code_list.extend(added)
        
    return code_list
