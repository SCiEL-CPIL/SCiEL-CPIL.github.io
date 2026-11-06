import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    def generate_extended_qr(poly_coeffs, n=17, k=9):
        # poly_coeffs: list of 0s and 1s for x^0 to x^(n-k)
        # Construct generator matrix rows (cyclic shifts of g)
        # g is represented as an integer
        g = 0
        for i, c in enumerate(poly_coeffs):
            if c:
                g |= (1 << i)
        
        # Generate 9 basis vectors (cyclic shifts)
        # Note: Cyclic shifts of g in GF(2)[x] / (x^n - 1)
        # Since n=17 is prime, x^17 - 1 = (x-1) * (x^16 + ... + 1)
        # The QR code is a subcode of the cyclic code.
        # Actually, the generator matrix of the cyclic code is just the cyclic shifts.
        basis = []
        for i in range(k):
            # Shift g by i positions
            # (g * 2^i) % (2^n - 1)  <-- Careful with bit operations for cyclic shift
            # Cyclic shift left by i: ((g << i) | (g >> (n - i))) & ((1 << n) - 1)
            # But we are in GF(2). Addition is XOR.
            # The code is linear span of cyclic shifts.
            # We just need the 9 shifts to form a basis?
            # Yes, for a cyclic code of dimension k, the k cyclic shifts of g form a basis.
            shift = (g << i) | (g >> (n - i))
            shift &= (1 << n) - 1
            basis.append(shift)
            
        # Generate all 2^k codewords
        codewords = []
        for m in range(1 << k):
            word = 0
            for j in range(k):
                if (m >> j) & 1:
                    word ^= basis[j]
            # Append parity bit
            parity = word.bit_count() % 2
            # Extend to length 18
            # The problem statement says "length 18". 
            # Usually extension appends a bit. 
            # If we append at the end (bit 17), the integer value increases by parity * 2^17.
            # Or is it prepended? 
            # "18-bit words". 
            # Standard extension appends a parity check bit.
            # Let's assume the bit index 17 is the parity bit.
            extended_word = word | (parity << n)
            codewords.append(extended_word)
            
        return codewords

    def check_min_distance(codewords):
        if not codewords:
            return 0
        min_dist = 18
        # Optimization: check against first few, or all pairs if small enough
        # 512 words -> 512*511/2 ~ 130k pairs. Fast in Python.
        for i in range(len(codewords)):
            wi = codewords[i]
            for j in range(i + 1, len(codewords)):
                wj = codewords[j]
                dist = (wi ^ wj).bit_count()
                if dist < min_dist:
                    min_dist = dist
                    if min_dist < 6:
                        return min_dist
        return min_dist

    # Candidate 1: g(x) = 1 + x + x^2 + x^4 + x^7 + x^8
    # Coeffs for x^0 to x^8: [1, 1, 1, 0, 1, 0, 0, 1, 1]
    poly1 = [1, 1, 1, 0, 1, 0, 0, 1, 1]
    
    # Candidate 2: g(x) = 1 + x + x^2 + x^3 + x^5 + x^8
    # Coeffs: [1, 1, 1, 1, 0, 1, 0, 0, 1]
    poly2 = [1, 1, 1, 1, 0, 1, 0, 0, 1]
    
    # Candidate 3: g(x) = 1 + x + x^3 + x^4 + x^6 + x^8 ?
    # Let's just try the two most likely ones.
    
    code1 = generate_extended_qr(poly1)
    dist1 = check_min_distance(code1)
    
    if dist1 >= 6:
        return code1
    
    if time.time() > deadline:
        return code1 # Return best effort

    code2 = generate_extended_qr(poly2)
    dist2 = check_min_distance(code2)
    
    if dist2 >= 6:
        return code2

    # Fallback: Return the one with larger min distance
    if dist1 > dist2:
        return code1
    else:
        return code2

