import time

def construct():
    N = 18
    MIN_DIST = 6
    
    # 1. Construct the [17, 9, 5] Quadratic Residue Code
    p = 17
    # Quadratic residues mod 17
    # 0, 1, 4, 9, 16, 8, 2, 15, 13, 3, 10, 5, 12, 11, 6, 7, 14 (squares of 0..16)
    # Distinct non-zero: 1, 2, 4, 8, 9, 13, 15, 16
    # Standard QR code generator uses the vector of residues including 0?
    # Let's use the vector with 1s at positions that are QRs mod 17.
    # Position 0 corresponds to 0.
    # Let's define the base vector g of length 17.
    
    def is_qr(x, p):
        if x == 0: return True # 0 is often included in the set for the cyclic code generation context, 
                                # but let's check the weight.
        return pow(x, (p-1)//2, p) == 1

    # Actually, the binary QR code is the span of shifts of the characteristic vector of the set of QRs.
    # Let's create the vector for indices 0..16.
    base_vec = 0
    for i in range(17):
        if i == 0:
            bit = 1 # 0 is a square (0^2)
        else:
            bit = 1 if pow(i, 8, 17) == 1 else 0
        if bit:
            base_vec |= (1 << i)
            
    # Generate 9 basis vectors (shifts 0 to 8)
    # Note: Cyclic shift right by 1 bit.
    # In Python integers, we can mask to 17 bits.
    mask_17 = (1 << 17) - 1
    
    basis = []
    for k in range(9):
        # Shift base_vec right by k? Or left?
        # Cyclic shift: (x >> k) | ((x & ((1<<k)-1)) << (17-k))
        # But we just need a basis. The shifts of a single vector form the code.
        # Are shifts 0..8 linearly independent?
        # Yes, for QR codes of prime length, the first (p+1)/2 shifts are independent.
        # (17+1)/2 = 9.
        
        # Let's do a cyclic shift to the right.
        v = base_vec
        for _ in range(k):
            # Shift right by 1
            lsb = v & 1
            v = (v >> 1) | (lsb << 16)
        basis.append(v)
        
    # Generate all 512 codewords
    codewords = []
    for mask in range(1 << 9):
        w = 0
        for i in range(9):
            if (mask >> i) & 1:
                w ^= basis[i]
        # Extend to length 18 by adding parity bit
        # Parity bit is at position 17.
        # Weight of w in 17 bits:
        wt = w.bit_count()
        parity = 1 if (wt % 2 != 0) else 0
        # If we want even weight, parity = wt % 2.
        # If we want odd weight, parity = 1 - (wt % 2).
        # Standard extension makes weight even.
        # Distance 5 -> 6 (if wt=5, parity=1, total wt=6).
        # Distance 7 -> 8 (if wt=7, parity=1, total wt=8).
        # So min distance becomes 6.
        
        w_18 = w | (parity << 17)
        codewords.append(w_18)
        
    # Verify (optional, but good for safety)
    # The prompt says "Every pair is checked exactly". 
    # Since we constructed it from a known optimal code, it should be valid.
    # But let's add a quick check to ensure no duplicates and min dist >= 6.
    # 512 words is small enough for O(N^2) check in Python? 
    # 512^2 = 262,144 checks. Each check is XOR and bit_count.
    # This is very fast (< 0.1s).
    
    seen = set()
    valid_codewords = []
    for w in codewords:
        if w in seen:
            continue
        seen.add(w)
        valid = True
        for other in valid_codewords:
            if (w ^ other).bit_count() < MIN_DIST:
                valid = False
                break
        if valid:
            valid_codewords.append(w)
            
    return valid_codewords
