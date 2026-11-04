import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We are decomposing A (2x3) * B (3x7) = C (2x7)
    # Split A into A_left (2x2) and A_right (2x1)
    # Split B into B_top (2x7) and B_bottom (1x7)
    # C = A_left * B_top + A_right * B_bottom
    
    # Rank contributions:
    # A_left (2x2) * B_top (2x7):
    #   Split B_top into 3 blocks of 2x2 and 1 block of 2x1.
    #   3 * Strassen(2x2, 2x2) rank 7 = 21
    #   1 * Standard(2x2, 2x1) rank 4 = 4
    #   Total for A_left * B_top = 25
    # A_right (2x1) * B_bottom (1x7):
    #   Standard rank 2 * 7 = 14
    # Total Rank = 39
    
    rank = 39
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Indices:
    # A: i in {0,1}, j in {0,1,2} -> index i*3 + j
    # B: j in {0,1,2}, k in {0..6} -> index j*7 + k
    # C: i in {0,1}, k in {0..6} -> index k*2 + i (Wait, W is p*n, index k*n + i)
    
    # A_left corresponds to j=0,1
    # A_right corresponds to j=2
    # B_top corresponds to j=0,1
    # B_bottom corresponds to j=2
    
    # Part 1: A_left * B_top
    # A_left elements: a00, a01, a10, a11
    # B_top elements: b00..b06, b10..b16
    
    # We will fill U, V, W for the 25 multiplications.
    r = 0
    
    # Helper to add a term
    # u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w
    # U[u_idx, r] += coeff_u
    # V[v_idx, r] += coeff_v
    # W[w_idx, r] += coeff_w
    
    # Strassen Block 1: B_top columns 0,1
    # B_block = [[b00, b01], [b10, b11]]
    # A_block = [[a00, a01], [a10, a11]]
    
    # M1 = (a00+a11) * (b00+b11)
    U[0, r] += 1; U[3, r] += 1 # a00, a11 (indices 0, 3)
    V[0, r] += 1; V[11, r] += 1 # b00, b11 (indices 0, 11)
    # C contributions: C00, C11
    # C00 index 0*2+0 = 0? No, W is p*n, index k*n+i.
    # C00: i=0, k=0 -> index 0
    # C11: i=1, k=1 -> index 1*2+1 = 3
    # But Strassen M1 contributes to C00 and C11 with +1.
    # Wait, the decomposition is sum of products.
    # The term is U * V * W.
    # We need to distribute the coefficients to W such that the sum works out.
    # This is tricky. The standard way is that each "multiplication" r corresponds to a specific linear combination of A, B, and C.
    # For Strassen, we define 7 multiplications.
    # M1 = L1(A) * L1(B). This contributes to C via L1(C).
    # We need to find L1(C) such that the total sum reconstructs C.
    
    # Actually, it's easier to just build the U, V, W matrices for the known Strassen identity.
    # Let's define the 7 Strassen multiplications for a 2x2 * 2x2 block.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the global indices.
    # For the first block (cols 0,1 of B_top):
    # A indices: 0(a), 1(b), 3(c), 4(d)  (since j=0,1 -> 0,1,3,4)
    # B indices: 0(e), 1(f), 7(g), 8(h) (since j=0,1 -> 0,1,7,8)
    # C indices:
    # C00 (i=0, k=0) -> W index 0
    # C01 (i=0, k=1) -> W index 1
    # C10 (i=1, k=0) -> W index 2
    # C11 (i=1, k=1) -> W index 3
    
    # Let's create a function to add a Strassen block
    def add_strassen_block(a_idx, b_idx, c_idx, start_r):
        # a_idx: [a00, a01, a10, a11]
        # b_idx: [b00, b01, b10, b11]
        # c_idx: [c00, c01, c10, c11]
        
        # M1
        U[a_idx[0], start_r] += 1
        U[a_idx[3], start_r] += 1
        V[b_idx[0], start_r] += 1
        V[b_idx[3], start_r] += 1
        W[c_idx[0], start_r] += 1 # C00
        W[c_idx[3], start_r] += 1 # C11
        r1 = start_r + 1
        
        # M2
        U[a_idx[2], r1] += 1
        U[a_idx[3], r1] += 1
        V[b_idx[0], r1] += 1
        W[c_idx[2], r1] += 1 # C10
        W[c_idx[3], r1] += -1 # C11
        r2 = r1 + 1
        
        # M3
        U[a_idx[0], r2] += 1
        V[b_idx[1], r2] += 1
        V[b_idx[3], r2] += -1
        W[c_idx[1], r2] += 1 # C01
        W[c_idx[3], r2] += 1 # C11
        r3 = r2 + 1
        
        # M4
        U[a_idx[3], r3] += 1
        V[b_idx[2], r3] += 1
        V[b_idx[0], r3] += -1
        W[c_idx[0], r3] += 1 # C00
        W[c_idx[2], r3] += 1 # C10
        r4 = r3 + 1
        
        # M5
        U[a_idx[0], r4] += 1
        U[a_idx[1], r4] += 1
        V[b_idx[3], r4] += 1
        W[c_idx[0], r4] += -1 # C00
        W[c_idx[1], r4] += 1 # C01
        r5 = r4 + 1
        
        # M6
        U[a_idx[2], r5] += 1
        U[a_idx[0], r5] += -1
        V[b_idx[0], r5] += 1
        V[b_idx[1], r5] += 1
        W[c_idx[3], r5] += 1 # C11
        r6 = r5 + 1
        
        # M7
        U[a_idx[1], r6] += 1
        U[a_idx[3], r6] += -1
        V[b_idx[2], r6] += 1
        V[b_idx[3], r6] += 1
        W[c_idx[0], r6] += 1 # C00
        return r6 + 1

    # Block 1: B_top cols 0,1
    # A: a00(0), a01(1), a10(3), a11(4)
    # B: b00(0), b01(1), b10(7), b11(8)
    # C: c00(0), c01(1), c10(2), c11(3)
    r = add_strassen_block([0, 1, 3, 4], [0, 1, 7, 8], [0, 1, 2, 3], r)
    
    # Block 2: B_top cols 2,3
    # B: b02(2), b03(3), b12(9), b13(10)
    # C: c02(4), c03(5), c12(6), c13(7)
    r = add_strassen_block([0, 1, 3, 4], [2, 3, 9, 10], [4, 5, 6, 7], r)
    
    # Block 3: B_top cols 4,5
    # B: b04(4), b05(5), b14(11), b15(12)
    # C: c04(8), c05(9), c14(10), c15(11)
    r = add_strassen_block([0, 1, 3, 4], [4, 5, 11, 12], [8, 9, 10, 11], r)
    
    # Block 4: B_top col 6 (2x1)
    # Standard 2x2 * 2x1
    # A: a00(0), a01(1), a10(3), a11(4)
    # B: b06(6), b16(13)
    # C: c06(12), c16(13)
    # M1 = a00 * b06
    U[0, r] += 1
    V[6, r] += 1
    W[12, r] += 1 # C06
    r += 1
    # M2 = a01 * b16
    U[1, r] += 1
    V[13, r] += 1
    W[12, r] += 1 # C06
    r += 1
    # M3 = a10 * b06
    U[3, r] += 1
    V[6, r] += 1
    W[13, r] += 1 # C16
    r += 1
    # M4 = a11 * b16
    U[4, r] += 1
    V[13, r] += 1
    W[13, r] += 1 # C16
    r += 1
    
    # Part 2: A_right * B_bottom
    # A_right: a02(2), a12(5)
    # B_bottom: b20(14), b21(15), b22(16), b23(17), b24(18), b25(19), b26(20)
    # C: c00(0)..c06(12), c10(13)..c16(19)
    # Wait, C indices for i=0 are 0..6, for i=1 are 7..13?
    # W index is k*n + i.
    # k=0..6, i=0 -> 0..6
    # k=0..6, i=1 -> 2,4,6,8,10,12,14? No.
    # n=2.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # ...
    # k=6, i=0 -> 12
    # k=6, i=1 -> 13
    
    # So C0k is index 2k, C1k is index 2k+1.
    
    # M = a02 * b2k
    for k in range(7):
        U[2, r] += 1 # a02
        V[14 + k, r] += 1 # b2k
        W[2*k, r] += 1 # C0k
        r += 1
        
    # M = a12 * b2k
    for k in range(7):
        U[5, r] += 1 # a12
        V[14 + k, r] += 1 # b2k
        W[2*k + 1, r] += 1 # C1k
        r += 1
        
    # Check r
    assert r == rank, f"Rank mismatch: {r} vs {rank}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
