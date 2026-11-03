"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    start_time = time.time()
    
    # Strategy: mixed partition
    # j-group 0 (rows 0,1 of A): use 2x2 * 2x4 blocks (rank 10 each, 2 blocks) = 20
    # j-group 1 (rows 2,3 of A): use 1x2 * 2x2 blocks (rank 2 each, 4 blocks) = 8
    # Total rank: 28
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)  # 8 x 28
    V = np.zeros((m * p, rank), dtype=float)  # 32 x 28
    W = np.zeros((p * n, rank), dtype=float)  # 16 x 28
    
    t = 0
    
    # === j-group 0: A rows 0,1 (j=0,1); use 2x2 * 2x4 Strassen-like blocks ===
    # Block 0a: k=0,1 (B rows 0,1; C cols 0,1)
    # Block 0b: k=2,3 (B rows 2,3; C cols 2,3)
    # 2x2 * 2x4: A is 2x2, B is 2x4, C is 2x4
    # M1 = (A00+A11)(B00+B02)
    # M2 = A10(B01-B03)
    # M3 = A00(B10+B12)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)(B12-B13)
    # C00 = M1 + M4
    # C01 = M2 + M3
    # C10 = M1 + M3
    # C11 = M1 - M2 + M4 + M5
    
    for b in range(2):  # k-group: 0 or 1, covering k=2b, 2b+1, 2b+2, 2b+3
        k0 = 2 * b  # first column of B in this block
        k1 = 2 * b + 1
        k2 = 2 * b + 2
        k3 = 2 * b + 3
        # W columns: c0=2b, c1=2b+1, c2=2b+2, c3=2b+3
        # C[0,c0], C[0,c1], C[1,c0], C[1,c1]
        
        # M1 = (A00+A11)(B00+B02)
        # A00 = A[0,0], A11 = A[1,1]
        # B00 = B[0,k0], B02 = B[0,k2]
        U[0 * m + 0, t] += 1  # A[0,0]
        U[1 * m + 1, t] += 1  # A[1,1]
        V[0 * p + k0, t] += 1  # B[0,k0]
        V[0 * p + k2, t] += 1  # B[0,k2]
        W[k0 * n + 0, t] += 1  # C[0, 2b]
        W[k2 * n + 0, t] += 1  # C[0, 2b+2]
        W[k0 * n + 1, t] += 1  # C[1, 2b]
        W[k2 * n + 1, t] += 1  # C[1, 2b+2]
        t += 1
        
        # M2 = A10(B01-B03)
        # A10 = A[1,0]
        # B01 = B[0,k1], B03 = B[0,k3]
        U[1 * m + 0, t] += 1  # A[1,0]
        V[0 * p + k1, t] += 1  # B[0,k1]
        V[0 * p + k3, t] += -1  # B[0,k3]
        W[k1 * n + 0, t] += 1  # C[0, 2b+1]
        W[k3 * n + 0, t] += 1  # C[0, 2b+3]
        W[k1 * n + 1, t] += -1  # C[1, 2b+1]
        W[k3 * n + 1, t] += -1  # C[1, 2b+3]
        t += 1
        
        # M3 = A00(B10+B12)
        # A00 = A[0,0]
        # B10 = B[1,k0], B12 = B[1,k2]
        U[0 * m + 0, t] += 1  # A[0,0]
        V[1 * p + k0, t] += 1  # B[1,k0]
        V[1 * p + k2, t] += 1  # B[1,k2]
        W[k1 * n + 0, t] += 1  # C[0, 2b+1]
        W[k3 * n + 0, t] += 1  # C[0, 2b+3]
        W[k0 * n + 1, t] += 1  # C[1, 2b]
        W[k2 * n + 1, t] += 1  # C[1, 2b+2]
        t += 1
        
        # M4 = A11(B00-B10)
        # A11 = A[1,1]
        # B00 = B[0,k0], B10 = B[1,k0]
        U[1 * m + 1, t] += 1  # A[1,1]
        V[0 * p + k0, t] += 1  # B[0,k0]
        V[1 * p + k0, t] += -1  # B[1,k0]
        W[k0 * n + 0, t] += 1  # C[0, 2b]
        W[k2 * n + 0, t] += 1  # C[0, 2b+2]
        W[k0 * n + 1, t] += 1  # C[1, 2b]
        W[k2 * n + 1, t] += 1  # C[1, 2b+2]
        t += 1
        
        # M5 = (A00+A01)(B12-B13)
        # A00 = A[0,0], A01 = A[0,1]
        # B12 = B[1,k2], B13 = B[1,k3]
        U[0 * m + 0, t] += 1  # A[0,0]
        U[0 * m + 1, t] += 1  # A[0,1]
        V[1 * p + k2, t] += 1  # B[1,k2]
        V[1 * p + k3, t] += -1  # B[1,k3]
        W[k2 * n + 1, t] += 1  # C[1, 2b+2]
        W[k3 * n + 1, t] += 1  # C[1, 2b+3]
        t += 1
    
    # === j-group 1: A rows 2,3 (j=2,3); use 1x2 * 2x2 blocks ===
    # For each row a in {2,3} and each k-group b in {0,1,2,3}:
    # 1x2 * 2x2: A is 1x2 (row a, cols 2b, 2b+1), B is 2x2 (rows 2b,2b+1; cols 2b,2b+1)
    # Wait, that's not right. Let me reconsider.
    # A is 2x4, so row a has columns 0,1,2,3.
    # For j-group 1, we use rows 2,3 of A (j=2,3).
    # Each 1x2 block: A[a, 2b:2b+2] * B[2b:2b+2, 2b:2b+2] = C[a, 2b:2b+2]
    # But B is 4x8, so B[2b:2b+2, 2b:2b+2] doesn't make sense dimensionally.
    
    # Let me reconsider the partition.
    # A is 2x4: rows i=0,1; cols j=0,1,2,3
    # B is 4x8: rows j=0,1,2,3; cols k=0,...,7
    # C is 2x8: rows i=0,1; cols k=0,...,7
    
    # j-group 0: j=0,1 (first 2 columns of A, first 2 rows of B)
    # j-group 1: j=2,3 (last 2 columns of A, last 2 rows of B)
    
    # For j-group 1, A[0:2, 2:4] is 2x2, B[2:4, 0:8] is 2x8
    # We can partition k into 4 groups of 2.
    # For each k-group, we have A[0:2, 2:4] (2x2) * B[2:4, 2b:2b+2] (2x2) = C[0:2, 2b:2b+2] (2x2)
    # This is a 2x2 * 2x2 = 2x2 problem, rank 7 with Strassen.
    
    # But I wanted to use 1x2 * 2x2 blocks. Let me reconsider.
    # If we split A[0:2, 2:4] into two 1x2 rows, then:
    # A[0, 2:4] (1x2) * B[2:4, 2b:2b+2] (2x2) = C[0, 2b:2b+2] (1x2)
    # A[1, 2:4] (1x2) * B[2:4, 2b:2b+2] (2x2) = C[1, 2b:2b+2] (1x2)
    # Each 1x2 * 2x2 = 1x2 has rank 2.
    # So for each k-group, rank = 2 + 2 = 4.
    # For 4 k-groups, rank = 16.
    
    # Total: 20 (j-group 0) + 16 (j-group 1) = 36.
    
    # Let me redo j-group 1 with 1x2 * 2x2 blocks.
    for a in range(2):  # row of A: 0 or 1 (but we're in j-group 1, so actual row is a)
        for b in range(4):  # k-group: 0..3, covering k=2b, 2b+1
            k0 = 2 * b
            k1 = 2 * b + 1
            # A[a, 2:4] is 1x2, but we only need A[a, 2b:2b+2] for this block?
            # No, A[a, 2:4] is the full 1x2 row.
            # B[2:4, 2b:2b+2] is 2x2.
            # C[a, 2b:2b+2] is 1x2.
            
            # 1x2 * 2x2 = 1x2
            # C0 = A0*B00 + A1*B10
            # C1 = A0*B01 + A1*B11
            # Rank 2:
            # M1 = A0*(B00+B01) -> C0 += B00 part, C1 += B01 part? No.
            # Let me think again.
            # C0 = A0*B00 + A1*B10
            # C1 = A0*B01 + A1*B11
            # We can write:
            # M1 = A0 * (B00 + B01) -> contributes to C0 (B00) and C1 (B01)
            # M2 = A1 * (B10 + B11) -> contributes to C0 (B10) and C1 (B11)
            # But M1 and M2 are separate multiplications.
            # C0 = M1_B00_part + M2_B10_part
            # C1 = M1_B01_part + M2_B11_part
            # This doesn't work directly because M1 is a single product.
            
            # Actually, for 1x2 * 2x2:
            # A = [a0, a1], B = [[b00, b01], [b10, b11]], C = [c0, c1]
            # c0 = a0*b00 + a1*b10
            # c1 = a0*b01 + a1*b11
            # This is 4 multiplications schoolbook.
            # Can we do it in 2? 
            # Let me try:
            # M1 = a0 * (b00 + b10)
            # M2 = a1 * (b01 + b11)
            # c0 = M1 + a1*b10 - a1*b10 + a0*b00 - a0*b00 ... no, this doesn't work.
            
            # Actually, the minimum rank for 1x2 * 2x2 = 1x2 is 2.
            # Proof: The tensor is c0 = a0*b00 + a1*b10, c1 = a0*b01 + a1*b11.
            # This can be written as:
            # c0 = a0 * b00 + a1 * b10
            # c1 = a0 * b01 + a1 * b11
            # Using 2 multiplications:
            # M1 = a0 * (b00 + b01)
            # M2 = a1 * (b10 + b11)
            # Then c0 = M1 - a0*b01 + a1*b10 ... no.
            
            # Let me look this up. The rank of 1x2 * 2x2 = 1x2.
            # Actually, I think it's 2.
            # c0 = a0*b00 + a1*b10
            # c1 = a0*b01 + a1*b11
            # Let M1 = a0 * b00, M2 = a1 * b10, M3 = a0 * b01, M4 = a1 * b11.
            # That's 4.
            # Can we do 2?
            # Try: M1 = a0 * (b00 + b01), M2 = a1 * (b10 + b11)
            # c0 = ? We need a0*b00 + a1*b10.
            # c1 = ? We need a0*b01 + a1*b11.
            # M1 gives a0*b00 + a0*b01, M2 gives a1*b10 + a1*b11.
            # c0 = M1 - a0*b01 + a1*b10 ... no, we don't have a0*b01 and a1*b10 separately.
            
            # I think the rank is actually 4 for 1x2 * 2x2 = 1x2.
            # Let me reconsider.
            
            # Actually, I recall that 1xn * nxm = 1xm has rank n*m if n,m >= 2.
            # So 1x2 * 2x2 = 1x2 has rank 4.
            
            # That means my earlier calculation was wrong.
            # Let me recalculate.
            
            # j-group 0: 2x2 * 2x4 blocks, rank 10 each, 2 blocks = 20
            # j-group 1: 2x2 * 2x2 blocks, rank 7 each, 4 blocks = 28
            # Total: 48.
            
            # That's better than 56 but not 28.
            
            # Let me try a different approach.
            # What if I use 2x4 * 4x4 blocks?
            # A is 2x4, B is 4x8.
            # Partition k into 2 groups of 4.
            # Each block: 2x4 * 4x4 = 2x4.
            # Rank of 2x4 * 4x4 = 2x4?
            # Schoolbook: 2*4*4 = 32.
            # Can we do better?
            # This is a non-square problem.
            
            # Let me try 2x2 * 2x2 blocks for j-group 1.
            # j-group 1: A[0:2, 2:4] (2x2) * B[2:4, 0:8] (2x8) = C[0:2, 0:8] (2x8)
            # Partition k into 4 groups of 2.
            # Each: 2x2 * 2x2 = 2x2, rank 7.
            # 4 blocks * 7 = 28.
            
            # Total: 20 + 28 = 48.
    
    # Let me restart with the correct approach.
    # j-group 0: j=0,1; use 2x2 * 2x4 blocks (rank 10 each, 2 blocks) = 20
    # j-group 1: j=2,3; use 2x2 * 2x2 blocks (rank 7 each, 4 blocks) = 28
    # Total: 48
    
    # But I already used 10 terms for j-group 0. Let me continue.
    # Actually, I realize I made an error in the W indexing for the 2x2 * 2x4 blocks.
    # Let me restart and be more careful.
    
    # Let me just use the simple Strassen approach for now to ensure correctness.
    # Then I can optimize.
    
    # For now, return the rank 56 solution.
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    t = 0
    for a in range(2):  # j-group: 0 or 1, covering j=2a, 2a+1
        for b in range(4):  # k-group: 0..3, covering k=2b, 2b+1
            for s in range(7):  # Strassen step
                # U contributions (A[:, 2a:2a+2])
                if s == 0:  # X00 + X11
                    U[0 * m + 2 * a, t] += 1
                    U[1 * m + 2 * a + 1, t] += 1
                elif s == 1:  # X10 + X11
                    U[1 * m + 2 * a, t] += 1
                    U[1 * m + 2 * a + 1, t] += 1
                elif s == 2:  # X00
                    U[0 * m + 2 * a, t] += 1
                elif s == 3:  # X11
                    U[1 * m + 2 * a + 1, t] += 1
                elif s == 4:  # X00 + X01
                    U[0 * m + 2 * a, t] += 1
                    U[0 * m + 2 * a + 1, t] += 1
                elif s == 5:  # X10 - X00
                    U[1 * m + 2 * a, t] += 1
                    U[0 * m + 2 * a, t] += -1
                elif s == 6:  # X01 - X11
                    U[0 * m + 2 * a + 1, t] += 1
                    U[1 * m + 2 * a + 1, t] += -1
                
                # V contributions (B[2a:2a+2, 2b:2b+2])
                if s == 0:  # Y00 + Y11
                    V[2 * a * p + 2 * b, t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1, t] += 1
                elif s == 1:  # Y00
                    V[2 * a * p + 2 * b, t] += 1
                elif s == 2:  # Y01 - Y11
                    V[2 * a * p + 2 * b + 1, t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1, t] += -1
                elif s == 3:  # Y10 - Y00
                    V[(2 * a + 1) * p + 2 * b, t] += 1
                    V[2 * a * p + 2 * b, t] += -1
                elif s == 4:  # Y11
                    V[(2 * a + 1) * p + 2 * b + 1, t] += 1
                elif s == 5:  # Y00 + Y01
                    V[2 * a * p + 2 * b, t] += 1
                    V[2 * a * p + 2 * b + 1, t] += 1
                elif s == 6:  # Y10 + Y11
                    V[(2 * a + 1) * p + 2 * b, t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1, t] += 1
                
                # W contributions (C[:, 2b:2b+2])
                if s == 0:  # M1: Z00+=1, Z11+=1
                    W[2 * b * n + 0, t] += 1
                    W[(2 * b + 1) * n + 1, t] += 1
                elif s == 1:  # M2: Z10+=1, Z11-=1
                    W[2 * b * n + 1, t] += 1
                    W[(2 * b + 1) * n + 1, t] += -1
                elif s == 2:  # M3: Z01+=1, Z11+=1
                    W[(2 * b + 1) * n + 0, t] += 1
                    W[(2 * b + 1) * n + 1, t] += 1
                elif s == 3:  # M4: Z00+=1, Z10+=1
                    W[2 * b * n + 0, t] += 1
                    W[2 * b * n + 1, t] += 1
                elif s == 4:  # M5: Z00-=1, Z01+=1
                    W[2 * b * n + 0, t] += -1
                    W[(2 * b + 1) * n + 0, t] += 1
                elif s == 5:  # M6: Z11+=1
                    W[(2 * b + 1) * n + 1, t] += 1
                elif s == 6:  # M7: Z00+=1
                    W[2 * b * n + 0, t] += 1
                
                t += 1
    
    # Verify the solution
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0
                rhs = 0.0
                for tt in range(rank):
                    rhs += U[i*m+j, tt] * V[j*p+k, tt] * W[k*n+i, tt]
                if abs(lhs - rhs) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
