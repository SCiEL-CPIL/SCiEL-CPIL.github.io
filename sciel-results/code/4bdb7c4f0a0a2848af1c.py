import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: Kronecker product structure
    # Split 4x4 * 4x6 into 2x2 block structure:
    # A = [[A00, A01], [A10, A11]]  (each 2x2)
    # B = [[B00, B01], [B10, B11]]  (each 2x3)
    # C = [[C00, C01], [C10, C11]]  (each 2x3)
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Use Winograd (rank 7) for the 2x2 * 2x2 part and schoolbook for the 2x3 extension.
    # The 2x2 * 2x3 tensor can be viewed as (2x2*2x2) ⊗ (1x1*1x1) for each of 3 k-columns,
    # but more efficiently: for each output column k in 0,1,2, we compute a 2x2 result
    # from A (2x2) and B[:,k] (2x1). But B[:,k] is shared structure.
    #
    # Better: The 2x2 * 2x3 tensor rank. 
    # Schoolbook: 12 terms.
    # We can use Winograd for the "row" structure: 
    # For 2x2 * 2x1 (matrix-vector), rank is 5 (Winograd F(2) for 2-point conv).
    # Actually 2x2 * 2x1: c0 = a00*b0 + a01*b1, c1 = a10*b0 + a11*b1
    # Rank 4 schoolbook. Can we do rank 3? 
    # (a00+a11)(b0) + (a10+a11)(b0) ... no.
    # Winograd for 2x2*2x1: M1=(a00+a11)b0, M2=a10*b0, M3=a01*b1, M4=a11*b1
    # c0 = M1 + M3 - M2 - M4 + M2... hmm.
    # Actually standard Winograd F(2) for length-2 convolution:
    # y0 = x0*h0 + x1*h1, y1 = x1*h0 (if we think of it as conv)
    # But matrix-vector 2x2*2x1 is just two dot products.
    # Let's stick with known results.
    #
    # Key insight from cell findings: Kronecker product of Strassen (rank 7) for 2x2x2
    # and schoolbook for the remaining dimension gives rank 7 * 3 = 21 for 2x2*2x3?
    # No, that's not quite right. The tensor is M(2,2,3).
    # M(2,2,3) = M(2,2,1) ⊗ M(1,1,3)? No.
    #
    # Let's think of M(2,2,3) as computing C (2x3) = A (2x2) * B (2x3).
    # This is equivalent to 3 independent 2x2 * 2x1 multiplications IF the B columns were independent,
    # but they share the same A. However, in tensor decomposition, we can't share A entries
    # across different "k" slices in a simple Kronecker way unless we restructure.
    #
    # Actually, M(2,2,3) can be decomposed by treating it as:
    # For each k in {0,1,2}, we compute C[:,k] = A * B[:,k].
    # The tensor is the sum over k of (M(2,2,1) for column k).
    # But the A part is the same for all k. In the tensor decomposition,
    # U[i*m+j, r] doesn't depend on k. So we can't simply sum 3 separate decompositions
    # because the U factors would need to be identical for the same A entries, which they are,
    # but the V and W factors differ.
    #
    # Wait, the tensor T[i,j,k] for M(2,2,3) is 1 for all i,j,k in valid ranges.
    # If we write T = sum_{k=0}^{2} T_k where T_k[i,j,k'] = delta_{k,k'} * 1,
    # then T_k is a tensor supported only on slice k.
    # The rank of T_k is the rank of M(2,2,1) which is 4 (schoolbook) or less.
    # But since they are on disjoint slices, rank(T) = sum rank(T_k) = 3 * rank(M(2,2,1)).
    # rank(M(2,2,1)) = 4 (I don't think it can be less than 4 for 2x2*2x1).
    # So rank(M(2,2,3)) <= 12.
    #
    # But wait, we can do better by not separating the k slices!
    # The Winograd trick works because it combines terms across the "convolution" dimension.
    # For M(2,2,2), Winograd gets rank 7 < 8.
    # For M(2,2,3), can we get less than 12?
    #
    # Let's try a different approach: use the known rank 7 for 2x2*2x2 and extend.
    # Write B as [B_left (2x2), B_right (2x1)].
    # C = A * B_left + A * B_right.
    # A * B_left: M(2,2,2) tensor, rank 7.
    # A * B_right: M(2,2,1) tensor, rank 4.
    # Total rank: 7 + 4 = 11.
    #
    # Let's verify this works.
    # For A*B_left (2x2 * 2x2), use Winograd rank 7.
    # For A*B_right (2x2 * 2x1), use schoolbook rank 4.
    # The A entries are shared! In the tensor decomposition, U[i*m+j, r] is the same for both.
    # So we just concatenate the decompositions.
    # Total rank for 2x2*2x3 = 7 + 4 = 11.
    #
    # Now for the full 4x4 * 4x6:
    # We have 4 block pairs, each 2x2 * 2x3.
    # But they share A and B blocks!
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # A00 appears in C00 and C01. A01 in C00 and C01. Etc.
    # So we can't just use 4 independent 2x2*2x3 decompositions.
    #
    # Let's group by A blocks:
    # A00 * [B00, B01] + A01 * [B10, B11] -> [C00, C01]
    # This is A00 * B_top + A01 * B_bottom where B_top is 2x6, B_bottom is 2x6.
    # So it's a 2x2 * 2x6 multiplication!
    # Similarly for the bottom: A10 * B_top + A11 * B_bottom -> [C10, C11], also 2x2 * 2x6.
    #
    # So the full problem is two 2x2 * 2x6 multiplications.
    # rank(M(2,2,6)) = ?
    # M(2,2,6) = sum_{k=0}^{5} M(2,2,1) for column k, but with shared A.
    # Using the split: B (2x6) = [B_left (2x2), B_mid (2x2), B_right (2x2)].
    # A * B_left: rank 7
    # A * B_mid: rank 7
    # A * B_right: rank 7
    # Total: 21.
    #
    # So rank(M(2,2,6)) <= 21.
    # Full problem: 2 * 21 = 42.
    #
    # Let's verify: 42 < 48 (current) < 96 (schoolbook).
    #
    # Let's construct this.
    
    # Winograd rank-7 decomposition for 2x2 * 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    # M1 = (a00+a11)(b00+b10)
    # M2 = (a10+a11)(b00)
    # M3 = a00(b10-b00)
    # M4 = a11(b00-b10)
    # M5 = (a00+a01)(b11+b01)  -- wait, standard Winograd for 2x2:
    # Let me use the standard one:
    # M1 = (a00+a11)(b00+b10)
    # M2 = (a10+a11)(b00)
    # M3 = a00(b10-b00)
    # M4 = a11(b00-b10)
    # M5 = a01(b11+b01)  -- no, let me look up the exact Winograd F(2) for 2x2 matrix mult.
    #
    # Standard Winograd 2x2*2x2 rank 7:
    # m1 = (a00+a11)(b00+b10)
    # m2 = (a10+a11)(b00)
    # m3 = a00(b10-b00)
    # m4 = a11(b00-b10)
    # m5 = (a00+a01)(b01+b11)
    # m6 = (a10+a11)(b11)
    # m7 = a10(b11-b01)  -- hmm, I need to be careful.
    #
    # Let me just use the well-known decomposition:
    # M1 = (a00+a11)(b00+b10)
    # M2 = (a10+a11)(b00)
    # M3 = a00(b10-b00)
    # M4 = a11(b00-b10)
    # M5 = (a00+a01)(b11+b01)
    # M6 = (a10+a11)(b11)
    # M7 = a10(b11-b01)  -- wait, this doesn't look symmetric.
    #
    # Actually, let me just use a simpler approach. I'll use the rank-7 Winograd
    # decomposition as defined by the standard formula:
    #
    # For A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]:
    # m1 = (a0+a3)(b0+b2)
    # m2 = (a2+a3)(b0)
    # m3 = a0(b2-b0)
    # m4 = a3(b0-b2)
    # m5 = (a0+a1)(b1+b3)
    # m6 = (a2+a3)(b3)
    # m7 = a2(b3-b1)
    #
    # c00 = m1 + m3 - m4 + m2 - m2 = m1 + m3 - m4  ... let me verify.
    # Actually c00 = a0*b0 + a1*b2.
    # m1 = a0*b0 + a0*b2 + a3*b0 + a3*b2
    # m3 = a0*b2 - a0*b0
    # m4 = a3*b0 - a3*b2
    # m1 + m3 - m4 = a0*b0 + a0*b2 + a3*b0 + a3*b2 + a0*b2 - a0*b0 - a3*b0 + a3*b2
    # = 2*a0*b2 + 2*a3*b2. That's not right.
    #
    # Let me use a verified Winograd decomposition. I'll use the one from:
    # m1 = (a00+a11)(b00+b10)
    # m2 = (a10+a11)(b00)
    # m3 = a00(b10-b00)
    # m4 = a11(b00-b10)
    # m5 = (a00+a01)(b01+b11)
    # m6 = (a10+a11)(b11)
    # m7 = a10(b11-b01)
    #
    # c00 = m1 + m3 - m4
    # c01 = m5 + m7 - m6  -- wait, let me check.
    # c01 = a00*b01 + a01*b11
    # m5 = a00*b01 + a00*b11 + a01*b01 + a01*b11
    # m7 = a10*b11 - a10*b01
    # m6 = a10*b11 + a11*b11
    # This is getting messy. Let me just use the schoolbook for 2x2*2x2 (rank 4)
    # and schoolbook for 2x2*2x1 (rank 4) and see if the total is better.
    #
    # Alternative: just use rank 11 for 2x2*2x3 (7+4) and verify computationally.
    # But I need the actual coefficients. Let me use a simpler known decomposition.
    #
    # Actually, let me just use the block approach with rank 12 for each 2x2*2x3
    # but exploit sharing. The current program uses rank 48 (4*12).
    # If I can get rank 11 for 2x2*2x3, and there are 4 such blocks but with sharing,
    # the total might be around 2*21 = 42 as I computed above.
    #
    # Let me just implement the 2x2*2x6 approach with rank 21 each, total 42.
    # For 2x2*2x6, split B into three 2x2 blocks, each using rank 7 Winograd.
    # I need the Winograd rank-7 coefficients. Let me hardcode a verified one.
    
    # Winograd rank-7 for 2x2*2x2 (verified):
    # U factors (for A entries a00,a01,a10,a11):
    # M1: a00=1, a11=1
    # M2: a10=1, a11=1
    # M3: a00=1
    # M4: a11=1
    # M5: a00=1, a01=1
    # M6: a10=1, a11=1
    # M7: a10=1
    #
    # V factors (for B entries b00,b01,b10,b11):
    # M1: b00=1, b10=1
    # M2: b00=1
    # M3: b10=1, b00=-1
    # M4: b00=1, b10=-1
    # M5: b01=1, b11=1
    # M6: b11=1
    # M7: b11=1, b01=-1
    #
    # W factors (for C entries c00,c01,c10,c11):
    # M1: c00=1, c10=1, c01=1, c11=1  -- no, W should be for the output.
    #
    # Let me just verify computationally. I'll build the decomposition and check.
    
    def winograd_2x2x2():
        """Return U, V, W for 2x2*2x2 with rank 7."""
        # A indices: 0=a00, 1=a01, 2=a10, 3=a11
        # B indices: 0=b00, 1=b01, 2=b10, 3=b11
        # C indices: 0=c00, 1=c01, 2=c10, 3=c11
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1 = (a00+a11)(b00+b10)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[2, 0] = 1
        W[0, 0] = 1; W[2, 0] = 1  # contributes to c00 and c10
        
        # M2 = (a10+a11)(b00)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # contributes to c10
        
        # M3 = a00(b10-b00)
        U[0, 2] = 1
        V[2, 2] = 1; V[0, 2] = -1
        W[0, 2] = 1  # contributes to c00
        
        # M4 = a11(b00-b10)
        U[3, 3] = 1
        V[0, 3] = 1; V[2, 3] = -1
        W[2, 3] = 1  # contributes to c10
        
        # M5 = (a00+a01)(b01+b11)
        U[0, 4] = 1; U[1, 4] = 1
        V[1, 4] = 1; V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1  # contributes to c01 and c11
        
        # M6 = (a10+a11)(b11)
        U[2, 5] = 1; U[3, 5] = 1
        V[3, 5] = 1
        W[3, 5] = 1  # contributes to c11
        
        # M7 = a10(b11-b01)
        U[2, 6] = 1
        V[3, 6] = 1; V[1, 6] = -1
        W[3, 6] = 1  # contributes to c11
        
        return U, V, W
    
    # Verify Winograd
    U_w, V_w, W_w = winograd_2x2x2()
    # Check: for all i,j,k, sum_r U[i,r]*V[j,r]*W[k,r] should give the MM tensor
    # For 2x2*2x2: T[i,j,k] = 1 for all i,j,k
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for r in range(7):
                    s += U_w[i, r] * V_w[j, r] * W_w[k, r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
    if not valid:
        # Fall back to schoolbook for 2x2*2x2
        U_w = np.eye(4)
        V_w = np.eye(4)
        W_w = np.eye(4)
        # This gives rank 4, not 7, but it's correct.
        # Actually, let me just use schoolbook for everything if Winograd fails.
        pass
    
    # Now build the full 4x4 * 4x6 decomposition.
    # Approach: Two 2x2 * 2x6 multiplications.
    # Top: A_top (2x2) * B_full (2x6) -> C_top (2x6)
    # Bottom: A_bottom (2x2) * B_full (2x6) -> C_bottom (2x6)
    #
    # Wait, that's not right. The structure is:
    # C_top = A00*B00 + A01*B10  (where B00,B10 are 2x3)
    # C_bottom = A10*B00 + A11*B10
    #
    # So it's not simply A_top * B_full. Let me reconsider.
    #
    # Actually, if I write B as a 4x6 matrix with blocks:
    # B = [[B00, B01], [B10, B11]]
    # Then C = A*B where A = [[A00, A01], [A10, A11]]
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # I can group as:
    # [C00, C01] = A00*[B00, B01] + A01*[B10, B11]
    # [C10, C11] = A10*[B00, B01] + A11*[B10, B11]
    #
    # So [C00, C01] = A00 * B_top + A01 * B_bottom where B_top = [B00, B01] (2x6), B_bottom = [B10, B11] (2x6)
    # This is a 2x2 * 2x6 multiplication!
    # Similarly for the bottom.
    #
    # So total rank = 2 * rank(M(2,2,6)).
    # rank(M(2,2,6)) <= 3 * 7 = 21 (using Winograd for each 2x2*2x2 block of B).
    # Total: 42.
    
    # Build M(2,2,6) decomposition with rank 21.
    # B is 2x6, split into three 2x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6].
    # For each block, use Winograd rank 7.
    
    U_226 = np.zeros((4, 21))
    V_226 = np.zeros((12, 21))  # 2x6 = 12 entries
    W_226 = np.zeros((12, 21))  # 2x6 = 12 entries
    
    for block in range(3):
        U_b, V_b, W_b = winograd_2x2x2()
        U_226[:, block*7:(block+1)*7] = U_b
        V_226[block*4:(block+1)*4, block*7:(block+1)*7] = V_b
        W_226[block*4:(block+1)*4, block*7:(block+1)*7] = W_b
    
    # Verify M(2,2,6)
    valid_226 = True
    for i in range(2):
        for j in range(2):
            for k in range(6):
                s = 0.0
                for r in range(21):
                    s += U_226[i, r] * V_226[j*6+k, r] * W_226[k*2+i, r]
                if abs(s - 1.0) > 1e-9:
                    valid_226 = False
    if not valid_226:
        # Fall back to schoolbook for 2x2*2x6
        U_226 = np.zeros((4, 12))
        V_226 = np.zeros((12, 12))
        W_226 = np.zeros((12, 12))
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    col = i*12 + j*6 + k
                    U_226[i, col] = 1
                    V_226[j*6+k, col] = 1
                    W_226[k*2+i, col] = 1
        # This gives rank 12, not 21.
    
    # Now build the full 4x4*4x6.
    # Top half: A00, A01 (rows 0,1 of A) * B (full 4x6) -> C00, C01 (rows 0,1 of C)
    # This is M(2,2,6) where A is 2x2, B is 2x6 (but B is actually the full 4x6 viewed as two 2x6 blocks).
    # Wait, I need to be more careful.
    #
    # C_top = A00 * B_top + A01 * B_bottom
    # where B_top = B[0:2, :] (2x6), B_bottom = B[2:4, :] (2x6).
    #
    # In the tensor decomposition, U factors for A00 and A01 are different.
    # U for A00: rows 0,1 of the 2x2 A block.
    # U for A01: rows 2,3 of the 4x4 A matrix, but in the 2x2 block sense, it's the "second" 2x2 block.
    #
    # Let me just build it directly.
    # U has shape (16, rank) for 4x4.
    # V has shape (24, rank) for 4x6.
    # W has shape (24, rank) for 6x4.
    
    # Top: A00 * B_top + A01 * B_bottom
    # A00 entries: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,4,5
    # A01 entries: A[0,2], A[0,3], A[1,2], A[1,3] -> U indices 2,3,6,7
    # B_top entries: B[0,0:6], B[1,0:6] -> V indices 0:6, 6:12
    # B_bottom entries: B[2,0:6], B[3,0:6] -> V indices 12:18, 18:24
    # C_top entries: C[0,0:6], C[1,0:6] -> W indices 0:6, 6:12 (W[k*n+i] = W[k*4+i])
    # C[0,k] -> W[k*4+0], C[1,k] -> W[k*4+1]
    
    rank_total = 42
    U = np.zeros((16, rank_total))
    V = np.zeros((24, rank_total))
    W = np.zeros((24, rank_total))
    
    # Top half: use M(2,2,6) decomposition
    # A00 * B_top: U_226 rows 0,1,2,3 map to A00 entries
    # A01 * B_bottom: U_226 rows 0,1,2,3 map to A01 entries
    # But wait, in M(2,2,6), the A is 2x2 and B is 2x6.
    # Here, A00 is 2x2, B_top is 2x6. So it's exactly M(2,2,6).
    # Similarly A01 is 2x2, B_bottom is 2x6.
    
    # For A00 * B_top:
    # U_226[i, r] -> U[A00_row, A00_col, r] where A00_row in {0,1}, A00_col in {0,1}
    # A00[i,j] is at U[i*4+j]
    for i in range(2):
        for j in range(2):
            U[i*4+j, 0:21] = U_226[i, 0:21]
    # V_226[j*6+k, r] -> V[B_top_row, B_top_col, r] where B_top_row in {0,1}, B_top_col in {0,5}
    # B_top[j,k] is at V[j*6+k]
    V[0:12, 0:21] = V_226[0:12, 0:21]
    # W_226[k*2+i, r] -> W[C_top_col, C_top_row, r] where C_top_col in {0,5}, C_top_row in {0,1}
    # C_top[i,k] is at W[k*4+i]
    # W_226 index k*2+i corresponds to C[i,k] in the 2x6 output.
    # In the full C, C_top[i,k] is at W[k*4+i].
    for k in range(6):
        for i in range(2):
            W[k*4+i, 0:21] = W_226[k*2+i, 0:21]
    
    # For A01 * B_bottom:
    # A01[i,j] is at U[i*4+2+j]
    for i in range(2):
        for j in range(2):
            U[i*4+2+j, 21:42] = U_226[i, 0:21]
    # B_bottom[j,k] is at V[(j+2)*6+k]
    V[12:24, 21:42] = V_226[0:12, 0:21]
    # C_top[i,k] is at W[k*4+i] (same as before, but now from the second term)
    for k in range(6):
        for i in range(2):
            W[k*4+i, 21:42] = W_226[k*2+i, 0:21]
    
    # Bottom half: A10 * B_top + A11 * B_bottom
    # A10[i,j] is at U[(i+2)*4+j]
    # A11[i,j] is at U[(i+2)*4+2+j]
    # C_bottom[i,k] is at W[k*4+(i+2)]
    
    # But wait, I've already used rank 42 for the top half. I need another 42 for the bottom.
    # Total rank = 84.
    # That's not better than 96, but it's close.
    #
    # Actually, I can reuse the same U_226, V_226, W_226 for the bottom half.
    # Let me extend the rank to 84.
    
    rank_total = 84
    U = np.zeros((16, rank_total))
    V = np.zeros((24, rank_total))
    W = np.zeros((24, rank_total))
    
    # Top: A00 * B_top + A01 * B_bottom (rank 42)
    for i in range(2):
        for j in range(2):
            U[i*4+j, 0:21] = U_226[i, 0:21]
            U[i*4+2+j, 21:42] = U_226[i, 0:21]
    V[0:12, 0:21] = V_226[0:12, 0:21]
    V[12:24, 21:42] = V_226[0:12, 0:21]
    for k in range(6):
        for i in range(2):
            W[k*4+i, 0:21] = W_226[k*2+i, 0:21]
            W[k*4+i, 21:42] = W_226[k*2+i, 0:21]
    
    # Bottom: A10 * B_top + A11 * B_bottom (rank 42)
    for i in range(2):
        for j in range(2):
            U[(i+2)*4+j, 42:63] = U_226[i, 0:21]
            U[(i+2)*4+2+j, 63:84] = U_226[i, 0:21]
    V[0:12, 42:63] = V_226[0:12, 0:21]
    V[12:24, 63:84] = V_226[0:12, 0:21]
    for k in range(6):
        for i in range(2):
            W[k*4+(i+2), 42:63] = W_226[k*2+i, 0:21]
            W[k*4+(i+2), 63:84] = W_226[k*2+i, 0:21]
    
    # Verify the full decomposition
    def verify_full(U, V, W, n, m, p):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_full(U, V, W, n, m, p):
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
