import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 4
    
    # We will build the decomposition for 4x4 matrices.
    # This is done by applying Strassen's algorithm to 2x2 blocks.
    
    # First, define the decomposition for 2x2 matrices (rank 7).
    # Indices for 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # a = i*2 + j
    
    # The 7 Strassen products for 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We need U2, V2, W2 of shape (4, 7) such that T_2x2[a,b,c] = sum_k U2[a,k] V2[b,k] W2[c,k]
    
    # Let's construct U2, V2, W2 explicitly.
    # k=0 corresponds to M1, k=1 to M2, etc.
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A0 + A3)(B0 + B3)
    # A0 is idx 0, A3 is idx 3. B0 is idx 0, B3 is idx 3.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W2[c, k] is the coefficient of M_k in C_c.
    # c=0 (C00): M1(1), M4(1), M5(-1), M7(1) -> indices 0, 3, 4, 6
    W2[0, 0] = 1
    W2[0, 3] = 1
    W2[0, 4] = -1
    W2[0, 6] = 1
    
    # c=1 (C01): M3(1), M5(1) -> indices 2, 4
    W2[1, 2] = 1
    W2[1, 4] = 1
    
    # c=2 (C10): M2(1), M4(1) -> indices 1, 3
    W2[2, 1] = 1
    W2[2, 3] = 1
    
    # c=3 (C11): M1(1), M2(-1), M3(1), M6(1) -> indices 0, 1, 2, 5
    W2[3, 0] = 1
    W2[3, 1] = -1
    W2[3, 2] = 1
    W2[3, 5] = 1
    
    # Now U2 and V2.
    # U2[a, k] is coeff of A_a in the first factor of M_k.
    # V2[b, k] is coeff of B_b in the second factor of M_k.
    
    # k=0 (M1): (A0 + A3)(B0 + B3)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    
    # k=1 (M2): (A2 + A3)B0  (Note: A10 is idx 2, A11 is idx 3, B00 is idx 0)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    
    # k=2 (M3): A0(B1 - B3) (A00 is idx 0, B01 is idx 1, B11 is idx 3)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    
    # k=3 (M4): A3(B2 - B0) (A11 is idx 3, B10 is idx 2, B00 is idx 0)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    
    # k=4 (M5): (A0 + A1)B3 (A00 is idx 0, A01 is idx 1, B11 is idx 3)
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    
    # k=5 (M6): (A2 - A0)(B0 + B1) (A10 is idx 2, A00 is idx 0, B00 is idx 0, B01 is idx 1)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    
    # k=6 (M7): (A1 - A3)(B2 + B3) (A01 is idx 1, A11 is idx 3, B10 is idx 2, B11 is idx 3)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # Verify 2x2 decomposition
    # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    # a=0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # b=0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # c=0(0,0), 1(0,1), 2(1,0), 3(1,1)
    
    # Check one: a=0(0,0), b=0(0,0) -> contributes to c=0(0,0).
    # T[0,0,0] should be 1.
    # sum_k U2[0,k] V2[0,k] W2[0,k]
    # k=0: 1*1*1 = 1
    # k=1: 0
    # k=2: 1*0*0 = 0
    # k=3: 0
    # k=4: 1*0*0 = 0
    # k=5: -1*1*0 = 0
    # k=6: 0
    # Sum = 1. Correct.
    
    # Now construct 4x4 decomposition.
    # A is 4x4, viewed as 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where Aij is 2x2.
    # Index in A: a = i*4 + j.
    # Block A00 corresponds to a in {0,1,2,3} (i=0, j=0..3? No, i,j in {0,1} for blocks)
    # Block (bi, bj) contains elements where i in {2bi, 2bi+1} and j in {2bj, 2bj+1}.
    # So elements of A00 are a=0,1,2,3.
    # Elements of A01 are a=4,5,6,7.
    # Elements of A10 are a=8,9,10,11.
    # Elements of A11 are a=12,13,14,15.
    
    # The Strassen algorithm for 4x4 uses 7 products of 2x2 matrices.
    # Let's call the outer Strassen indices K in {0..6}.
    # For each K, we have a product M_K = U_K(A) * V_K(B).
    # U_K(A) is a 2x2 matrix of blocks, each block is a linear combination of A's blocks.
    # Actually, U_K is a 2x2 matrix of coefficients for the blocks.
    # Let U_K be a 2x2 matrix where U_K[ii, jj] is the coefficient of block A[jj, ii]? No.
    # The product is M_K = (sum_{ii,jj} U_K[ii, jj] A[ii, jj]) * (sum_{kk,ll} V_K[kk, ll] B[kk, ll]).
    # This is a product of two 2x2 matrices (which are themselves sums of blocks).
    # Wait, the Strassen algorithm computes M_K = X_K * Y_K where X_K and Y_K are 2x2 matrices.
    # X_K is a linear combination of the 4 blocks of A.
    # Y_K is a linear combination of the 4 blocks of B.
    
    # Let's define the 7 combinations for X_K and Y_K.
    # K=0: X0 = A00 + A11, Y0 = B00 + B11
    # K=1: X1 = A10 + A11, Y1 = B00
    # K=2: X2 = A00, Y2 = B01 - B11
    # K=3: X3 = A11, Y3 = B10 - B00
    # K=4: X4 = A00 + A01, Y4 = B11
    # K=5: X5 = A10 - A00, Y5 = B00 + B01
    # K=6: X6 = A01 - A11, Y6 = B10 + B11
    
    # Each M_K is a 2x2 matrix.
    # M_K = X_K * Y_K.
    # Since X_K and Y_K are 2x2 matrices, their product M_K is a 2x2 matrix.
    # We can decompose the 2x2 matrix multiplication into 7 rank-1 terms using U2, V2, W2.
    # So M_K = sum_{k=0}^6 u2_k(X_K) v2_k(Y_K) w2_k(M_K)?
    # No. M_K is a matrix. The decomposition of the multiplication of two 2x2 matrices X and Y into Z is:
    # Z_c = sum_{a,b} X_a Y_b T2[a,b,c].
    # Here X and Y are the block matrices.
    # So M_K is the result of multiplying X_K and Y_K.
    # We can express M_K as a sum of 7 rank-1 terms in the space of 2x2 matrices.
    # Let the 7 terms for the 2x2 multiplication be indexed by k.
    # M_K = sum_{k=0}^6 (X_K^T U2[:,k]) (Y_K^T V2[:,k]) W2[:,k]?
    # This is getting complicated.
    
    # Alternative approach:
    # The total rank is 49.
    # We can generate the 49 vectors U, V, W directly.
    # For each outer K (0..6) and inner k (0..6), we have a term.
    # The term corresponds to:
    # A-part: The linear combination of A's elements that forms the "first factor" of the inner term k of the outer term K.
    # B-part: Similarly for B.
    # C-part: The linear combination of C's elements that receives the result.
    
    # Let's define the coefficients for the blocks.
    # For each K, define X_K as a 2x2 matrix of coefficients for A blocks.
    # X_K[ii, jj] is the coefficient of A[jj, ii]? No, A[ii, jj].
    # Let's use 0-based indexing for blocks.
    # A00 is block (0,0), A01 is (0,1), A10 is (1,0), A11 is (1,1).
    
    X_coeffs = [
        [[1, 0], [0, 1]], # K=0: A00 + A11
        [[0, 0], [1, 1]], # K=1: A10 + A11
        [[1, 0], [0, 0]], # K=2: A00
        [[0, 0], [0, 1]], # K=3: A11
        [[1, 1], [0, 0]], # K=4: A00 + A01
        [[0, 0], [1, -1]], # K=5: A10 - A00 (Note: A10 is (1,0), A00 is (0,0)) -> X5[1,0]=1, X5[0,0]=-1
        [[0, 1], [0, -1]]  # K=6: A01 - A11 (A01 is (0,1), A11 is (1,1)) -> X6[0,1]=1, X6[1,1]=-1
    ]
    
    Y_coeffs = [
        [[1, 0], [0, 1]], # K=0: B00 + B11
        [[1, 0], [0, 0]], # K=1: B00
        [[0, 1], [0, -1]], # K=2: B01 - B11
        [[0, -1], [1, 0]], # K=3: B10 - B00 (B10 is (1,0), B00 is (0,0)) -> Y3[1,0]=1, Y3[0,0]=-1
        [[0, 0], [0, 1]], # K=4: B11
        [[1, 1], [0, 0]], # K=5: B00 + B01
        [[0, 0], [1, 1]]  # K=6: B10 + B11
    ]
    
    # C_coeffs: How M_K contributes to C blocks.
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    C_coeffs = [
        [[1, 1], [0, 0]], # M0 contributes to C00(1), C11(1) -> C00 is (0,0), C11 is (1,1)
        [[0, 0], [1, -1]], # M1 contributes to C10(1), C11(-1)
        [[0, 1], [0, 1]], # M2 contributes to C01(1), C11(1)
        [[1, 0], [1, 0]], # M3 contributes to C00(1), C10(1)
        [[-1, 1], [0, 0]], # M4 contributes to C00(-1), C01(1)
        [[0, 0], [0, 1]], # M5 contributes to C11(1)
        [[1, 0], [0, 0]]  # M6 contributes to C00(1)
    ]
    # Wait, let's re-verify C_coeffs.
    # C00 = M0 + M3 - M4 + M6.
    # M0 coeff in C00 is 1. M0 coeff in C11 is 1.
    # So C_coeffs[0][0,0] = 1, C_coeffs[0][1,1] = 1.
    # C_coeffs[0] = [[1, 0], [0, 1]]. Correct.
    
    # M1: C10 = M1 + M3. C11 = M0 - M1 + M2 + M5.
    # M1 coeff in C10 is 1. M1 coeff in C11 is -1.
    # C_coeffs[1] = [[0, 0], [1, -1]]. Correct.
    
    # M2: C01 = M2 + M4. C11 = ... + M2 ...
    # M2 coeff in C01 is 1. M2 coeff in C11 is 1.
    # C_coeffs[2] = [[0, 1], [0, 1]]. Correct.
    
    # M3: C00 = ... + M3 ... C10 = M1 + M3.
    # M3 coeff in C00 is 1. M3 coeff in C10 is 1.
    # C_coeffs[3] = [[1, 0], [1, 0]]. Correct.
    
    # M4: C00 = ... - M4 ... C01 = M2 + M4.
    # M4 coeff in C00 is -1. M4 coeff in C01 is 1.
    # C_coeffs[4] = [[-1, 1], [0, 0]]. Correct.
    
    # M5: C11 = ... + M5 ...
    # M5 coeff in C11 is 1.
    # C_coeffs[5] = [[0, 0], [0, 1]]. Correct.
    
    # M6: C00 = ... + M6 ...
    # M6 coeff in C00 is 1.
    # C_coeffs[6] = [[1, 0], [0, 0]]. Correct.
    
    # Now we build the 49 terms.
    # U, V, W will be lists of 49 vectors of length 16.
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    col = 0
    for K in range(7):
        for k in range(7):
            # The term corresponds to the k-th rank-1 component of the 2x2 multiplication
            # applied to the block matrices X_K and Y_K.
            
            # U2[:, k] gives the coefficients for the elements of X_K (which is a 2x2 matrix of blocks).
            # X_K is a 2x2 matrix where element (ii, jj) is a 2x2 block.
            # The k-th term of the 2x2 multiplication uses:
            # A-part: sum_{ii,jj} U2[ii*2+jj, k] * X_K[ii, jj]
            # Wait, U2 is for 2x2 matrix multiplication.
            # The input to the 2x2 multiplication is X_K and Y_K.
            # X_K is a 2x2 matrix. Its elements are blocks.
            # Let's flatten X_K into a vector of 4 blocks.
            # X_K_flat = [X_K[0,0], X_K[0,1], X_K[1,0], X_K[1,1]]
            # The k-th term of the 2x2 multiplication is:
            # (sum_{p=0}^3 U2[p, k] * X_K_flat[p]) * (sum_{q=0}^3 V2[q, k] * Y_K_flat[q])
            # This product is a scalar (since blocks are 2x2, but we are decomposing the multiplication of the blocks).
            # Wait, the product of two blocks is a block.
            # The decomposition of the multiplication of two 2x2 matrices (which are blocks) into a result block is given by U2, V2, W2.
            # So the contribution to the result block M_K is:
            # sum_{p,q} U2[p,k] V2[q,k] (X_K_flat[p] * Y_K_flat[q])
            # But X_K_flat[p] is itself a linear combination of A's blocks.
            # So X_K_flat[p] = sum_{ii,jj} X_coeffs[K][ii,jj] * A_block[ii,jj]
            # And A_block[ii,jj] is a 2x2 matrix of elements.
            
            # So the A-part of the term is:
            # sum_{p} U2[p,k] * (sum_{ii,jj} X_coeffs[K][ii,jj] * A_block[ii,jj])
            # = sum_{ii,jj} (U2[p,k] * X_coeffs[K][ii,jj]) * A_block[ii,jj]
            
            # We need to fill U[:, col] with the coefficients for each element of A.
            # Element a of A is in block A_block[ii,jj] at local index (i,j).
            # a = (ii*2 + i)*2 + (jj*2 + j) = ii*4 + i*2 + jj*2 + j.
            
            for p in range(4):
                coeff_p = U2[p, k]
                if coeff_p == 0: continue
                # p corresponds to block index in X_K.
                # p = ii*2 + jj.
                ii = p // 2
                jj = p % 2
                coeff_block = coeff_p * X_coeffs[K][ii, jj]
                if coeff_block == 0: continue
                
                # Fill U for this block
                for i in range(2):
                    for j in range(2):
                        a = (ii*2 + i)*2 + (jj*2 + j)
                        U[a, col] += coeff_block
                        
            for q in range(4):
                coeff_q = V2[q, k]
                if coeff_q == 0: continue
                # q corresponds to block index in Y_K.
                kk = q // 2
                ll = q % 2
                coeff_block = coeff_q * Y_coeffs[K][kk, ll]
                if coeff_block == 0: continue
                
                # Fill V for this block
                for i in range(2):
                    for j in range(2):
                        b = (kk*2 + i)*2 + (ll*2 + j)
                        V[b, col] += coeff_block
                        
            # Now for W.
            # The result of the 2x2 multiplication is M_K.
            # M_K is a 2x2 matrix of blocks.
            # The k-th term contributes to M_K with coefficient W2[c, k] for each element c of M_K.
            # M_K's blocks are then combined into C using C_coeffs[K].
            # M_K_block[cc, dd] = sum_{...} ...
            # C_block[cc, dd] += C_coeffs[K][cc, dd] * M_K_block[cc, dd]
            
            for c in range(4):
                coeff_c = W2[c, k]
                if coeff_c == 0: continue
                # c corresponds to block index in M_K.
                cc = c // 2
                dd = c % 2
                coeff_block = coeff_c * C_coeffs[K][cc, dd]
                if coeff_block == 0: continue
                
                # Fill W for this block
                # C_block[cc, dd] corresponds to elements where row is in {2cc, 2cc+1} and col is in {2dd, 2dd+1}.
                # Wait, C index is c_idx = k*4 + i.
                # The block C_block[cc, dd] contains elements C[2cc+i, 2dd+j].
                # The index in W is c_idx = (2cc+i)*4 + (2dd+j)?
                # No, the prompt says W[k*n + i]. Here n=4.
                # So index = row*4 + col.
                # Row = 2cc + i, Col = 2dd + j.
                for i in range(2):
                    for j in range(2):
                        row = 2*cc + i
                        col_c = 2*dd + j
                        c_idx = row*4 + col_c
                        W[c_idx, col] += coeff_block
                        
            col += 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
