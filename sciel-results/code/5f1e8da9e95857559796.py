"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = 3, 5, 7

    def verify_modular(U, V, W, primes=(1000003, 1000033, 999983)):
        """
        Fast exact verification using modular arithmetic with multiple primes.
        For entries bounded by ~100 in magnitude and rank <= 200, the product
        sum is bounded by ~200*100^3 = 2e7, well below any single prime.
        Using 3 primes makes false positives astronomically unlikely.
        """
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        r = U.shape[1]
        
        for mod in primes:
            U_m = U % mod
            V_m = V % mod
            W_m = W % mod
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] mod p
            T = np.einsum('ar,br,cr->abc', U_m.astype(np.int64), V_m.astype(np.int64), W_m.astype(np.int64), optimize=True)
            T = T % mod
            for a in range(n * m):
                i = a // m
                j = a % m
                for b in range(m * p):
                    j2 = b // p
                    k = b % p
                    for c in range(p * n):
                        k2 = c // n
                        i2 = c % n
                        expected = 1 if (i == i2 and j == j2 and k == k2) else 0
                        if T[a, b, c] != expected % mod:
                            return False
        return True

    def verify_fast_float(U, V, W):
        """Fast floating-point verification (may have false positives)."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        r = U.shape[1]
        T = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        for a in range(n * m):
            i = a // m
            j = a % m
            for b in range(m * p):
                j2 = b // p
                k = b % p
                for c in range(p * n):
                    k2 = c // n
                    i2 = c % n
                    expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                    if abs(T[a, b, c] - expected) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p = 105."""
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, r

    def strassen_2x2x2():
        """
        Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
        Returns U, V, W of shapes (4,7), (4,7), (4,7) where:
        U[i*2+j, r], V[j'*2+k, r], W[k'*2+i', r]
        """
        # Strassen's 7 products for 2x2 matrix multiplication
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        r = 7
        U = np.zeros((4, r), dtype=np.int64)
        V = np.zeros((4, r), dtype=np.int64)
        W = np.zeros((4, r), dtype=np.int64)
        
        # Index mapping: a = i*2+j (i=0,1; j=0,1), b = j'*2+k, c = k'*2+i'
        # U[i*2+j, r], V[j'*2+k, r], W[k'*2+i', r]
        
        # M1: (a11+a22)(b11+b22)
        # a11: i=0,j=0 -> idx 0; a22: i=1,j=1 -> idx 3
        # b11: j'=0,k=0 -> idx 0; b22: j'=1,k=1 -> idx 3
        # c11: k'=0,i'=0 -> idx 0; c22: k'=1,i'=1 -> idx 3
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (a21+a22)b11
        # a21: i=1,j=0 -> idx 2; a22: i=1,j=1 -> idx 3
        # b11: j'=0,k=0 -> idx 0
        # c21: k'=1,i'=0 -> idx 2; c22: k'=1,i'=1 -> idx 3
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3: a11(b12-b22)
        # a11: i=0,j=0 -> idx 0
        # b12: j'=0,k=1 -> idx 1; b22: j'=1,k=1 -> idx 3
        # c12: k'=0,i'=1 -> idx 1; c22: k'=1,i'=1 -> idx 3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4: a22(b21-b11)
        # a22: i=1,j=1 -> idx 3
        # b21: j'=1,k=0 -> idx 2; b11: j'=0,k=0 -> idx 0
        # c11: k'=0,i'=0 -> idx 0; c21: k'=1,i'=0 -> idx 2
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5: (a11+a12)b22
        # a11: i=0,j=0 -> idx 0; a12: i=0,j=1 -> idx 1
        # b22: j'=1,k=1 -> idx 3
        # c11: k'=0,i'=0 -> idx 0; c12: k'=0,i'=1 -> idx 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[1, 4] = 1
        
        # M6: (a21-a11)(b11+b12)
        # a21: i=1,j=0 -> idx 2; a11: i=0,j=0 -> idx 0
        # b11: j'=0,k=0 -> idx 0; b12: j'=0,k=1 -> idx 1
        # c21: k'=1,i'=0 -> idx 2; c22: k'=1,i'=1 -> idx 3
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[3, 5] = 1
        
        # M7: (a12-a22)(b21+b22)
        # a12: i=0,j=1 -> idx 1; a22: i=1,j=1 -> idx 3
        # b21: j'=1,k=0 -> idx 2; b22: j'=1,k=1 -> idx 3
        # c11: k'=0,i'=0 -> idx 0; c12: k'=0,i'=1 -> idx 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[1, 6] = 1
        
        return U, V, W, r

    def winograd_1x4x4():
        """
        Winograd's algorithm for 1x4x4 matrix multiplication.
        This is equivalent to computing y[i] = sum_j A[j] * B[j,i] for i=0..3,
        where A is 1x4 and B is 4x4.
        
        Winograd's F(2,2) transform can reduce the rank.
        For 1x4x4, we can use the fact that this is a 1D convolution-like problem.
        
        Actually, 1x4x4 is just: C[k] = sum_{j=0}^{3} A[j] * B[j,k] for k=0..3.
        This is a 4x4 matrix-vector product. The tensor rank is at most 16 (schoolbook).
        But using Winograd-style transforms, we might reduce it.
        
        Let's use the F(2,2) Winograd transform:
        For computing 2 outputs from 4 inputs with 4 filters of length 2:
        Actually, let me think of this as two 1x2x2 problems.
        
        C[0] = A[0]*B[0,0] + A[1]*B[1,0] + A[2]*B[2,0] + A[3]*B[3,0]
        C[1] = A[0]*B[0,1] + A[1]*B[1,1] + A[2]*B[2,1] + A[3]*B[3,1]
        C[2] = A[0]*B[0,2] + A[1]*B[1,2] + A[2]*B[2,2] + A[3]*B[3,2]
        C[3] = A[0]*B[0,3] + A[1]*B[1,3] + A[2]*B[2,3] + A[3]*B[3,3]
        
        Group outputs into pairs: (C[0],C[1]) and (C[2],C[3])
        Group inputs into pairs: (A[0],A[1]) and (A[2],A[3])
        
        For (C[0],C[1]) from (A[0],A[1]):
        C[0] += A[0]*B[0,0] + A[1]*B[1,0]
        C[1] += A[0]*B[0,1] + A[1]*B[1,1]
        This is a 1x2x2 problem. Using Strassen-like or Winograd:
        Actually for 1x2x2, the rank is 4 (schoolbook) and it's hard to reduce.
        
        Let me try a different approach: use the fact that we can apply
        Winograd's transform to the entire 1x4x4 problem.
        
        Winograd's F(4,4) transform for 1D convolution:
        The idea is to transform the input and filter into a domain where
        pointwise multiplication gives the result, then transform back.
        
        For length-4 convolution, we can use a 4-point DFT or a custom transform.
        The rank of the transform-based approach is related to the size of the transform.
        
        Actually, for 1x4x4, the minimal rank is known to be 16 (no reduction possible
        for 1-row matrix multiplication in general, since each output depends on all inputs
        independently).
        
        Let me just return schoolbook for 1x4x4.
        """
        r = 16
        U = np.zeros((4, r), dtype=np.int64)  # 1*4 = 4 rows
        V = np.zeros((4*4, r), dtype=np.int64)  # 4*4 = 16 rows
        W = np.zeros((4*1, r), dtype=np.int64)  # 4*1 = 4 rows
        
        col = 0
        for j in range(4):
            for k in range(4):
                U[j, col] = 1
                V[j*4+k, col] = 1
                W[k, col] = 1
                col += 1
        return U, V, W, r

    def combine_decompositions(decomp_list):
        """
        Combine multiple tensor decompositions into one.
        Each decomp is (U, V, W, r) where U has shape (dim1, r), etc.
        But the dimensions must match the target n,m,p.
        
        For block decomposition: if we split the problem into blocks,
        each block has its own U,V,W and we concatenate them.
        """
        total_rank = sum(d[3] for d in decomp_list)
        U_parts = []
        V_parts = []
        W_parts = []
        for U, V, W, r in decomp_list:
            U_parts.append(U)
            V_parts.append(V)
            W_parts.append(W)
        
        U_full = np.hstack(U_parts)
        V_full = np.hstack(V_parts)
        W_full = np.hstack(W_parts)
        
        return U_full, V_full, W_full, total_rank

    def try_block_strassen():
        """
        Try to decompose 3x5x7 using block structure with Strassen.
        
        Split A (3x5) into [A1 (2x5); A2 (1x5)]
        Split B (5x7) into [B1 (5x5) B2 (5x2)]
        
        C = [A1*B1  A1*B2; A2*B1  A2*B2]
        
        For A1*B1 (2x5x5): split B1 into [B1a (5x3) B1b (5x2)]
        A1*B1a (2x5x3): schoolbook rank 30
        A1*B1b (2x5x2): can we use Strassen? 
          2x5x2: C[i,k] = sum_j A[i,j]*B[j,k] for i=0,1; k=0,1
          This is like computing a 2x2 matrix from 5-dimensional vectors.
          We can write it as: for each j, add A[:,j] * B[j,:] (outer product of 2x1 and 1x2).
          Each such outer product has rank 4 in the tensor sense? No, each (i,j,k) triple
          is a separate term in schoolbook.
          
          Actually, 2x5x2 has tensor rank at most 2*5*2 = 20.
          Can we do better? Let's think of it as:
          C = sum_j A[:,j] B[j,:]
          where A[:,j] is a 2x1 vector and B[j,:] is a 1x2 vector.
          Each term A[:,j] B[j,:] is a 2x2 matrix of rank at most 1.
          The sum of 5 such rank-1 matrices gives C.
          But the tensor rank is about the number of multiplications needed.
          Each multiplication A[i,j]*B[j,k] is one scalar multiplication.
          So 2*5*2 = 20 multiplications in schoolbook.
          
          Using Strassen: we can group the 5 dimensions.
          Write A[:,j] for j=0..4 as a 2x5 matrix, and B[j,:] for j=0..4 as a 5x2 matrix.
          C = A_2x5 * B_5x2.
          This is exactly M_{2,5,2}.
          
          Can we decompose M_{2,5,2} with rank < 20?
          One approach: use the fact that 5 = 2+2+1.
          Split into M_{2,2,2} + M_{2,2,2} + M_{2,1,2}.
          M_{2,2,2} has rank 7 (Strassen).
          M_{2,1,2}: C[i,k] = A[i,1]*B[1,k], which is an outer product.
          The tensor for M_{2,1,2}: T[i, (j',k), (k',i')] where j=1 only.
          T[i, j'*2+k, k'*2+i'] = 1 iff i==i' and j==j' (j=1) and k==k'.
          Since j is fixed at 1, we have:
          T[i, 2+k, 2*k'+i'] = delta_{i,i'} * delta_{k,k'}
          This is a 2x4x4 tensor. Rank: for each (i,k) pair, one term.
          U[i, i*4+k] = 1, V[2+k, i*4+k] = 1, W[2*k'+i', i*4+k] = 1
          Wait, let me re-index.
          a = i*1+1 = i+1 (since m=1, a = i*m+j = i*1+1 = i+1, but i=0,1 so a=1,2)
          Hmm, this is getting confusing. Let me just use schoolbook for 2x1x2.
          Rank = 2*1*2 = 4.
          
          So M_{2,5,2} <= 7 + 7 + 4 = 18. Better than 20!
          
          Similarly, M_{2,5,3}: split into M_{2,2,3} + M_{2,2,3} + M_{2,1,3}.
          M_{2,2,3}: rank at most 2*2*3 = 12. Can we do better?
          Split B into 2+1: M_{2,2,2} + M_{2,2,1}.
          M_{2,2,2} rank 7.
          M_{2,2,1}: C[i,k] = sum_j A[i,j]*B[j,0] for i=0,1. k is only 0.
          This is computing a 2x1 matrix from 2x2 matrices.
          T[i, j'*2+0, 0*2+i'] = 1 iff i==i', j==j', k==0.
          a = i*2+j, b = j'*2+0, c = 0*2+i' = i'
          Rank: for each (i,j) pair, one term. 2*2 = 4 terms.
          So M_{2,2,3} <= 7 + 4 = 11.
          M_{2,1,3}: C[i,k] = A[i,0]*B[0,k] for i=0,1; k=0,1,2.
          This is an outer product of 2x1 and 1x3.
          Rank = 2*3 = 6.
          So M_{2,5,3} <= 11 + 11 + 6 = 28. Better than 30!
          
          Now let's compute the total:
          A1*B1a (2x5x3): 28
          A1*B1b (2x5x2): 18
          A2*B1 (1x5x5): 25 (schoolbook, no reduction for 1-row)
          A1*B2 (2x5x2): 18
          A2*B2 (1x5x2): 10 (schoolbook)
          
          Total: 28 + 18 + 25 + 18 + 10 = 99. Better than 105!
        """
        # This is a complex construction. Let me implement it step by step.
        # For now, let's just try to build it and verify.
        
        # M_{2,2,2} Strassen decomposition
        U_s, V_s, W_s, r_s = strassen_2x2x2()
        
        # M_{2,1,2} schoolbook
        def m212():
            r = 4
            U = np.zeros((2, r), dtype=np.int64)
            V = np.zeros((1*2, r), dtype=np.int64)
            W = np.zeros((2*2, r), dtype=np.int64)
            col = 0
            for i in range(2):
                for k in range(2):
                    U[i, col] = 1
                    V[0*2+k, col] = 1  # j=0, so b = 0*2+k = k
                    W[k*2+i, col] = 1
                    col += 1
            return U, V, W, r
        
        # M_{2,2,1} schoolbook
        def m221():
            r = 4
            U = np.zeros((2*2, r), dtype=np.int64)
            V = np.zeros((2*1, r), dtype=np.int64)
            W = np.zeros((1*2, r), dtype=np.int64)
            col = 0
            for i in range(2):
                for j in range(2):
                    U[i*2+j, col] = 1
                    V[j*1+0, col] = 1  # k=0, so b = j*1+0 = j
                    W[0*2+i, col] = 1
                    col += 1
            return U, V, W, r
        
        # M_{1,5,5} schoolbook
        def m155():
            r = 25
            U = np.zeros((1*5, r), dtype=np.int64)
            V = np.zeros((5*5, r), dtype=np.int64)
            W = np.zeros((5*1, r), dtype=np.int64)
            col = 0
            for j in range(5):
                for k in range(5):
                    U[0*5+j, col] = 1
                    V[j*5+k, col] = 1
                    W[k*1+0, col] = 1
                    col += 1
            return U, V, W, r
        
        # M_{1,5,2} schoolbook
        def m152():
            r = 10
            U = np.zeros((1*5, r), dtype=np.int64)
            V = np.zeros((5*2, r), dtype=np.int64)
            W = np.zeros((2*1, r), dtype=np.int64)
            col = 0
            for j in range(5):
                for k in range(2):
                    U[0*5+j, col] = 1
                    V[j*2+k, col] = 1
                    W[k*1+0, col] = 1
                    col += 1
            return U, V, W, r
        
        # Now build M_{2,5,2} = M_{2,2,2} + M_{2,2,2} + M_{2,1,2}
        # The 5 dimensions of j are split into 2+2+1.
        # We need to map the indices correctly.
        
        def build_m252():
            """Build M_{2,5,2} using Strassen for the 2x2 blocks."""
            # j indices: 0,1 (first 2x2), 2,3 (second 2x2), 4 (1x2)
            # A is 2x5, B is 5x2, C is 2x2
            
            # First M_{2,2,2}: j=0,1
            U1, V1, W1, r1 = strassen_2x2x2()
            # U1 has shape (4, 7) where row = i*2+j for i=0,1; j=0,1
            # V1 has shape (4, 7) where row = j'*2+k for j'=0,1; k=0,1
            # W1 has shape (4, 7) where row = k'*2+i' for k'=0,1; i'=0,1
            
            # Second M_{2,2,2}: j=2,3
            U2, V2, W2, r2 = strassen_2x2x2()
            # Need to shift j indices by 2
            
            # Third M_{2,1,2}: j=4
            U3, V3, W3, r3 = m212()
            # U3 has shape (2, 4) where row = i for i=0,1
            # V3 has shape (2, 4) where row = k for k=0,1 (since j=0 in local, but j=4 globally)
            # W3 has shape (4, 4) where row = k*2+i for k=0,1; i=0,1
            
            # Now we need to embed these into the full 2x5x2 problem.
            # Full U has shape (2*5, r_total) = (10, r_total)
            # Full V has shape (5*2, r_total) = (10, r_total)
            # Full W has shape (2*2, r_total) = (4, r_total)
            
            r_total = r1 + r2 + r3
            U_full = np.zeros((10, r_total), dtype=np.int64)
            V_full = np.zeros((10, r_total), dtype=np.int64)
            W_full = np.zeros((4, r_total), dtype=np.int64)
            
            # First block: j=0,1
            # U1[i*2+j, r] -> U_full[i*5+j, r] for i=0,1; j=0,1
            for i in range(2):
                for j in range(2):
                    for r_idx in range(r1):
                        U_full[i*5+j, r_idx] = U1[i*2+j, r_idx]
            # V1[j'*2+k, r] -> V_full[j'*2+k, r] for j'=0,1; k=0,1
            for j2 in range(2):
                for k in range(2):
                    for r_idx in range(r1):
                        V_full[j2*2+k, r_idx] = V1[j2*2+k, r_idx]
            # W1[k'*2+i', r] -> W_full[k'*2+i', r]
            for k2 in range(2):
                for i2 in range(2):
                    for r_idx in range(r1):
                        W_full[k2*2+i2, r_idx] = W1[k2*2+i2, r_idx]
            
            # Second block: j=2,3
            offset = r1
            for i in range(2):
                for j in range(2):
                    for r_idx in range(r2):
                        U_full[i*5+(j+2), offset+r_idx] = U2[i*2+j, r_idx]
            for j2 in range(2):
                for k in range(2):
                    for r_idx in range(r2):
                        V_full[(j2+2)*2+k, offset+r_idx] = V2[j2*2+k, r_idx]
            for k2 in range(2):
                for i2 in range(2):
                    for r_idx in range(r2):
                        W_full[k2*2+i2, offset+r_idx] = W2[k2*2+i2, r_idx]
            
            # Third block: j=4
            offset2 = r1 + r2
            for i in range(2):
                for r_idx in range(r3):
                    U_full[i*5+4, offset2+r_idx] = U3[i, r_idx]
            for k in range(2):
                for r_idx in range(r3):
                    V_full[4*2+k, offset2+r_idx] = V3[k, r_idx]
            for k2 in range(2):
                for i2 in range(2):
                    for r_idx in range(r3):
                        W_full[k2*2+i2, offset2+r_idx] = W3[k2*2+i2, r_idx]
            
            return U_full, V_full, W_full, r_total
        
        # Build M_{2,5,3} = M_{2,2,3} + M_{2,2,3} + M_{2,1,3}
        # M_{2,2,3} = M_{2,2,2} + M_{2,2,1}
        
        def m213():
            r = 6
            U = np.zeros((2, r), dtype=np.int64)
            V = np.zeros((1*3, r), dtype=np.int64)
            W = np.zeros((3*2, r), dtype=np.int64)
            col = 0
            for i in range(2):
                for k in range(3):
                    U[i, col] = 1
                    V[0*3+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
            return U, V, W, r
        
        def build_m223():
            """Build M_{2,2,3} = M_{2,2,2} + M_{2,2,1}"""
            # k indices: 0,1 (first 2x2x2), 2 (2x2x1)
            U1, V1, W1, r1 = strassen_2x2x2()
            U2, V2, W2, r2 = m221()
            
            r_total = r1 + r2
            U_full = np.zeros((2*2, r_total), dtype=np.int64)
            V_full = np.zeros((2*3, r_total), dtype=np.int64)
            W_full = np.zeros((3*2, r_total), dtype=np.int64)
            
            # First block: k=0,1
            for i in range(2):
                for j in range(2):
                    for r_idx in range(r1):
                        U_full[i*2+j, r_idx] = U1[i*2+j, r_idx]
            for j2 in range(2):
                for k in range(2):
                    for r_idx in range(r1):
                        V_full[j2*3+k, r_idx] = V1[j2*2+k, r_idx]
            for k2 in range(2):
                for i2 in range(2):
                    for r_idx in range(r1):
                        W_full[k2*2+i2, r_idx] = W1[k2*2+i2, r_idx]
            
            # Second block: k=2
            offset = r1
            for i in range(2):
                for j in range(2):
                    for r_idx in range(r2):
                        U_full[i*2+j, offset+r_idx] = U2[i*2+j, r_idx]
            for j2 in range(2):
                for r_idx in range(r2):
                    V_full[j2*3+2, offset+r_idx] = V2[j2*1+0, r_idx]
            for i2 in range(2):
                for r_idx in range(r2):
                    W_full[2*2+i2, offset+r_idx] = W2[0*2+i2, r_idx]
            
            return U_full, V_full, W_full, r_total
        
        def build_m253():
            """Build M_{2,5,3} = M_{2,2,3} + M_{2,2,3} + M_{2,1,3}"""
            U1, V1, W1, r1 = build_m223()
            U2, V2, W2, r2 = build_m223()
            U3, V3, W3, r3 = m213()
            
            r_total = r1 + r2 + r3
            U_full = np.zeros((2*5, r_total), dtype=np.int64)
            V_full = np.zeros((5*3, r_total), dtype=np.int64)
            W_full = np.zeros((3*2, r_total), dtype=np.int64)
            
            # First block: j=0,1
            for i in range(2):
                for j in range(2):
                    for r_idx in range(r1):
                        U_full[i*5+j, r_idx] = U1[i*2+j, r_idx]
            for j2 in range(2):
                for k in range(3):
                    for r_idx in range(r1):
                        V_full[j2*3+k, r_idx] = V1[j2*3+k, r_idx]
            for k2 in range(3):
                for i2 in range(2):
                    for r_idx in range(r1):
                        W_full[k2*2+i2, r_idx] = W1[k2*2+i2, r_idx]
            
            # Second block: j=2,3
            offset = r1
            for i in range(2):
                for j in range(2):
                    for r_idx in range(r2):
                        U_full[i*5+(j+2), offset+r_idx] = U2[i*2+j, r_idx]
            for j2 in range(2):
                for k in range(3):
                    for r_idx in range(r2):
                        V_full[(j2+2)*3+k, offset+r_idx] = V2[j2*3+k, r_idx]
            for k2 in range(3):
                for i2 in range(2):
                    for r_idx in range(r2):
                        W_full[k2*2+i2, offset+r_idx] = W2[k2*2+i2, r_idx]
            
            # Third block: j=4
            offset2 = r1 + r2
            for i in range(2):
                for r_idx in range(r3):
                    U_full[i*5+4, offset2+r_idx] = U3[i, r_idx]
            for k in range(3):
                for r_idx in range(r3):
                    V_full[4*3+k, offset2+r_idx] = V3[0*3+k, r_idx]
            for k2 in range(3):
                for i2 in range(2):
                    for r_idx in range(r3):
                        W_full[k2*2+i2, offset2+r_idx] = W3[k2*2+i2, r_idx]
            
            return U_full, V_full, W_full, r_total
        
        # Now build the full 3x5x7 decomposition
        # A (3x5) = [A1 (2x5); A2 (1x5)]
        # B (5x7) = [B1 (5x5) B2 (5x2)]
        # C (3x7) = [A1*B1 (2x5)  A1*B2 (2x2); A2*B1 (1x5)  A2*B2 (1x2)]
        
        # A1*B1 (2x5x5): split B1 into [B1a (5x3) B1b (5x2)]
        # A1*B1a (2x5x3): rank 28
        # A1*B1b (2x5x2): rank 18
        # A2*B1 (1x5x5): rank 25
        # A1*B2 (2x5x2): rank 18
        # A2*B2 (1x5x2): rank 10
        
        U_a1b1a, V_a1b1a, W_a1b1a, r_a1b1a = build_m253()
        U_a1b1b, V_a1b1b, W_a1b1b, r_a1b1b = build_m252()
        U_a2b1, V_a2b1, W_a2b1, r_a2b1 = m155()
        U_a1b2, V_a1b2, W_a1b2, r_a1b2 = build_m252()
        U_a2b2, V_a2b2, W_a2b2, r_a2b2 = m152()
        
        # Now we need to assemble the full 3x5x7 decomposition.
        # The full U has shape (3*5, r_total) = (15, r_total)
        # The full V has shape (5*7, r_total) = (35, r_total)
        # The full W has shape (7*3, r_total) = (21, r_total)
        
        r_total = r_a1b1a + r_a1b1b + r_a2b1 + r_a1b2 + r_a2b2
        U_full = np.zeros((15, r_total), dtype=np.int64)
        V_full = np.zeros((35, r_total), dtype=np.int64)
        W_full = np.zeros((21, r_total), dtype=np.int64)
        
        # A1*B1a: i=0,1; j=0..4; k=0,1,2
        # U_a1b1a has shape (10, r_a1b1a) where row = i*5+j
        # V_a1b1a has shape (15, r_a1b1a) where row = j*3+k
        # W_a1b1a has shape (6, r_a1b1a) where row = k*2+i
        
        offset = 0
        for i in range(2):
            for j in range(5):
                for r_idx in range(r_a1b1a):
                    U_full[i*5+j, offset+r_idx] = U_a1b1a[i*5+j, r_idx]
        for j in range(5):
            for k in range(3):
                for r_idx in range(r_a1b1a):
                    V_full[j*7+k, offset+r_idx] = V_a1b1a[j*3+k, r_idx]
        for k in range(3):
            for i in range(2):
                for r_idx in range(r_a1b1a):
                    W_full[k*3+i, offset+r_idx] = W_a1b1a[k*2+i, r_idx]
        
        # A1*B1b: i=0,1; j=0..4; k=3,4
        offset += r_a1b1a
        for i in range(2):
            for j in range(5):
                for r_idx in range(r_a1b1b):
                    U_full[i*5+j, offset+r_idx] = U_a1b1b[i*5+j, r_idx]
        for j in range(5):
            for k in range(2):
                for r_idx in range(r_a1b1b):
                    V_full[j*7+(k+3), offset+r_idx] = V_a1b1b[j*2+k, r_idx]
        for k in range(2):
            for i in range(2):
                for r_idx in range(r_a1b1b):
                    W_full[(k+3)*3+i, offset+r_idx] = W_a1b1b[k*2+i, r_idx]
        
        # A2*B1: i=2; j=0..4; k=0..4
        offset += r_a1b1b
        for j in range(5):
            for r_idx in range(r_a2b1):
                U_full[2*5+j, offset+r_idx] = U_a2b1[0*5+j, r_idx]
        for j in range(5):
            for k in range(5):
                for r_idx in range(r_a2b1):
                    V_full[j*7+k, offset+r_idx] = V_a2b1[j*5+k, r_idx]
        for k in range(5):
            for r_idx in range(r_a2b1):
                W_full[k*3+2, offset+r_idx] = W_a2b1[k*1+0, r_idx]
        
        # A1*B2: i=0,1; j=0..4; k=5,6
        offset += r_a2b1
        for i in range(2):
            for j in range(5):
                for r_idx in range(r_a1b2):
                    U_full[i*5+j, offset+r_idx] = U_a1b2[i*5+j, r_idx]
        for j in range(5):
            for k in range(2):
                for r_idx in range(r_a1b2):
                    V_full[j*7+(k+5), offset+r_idx] = V_a1b2[j*2+k, r_idx]
        for k in range(2):
            for i in range(2):
                for r_idx in range(r_a1b2):
                    W_full[(k+5)*3+i, offset+r_idx] = W_a1b2[k*2+i, r_idx]
        
        # A2*B2: i=2; j=0..4; k=5,6
        offset += r_a1b2
        for j in range(5):
            for r_idx in range(r_a2b2):
                U_full[2*5+j, offset+r_idx] = U_a2b2[0*5+j, r_idx]
        for j in range(5):
            for k in range(2):
                for r_idx in range(r_a2b2):
                    V_full[j*7+(k+5), offset+r_idx] = V_a2b2[j*2+k, r_idx]
        for k in range(2):
            for r_idx in range(r_a2b2):
                W_full[(k+5)*3+2, offset+r_idx] = W_a2b2[k*1+0, r_idx]
        
        return U_full, V_full, W_full, r_total

    # Start with schoolbook
    U_sb, V_sb, W_sb, base_rank = schoolbook()
    current_best_rank = base_rank
    current_best = (U_sb, V_sb, W_sb)
    
    # Try block Strassen decomposition
    try:
        if time.time() < deadline - 30:
            U_bs, V_bs, W_bs, bs_rank = try_block_strassen()
            if bs_rank < current_best_rank:
                # Verify the block Strassen decomposition
                if verify_fast_float(U_bs, V_bs, W_bs):
                    # Double-check with exact verification
                    if verify_exact(U_bs.tolist(), V_bs.tolist(), W_bs.tolist()):
                        current_best_rank = bs_rank
                        current_best = (U_bs, V_bs, W_bs)
    except Exception as e:
        pass
    
    U, V, W = current_best
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
