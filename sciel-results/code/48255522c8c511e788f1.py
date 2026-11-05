import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Build the multiplication tensor T as a matrix:
    # T has shape (n*p, m*p) where T[k*n+i, j*p+l] = delta_{i,j} * delta_{k,l}
    # Actually for matmul: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T_{i,j,k} = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Flatten: row index = (i,k) in [0, n*p), col index = (j,l) in [0, m*p)
    # T[(i,k),(j,l)] = 1 if i==j and k==l ... no.
    # T[i,j,k] = delta_{i,j} * delta_{k,l} where l is the second index of B.
    # Let's be precise: A is n×m, B is m×p, C is n×p.
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) with T[i,j,k] = 1 for all valid i,j,k
    # (it's the identity tensor for matrix multiplication).
    # To form a matrix: reshape T to (n*p, m*p) where row=(i,k), col=(j, ?)
    # Actually the standard way: T as a 3-tensor T[i,j,k].
    # Matrix form M has M[row, col] where row = i*p + k (output index),
    # col = j*p + k? No, that doesn't work because j and k are both in B.
    # 
    # Better: The tensor T ∈ R^{n×m × m×p × n×p}... no.
    # T ∈ R^{(n×m) × (m×p) × (n×p)} is wrong dimensionally.
    #
    # Correct: T is a 3-index tensor with T[i,j,k] = 1 (the matmul tensor).
    # As a matrix for rank computation: M[a,b] where a indexes (i,j) in A-space [0, n*m),
    # b indexes (k,l) ... hmm.
    #
    # Standard approach: The matmul tensor T ∈ R^{n×m × m×p × n×p} doesn't make sense.
    # T ∈ R^{(n*m) * (m*p) * (n*p)}? No.
    #
    # T has 3 modes: mode 1 = (i,j) with i∈[0,n), j∈[0,m) → size n*m = 20
    # mode 2 = (j,k) with j∈[0,m), k∈[0,p) → size m*p = 25
    # mode 3 = (i,k) with i∈[0,n), k∈[0,p) → size n*p = 20
    # T[(i,j),(j',k),(i',k')] = 1 if i==i' and j==j' and k==k', else 0.
    #
    # So T is a tensor in R^{20 × 25 × 20}.
    # To compute its rank, we can flatten to a matrix: M[row, col] where
    # row = (i,j) index in [0,20), col = (j',k) index in [0,25), and we fix the third mode?
    # No — tensor rank is the minimum r such that T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r.
    #
    # For rank computation via matrix: fix one mode. E.g., flatten T to a matrix
    # M of size (n*m) × (m*p * n*p) by treating mode 3 as part of the column space.
    # M[(i,j), ((j',k), (i',k'))] = T[(i,j),(j',k),(i',k')]
    # This matrix has rank equal to the tensor rank (this is a standard result:
    # the rank of the matricization equals the tensor rank when the matricization
    # is "generic" enough, but actually the rank of ANY matricization is ≤ tensor rank,
    # and for the matmul tensor, the rank of the (n*m)×(m*p*(n*p)) matricization
    # equals the tensor rank because the tensor is "non-degenerate" in that mode).
    #
    # Actually, more carefully: the tensor rank r(T) satisfies:
    # rank(M_T^{(1|2,3)}) ≤ r(T) ≤ ... 
    # And for the matmul tensor, it's known that the rank of the standard matricization
    # equals the tensor rank. Let me just compute it.
    
    # Build T as a 3D array
    T = np.zeros((n*m, m*p, n*p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i  # W[k*n+i, r] per the trace convention
                T[a_idx, b_idx, c_idx] = 1.0

    # Matricize: M has shape (n*m, m*p * n*p)
    # M[a, b * (n*p) + c] = T[a, b, c]
    M = T.reshape(n*m, m*p * n*p)
    
    # The rank of this matrix gives a lower bound on tensor rank.
    # For the matmul tensor, this should equal the tensor rank.
    # Let's compute the exact rank using fractions.
    
    def exact_rank(M_float):
        """Compute exact rank of a 0/1 matrix using Fraction arithmetic."""
        rows, cols = M_float.shape
        # Convert to Fraction matrix
        M_frac = [[Fraction(int(M_float[i,j]), 1) for j in range(cols)] for i in range(rows)]
        
        # Gaussian elimination
        rank = 0
        pivot_cols = []
        for col in range(cols):
            if rank >= rows:
                break
            # Find pivot
            pivot = None
            for row in range(rank, rows):
                if M_frac[row][col] != 0:
                    pivot = row
                    break
            if pivot is None:
                continue
            # Swap
            M_frac[rank], M_frac[pivot] = M_frac[pivot], M_frac[rank]
            pivot_cols.append(col)
            # Eliminate
            pivot_val = M_frac[rank][col]
            for row in range(rank + 1, rows):
                if M_frac[row][col] != 0:
                    factor = M_frac[row][col] / pivot_val
                    for c in range(col, cols):
                        M_frac[row][c] -= factor * M_frac[rank][c]
            rank += 1
        return rank
    
    # Compute the rank of the full matricization
    # This is expensive for 20 x 500 matrix, but let's try.
    # Actually 20 rows, 500 cols. Gaussian elimination on 20x500 with Fractions.
    # This should be fast enough.
    
    t0 = time.time()
    base_rank = exact_rank(M)
    t1 = time.time()
    
    # The base rank should be 100 (schoolbook) for the full matmul tensor.
    # Now, we want to find a lower-rank decomposition.
    # 
    # Key insight: The tensor rank of the matmul tensor T_{4,5,5} is what we seek.
    # Known results: T(2,2,2) = 7 (Strassen), T(3,3,3) ≤ 23, T(4,4,4) ≤ 49.
    # For T(4,5,5), we can use block structure.
    #
    # Alternative approach: Use the fact that T(4,5,5) can be decomposed by
    # embedding in T(5,5,5) and projecting. T(5,5,5) ≤ 175 (Laser) but better bounds exist.
    # Actually T(5,5,5) is not well-known to be below 125.
    #
    # Let me try a different structural approach:
    # Decompose 4x5x5 as:
    #   - 4x4x4 block (indices i<4, j<4, k<4): use Strassen-based rank 49
    #   - Remaining terms handled with minimal additional rank
    #
    # The current program does this and gets rank 85.
    # 
    # Let me try to improve the 4x4x4 part. The optimal T(4,4,4) is unknown but ≤ 49.
    # If T(4,4,4) = 48, we'd get 84. If 47, we'd get 83.
    #
    # Actually, let me try a completely different approach:
    # Use the "border rank" or "approximate rank" ideas? No, we need exact.
    #
    # Let me try: Can we do better than 85 by a smarter decomposition of the boundary blocks?
    #
    # Boundary blocks in the 4x4x4 + schoolbook approach:
    # - 4x1x4: rank 16
    # - 1x4x4: rank 16  
    # - 1x1x4: rank 4
    # Total boundary: 36
    #
    # Can we do the boundary blocks with less than 36?
    # The 4x1x4 tensor: T[i,j,k] for i∈[0,4), j=4, k∈[0,4). This is a 4x1x4 tensor.
    # Its rank: it's like a "line" tensor. T[i,4,k] = 1 for all i,k.
    # As a tensor in R^{4×1×4}, this has rank 4 (it's a "matrix" of all ones in the i-k plane,
    # but since j is fixed, it's actually rank 4: T = sum_{i=0}^3 sum_{k=0}^3 e_i ⊗ e_4 ⊗ e_k
    # which has rank 16. But can we do better?
    # T[i,4,k] = 1 for all i,k. This is the tensor for multiplying a 4x1 by 1x4 matrix.
    # The rank of T(4,1,4) = 4*1*4 = 16 in schoolbook, but actually:
    # C[i,k] = A[i,4] * B[4,k]. This is an outer product! 
    # C = A_col * B_row where A_col is 4x1 and B_row is 1x4.
    # The tensor T[i,j,k] with j fixed = 4: T[i,4,k] = 1.
    # This can be written as: T = sum_{r=0}^3 u_r ⊗ v ⊗ w_r where...
    # Actually: T[i,4,k] = 1 = (1)*(1) for all i,k.
    # As a tensor: T = (sum_i e_i) ⊗ e_4 ⊗ (sum_k e_k)? No, that gives T[i,4,k] = 1 for all i,k,
    # which is exactly what we want! So rank 1!
    # Wait: (sum_i e_i) ⊗ e_4 ⊗ (sum_k e_k) has components [i,4,k] = 1*1*1 = 1. Yes!
    # So T(4,1,4) has rank 1!
    #
    # Similarly, T(1,4,4) has rank 1.
    # And T(1,1,4) has rank 1.
    #
    # So the boundary can be done with rank 1+1+1 = 3, not 36!
    #
    # But wait — we need to be careful. The full tensor is T[i,j,k] = 1 for all valid i,j,k.
    # When we split into blocks, the blocks share the same tensor entries.
    # Block 1: i<4, j<4, k<4 → T[i,j,k] = 1 (4x4x4 sub-tensor)
    # Block 2: i<4, j=4, k<4 → T[i,4,k] = 1 (4x1x4 sub-tensor)
    # Block 3: i<4, j<4, k=4 → T[i,j,4] = 1 (4x4x1 sub-tensor)
    # Block 4: i<4, j=4, k=4 → T[i,4,4] = 1 (4x1x1 sub-tensor)
    #
    # Each sub-tensor can be decomposed independently, and the total rank is the sum.
    #
    # Block 2: T[i,4,k] = 1 for i∈[0,4), k∈[0,4). Rank = 1 as shown above.
    # Block 3: T[i,j,4] = 1 for i∈[0,4), j∈[0,4). Similarly rank = 1.
    # Block 4: T[i,4,4] = 1 for i∈[0,4). This is a 4x1x1 tensor.
    # T[i,4,4] = 1 = (sum_i e_i) ⊗ e_4 ⊗ e_4. Rank = 1.
    #
    # So total rank = rank(T(4,4,4)) + 1 + 1 + 1 = rank(T(4,4,4)) + 3.
    #
    # With Strassen-based T(4,4,4) ≤ 49, total ≤ 52!
    #
    # Let me verify this is correct. The key claim: T(4,1,4) has tensor rank 1.
    # T ∈ R^{4×5×5} restricted to j=4: T[i,4,k] = 1 for all i,k.
    # Decomposition: T = u ⊗ v ⊗ w where u[i] = 1 for all i∈[0,4), v[4] = 1 (and v[j]=0 for j≠4),
    # w[k] = 1 for all k∈[0,4).
    # Then (u⊗v⊗w)[i,j,k] = u[i]*v[j]*w[k] = 1*1*1 = 1 if j=4, else 0.
    # Yes! This is exactly the sub-tensor. Rank 1.
    #
    # Similarly for the other blocks.
    #
    # So the strategy is:
    # 1. Build a rank-49 decomposition of T(4,4,4) using Strassen.
    # 2. Add rank-1 terms for the three boundary blocks.
    # 3. Total rank = 52.
    #
    # Let me implement this.
    
    if time.time() > deadline - 20:
        # Fallback: schoolbook
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
                    r += 1
        return {'U': U, 'V': V, 'W': W}
    
    # Strassen 2x2x2 decomposition (7 terms)
    # Each term is (u, v, w) where u,v,w are vectors of length 4 (for 2x2 matrices)
    # u[i*2+j], v[j*2+k], w[k*2+i]
    strassen_2x2x2 = [
        # (u, v, w) for each of the 7 Strassen terms
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1 = (A00+A11)(B00+B11)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),    # M2 = (A10+A11)B00
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),   # M3 = A00(B10-B11)
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),   # M4 = A11(B01-B00)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5 = (A00+A01)B11
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6 = (A10-A00)(B00+B01)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7 = (A01-A11)(B10+B11)
    ]
    
    # Verify Strassen 2x2x2
    def verify_strassen_2x2x2():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for (u, v, w) in strassen_2x2x2:
                        s += u[i*2+j] * v[j*2+k] * w[k*2+i]
                    if s != 1:
                        return False
        return True
    
    assert verify_strassen_2x2x2(), "Strassen 2x2x2 verification failed"
    
    # Tensor product of two 2x2x2 decompositions to get 4x4x4
    def tensor_product_decomp(decomp1, decomp2):
        """Tensor product of two 2x2x2 decompositions to get 4x4x4 decomposition.
        decomp1, decomp2: lists of (u, v, w) tuples with vectors of length 4.
        Returns list of (u, v, w) tuples with vectors of length 16.
        """
        result = []
        for (u1, v1, w1) in decomp1:
            for (u2, v2, w2) in decomp2:
                u = [0] * 16
                v = [0] * 16
                w = [0] * 16
                for a1 in range(2):
                    for b1 in range(2):
                        for a2 in range(2):
                            for b2 in range(2):
                                idx = (a1*2+a2)*4 + (b1*2+b2)
                                u[idx] = u1[a1*2+b1] * u2[a2*2+b2]
                                v[idx] = v1[a1*2+b1] * v2[a2*2+b2]
                                w[idx] = w1[a1*2+b1] * w2[a2*2+b2]
                result.append((u, v, w))
        return result
    
    strassen_4x4x4 = tensor_product_decomp(strassen_2x2x2, strassen_2x2x2)
    
    # Verify Strassen 4x4x4
    def verify_4x4x4():
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = 0
                    for (u, v, w) in strassen_4x4x4:
                        s += u[i*4+j] * v[j*4+k] * w[k*4+i]
                    if s != 1:
                        return False
        return True
    
    assert verify_4x4x4(), "Strassen 4x4x4 verification failed"
    
    # Now build the full 4x5x5 decomposition
    # Rank = 49 (for 4x4x4) + 1 (for 4x1x4) + 1 (for 4x4x1) + 1 (for 4x1x1) = 52
    rank = 49 + 1 + 1 + 1
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Add Strassen 4x4x4 terms (handles i<4, j<4, k<4)
    for r_idx, (u, v, w) in enumerate(strassen_4x4x4):
        for i in range(4):
            for j in range(4):
                U[i*5+j][r_idx] = u[i*4+j]
        for j in range(4):
            for k in range(4):
                V[j*5+k][r_idx] = v[j*4+k]
        for k in range(4):
            for i in range(4):
                W[k*5+i][r_idx] = w[k*4+i]
    
    # Add 4x1x4 block (i<4, j=4, k<4): rank 1 term
    # u[i] = 1 for i<4, v[4] = 1, w[k] = 1 for k<4
    r_offset = 49
    for i in range(4):
        U[i*5+4][r_offset] = 1
    V[4*5][r_offset] = 1  # V[j*p+k] with j=4, k=0... wait, V index is j*p+k
    # V has shape (m*p, rank) = (25, rank). V[j*p+k][r].
    # For j=4, k varies: V[4*5+k][r] = V[20+k][r]
    # But we want v[4] = 1 in the 5-dim space, meaning the j=4 component.
    # V index is j*p+k = 4*5+k = 20+k for k in [0,4).
    # The "v" vector in the decomposition is over the (j,k) space of size m*p=25.
    # For the 4x1x4 block, we need v[j,k] = 1 if j=4 and k is arbitrary? No.
    # The tensor entry is T[i,4,k] = 1. In the decomposition:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # For the boundary term: U[i,4,r] * V[j,k,r] * W[i,k,r] should equal 1 when i<4, j=4, k<4.
    # Let U[i,4,r] = 1 for i<4, V[4,k,r] = 1 for k<4 (i.e., V[j=4,k,r]=1), W[i,k,r] = 1 for i<4,k<4.
    # Then for i<4, j=4, k<4: 1*1*1 = 1. Good.
    # For other (i,j,k): at least one factor is 0. Good.
    for k in range(4):
        V[4*5+k][r_offset] = 1
    for i in range(4):
        for k in range(4):
            W[k*5+i][r_offset] = 1  # W[k*n+i] = W[k*4+i]... wait n=4, so W[k*4+i]
    # W has shape (p*n, rank) = (20, rank). W[k*n+i] = W[k*4+i].
    # But p=5, so k ranges [0,5). For k<4: W[k*4+i].
    # Let me fix: W index is k*n+i = k*4+i.
    
    # Redo the W indexing carefully.
    # W has shape (p*n, rank) = (5*4, rank) = (20, rank).
    # W[k*n+i] where k∈[0,5), i∈[0,4).
    # For the 4x1x4 block: we need W[k*4+i] = 1 for k<4, i<4.
    for i in range(4):
        for k in range(4):
            W[k*4+i][r_offset] = 1
    
    r_offset += 1  # r_offset = 50
    
    # Add 4x4x1 block (i<4, j<4, k=4): rank 1 term
    # T[i,j,4] = 1 for i<4, j<4.
    # U[i,j,r] = 1 for i<4, j<4; V[j,4,r] = 1 for j<4; W[4,i,r] = 1 for i<4.
    for i in range(4):
        for j in range(4):
            U[i*5+j][r_offset] = 1
    for j in range(4):
        V[j*5+4][r_offset] = 1  # V[j*p+k] = V[j*5+4]
    for i in range(4):
        W[4*4+i][r_offset] = 1  # W[k*n+i] = W[4*4+i]
    
    r_offset += 1  # r_offset = 51
    
    # Add 4x1x1 block (i<4, j=4, k=4): rank 1 term
    # T[i,4,4] = 1 for i<4.
    # U[i,4,r] = 1 for i<4; V[4,4,r] = 1; W[4,i,r] = 1 for i<4.
    for i in range(4):
        U[i*5+4][r_offset] = 1
    V[4*5+4][r_offset] = 1  # V[4*5+4] = V[24]
    for i in range(4):
        W[4*4+i][r_offset] = 1  # W[4*4+i] = W[16+i]
    
    # Verify the full decomposition
    def verify_full(U_mat, V_mat, W_mat):
        rank_v = len(U_mat[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r_idx in range(rank_v):
                        s += U_mat[i*m+j][r_idx] * V_mat[j*p+k][r_idx] * W_mat[k*n+i][r_idx]
                    if s != 1:
                        return False
        return True
    
    if verify_full(U, V, W):
        return {'U': U, 'V': V, 'W': W}
    
    # If verification failed, fall back to schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][r] = 1
                V_school[j*p+k][r] = 1
                W_school[k*n+i][r] = 1
                r += 1
    return {'U': U_school, 'V': V_school, 'W': W_school}
