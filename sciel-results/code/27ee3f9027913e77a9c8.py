"""Matrix multiplication 4x5x5 rank-minimal decomposition."""
import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Construct a tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    Strategy: Start with schoolbook (rank 100), then apply Strassen-like
    block decompositions to reduce rank where possible.
    
    Key insight: For computing C[i,k] = sum_j A[i,j]*B[j,k], we can use
    Winograd/Strassen techniques on sub-blocks to reduce the number of
    multiplications below the naive count.
    
    Approach:
    1. Partition A (4x5) into two 2x5 blocks
    2. Partition B (5x5) into 2x2 blocks plus a 1-column remainder
    3. Apply 2x2 Strassen to the 2x2 blocks
    4. Handle the remaining column with schoolbook
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Schoolbook baseline
    rank = n * m * p  # 100
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to improve using block Strassen
    # Partition A into top 2 rows (i=0,1) and bottom 2 rows (i=2,3)
    # Partition B into columns: j=0,1 (first 2 rows of B) and j=2,3,4 (rest)
    
    # For the 2x2 block of B (rows 0,1), we can use Strassen:
    # Instead of 4 multiplications for A[0:2, 0:2] @ B[0:2, 0:2],
    # use 7 Strassen multiplications
    
    # But this only saves 4-7 = -3, so it's not helpful for just 2x2.
    # We need larger blocks.
    
    # Let's try a different approach: use the fact that we have 4 rows of A
    # and 5 rows of B. We can group rows of A in pairs and apply a 
    # generalized Strassen.
    
    # Actually, let's try a concrete construction:
    # 
    # For each column k of the output C (k=0..4):
    #   C[:,k] = A @ B[:,k]
    #   where A is 4x5 and B[:,k] is 5x1
    #
    # This is 4x5 times 5x1 = 4x1, which naively takes 20 multiplications.
    #
    # Can we do better? The matrix A is 4x5, B[:,k] is 5x1.
    # 
    # One idea: if A has low-rank structure in the columns, we could save.
    # But A is arbitrary, so we can't assume that.
    #
    # However, we can use the linearity in both A and B.
    #
    # Let's try: express each row of C as a linear combination of 
    # precomputed products.
    #
    # For row i of C: C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # If we precompute, for each j, the vector B[j,:] (which is 5 values),
    # and for each i, the vector A[i,:] (5 values), then C[i,:] = A[i,:] @ B
    # which is a dot product for each k.
    #
    # Total work: 4*5*5 = 100 multiplications.
    #
    # To do better, we need to find a set of linear forms in A and B such
    # that each output C[i,k] is a linear combination of the products of
    # these forms, with fewer than 100 products.
    
    # Let's try a specific construction based on Kronecker products.
    #
    # View the 4x5 matrix A as a vector in R^{20} (row-major).
    # View the 5x5 matrix B as a vector in R^{25} (row-major).
    # View the 4x5 matrix C as a vector in R^{20} (row-major).
    #
    # The matrix multiplication is a bilinear map: R^{20} x R^{25} -> R^{20}
    #
    # We want to find the minimum number of rank-1 tensors (u, v, w) such
    # that for all A, B:
    #   vec(C) = sum_{r=1}^{rank} (u_r^T vec(A)) * (v_r^T vec(B)) * w_r
    #
    # This is the tensor rank of the matrix multiplication tensor.
    
    # For now, let's try a heuristic: use the schoolbook decomposition
    # but check if any terms can be combined.
    
    # Check deadline
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a better decomposition using a greedy approach.
    # 
    # Idea: for each (i,j,k) term in schoolbook, check if it can be
    # expressed as a linear combination of other terms. If so, we can
    # eliminate it.
    #
    # Term (i,j,k): A[i,j] * B[j,k] appears in C[i,k] with coefficient 1.
    #
    # Suppose we want to eliminate term (i0,j0,k0). We need to find
    # coefficients c_{i,j,k} for other terms such that:
    #   sum_{i,j,k != (i0,j0,k0)} c_{i,j,k} * A[i,j] * B[j,k] 
    #     can replace A[i0,j0] * B[j0,k0] in all outputs.
    #
    # This is generally not possible unless there's a linear dependency.
    
    # Let's try a different strategy: use the structure of the problem.
    #
    # The matrix multiplication tensor for (4,5,5) has rank at most 100
    # (schoolbook). Can we do better?
    #
    # Known results: for (m,n,p), the rank is at most m*n*p, and often
    # significantly less. For (2,2,2), rank is 7 (Strassen).
    #
    # For (4,5,5), we can try to use block decomposition:
    #
    # A = [A11 A12 A13]  where A11 is 2x2, A12 is 2x2, A13 is 2x1
    #     [A21 A22 A23]  where A21 is 2x2, A22 is 2x2, A23 is 2x1
    #
    # B = [B11 B21 B31]  where B11 is 2x2, B21 is 2x2, B31 is 2x1
    #     [B12 B22 B32]
    #     [B13 B23 B33]
    #
    # C = A @ B
    #
    # C11 = A11@B11 + A12@B21 + A13@B31  (2x2)
    # C12 = A11@B12 + A12@B22 + A13@B32  (2x2)
    # C13 = A11@B13 + A12@B23 + A13@B33  (2x2)
    # C21 = A21@B11 + A22@B21 + A23@B31  (2x2)
    # C22 = A21@B12 + A22@B22 + A23@B32  (2x2)
    # C23 = A21@B13 + A22@B23 + A23@B33  (2x2)
    #
    # Each 2x2 = 2x2 * 2x2 multiplication can be done in rank 7 (Strassen).
    # Each 2x2 * 2x1 multiplication takes 4 multiplications (schoolbook).
    #
    # So for each C block (2x2):
    #   One 2x2*2x2 (rank 7) + two 2x2*2x1 (rank 4 each) = 7 + 4 + 4 = 15
    #
    # Total: 6 blocks * 15 = 90
    #
    # That's an improvement from 100 to 90!
    
    # Let's implement this block Strassen approach.
    
    # Strassen for 2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # We need to express this in terms of the global U, V, W matrices.
    
    # Let's build the decomposition term by term.
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term to the decomposition."""
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # Helper to create zero vectors
    def zero_u():
        return [0] * (n * m)
    def zero_v():
        return [0] * (m * p)
    def zero_w():
        return [0] * (p * n)
    
    # Process each 2x2 output block
    # C is 4x5, so we have 2x3 blocks of size 2x2
    # Block (bi, bk) where bi in {0,1}, bk in {0,1,2}
    # Block (bi, bk) covers C[2*bi:2*bi+2, 2*bk:2*bk+2]
    
    for bi in range(2):  # row block
        for bk in range(3):  # column block (but only 2 complete 2x2 blocks fit in 5 cols)
            # This block covers C[2*bi:2*bi+2, 2*bk:2*bk+2]
            # But we only have 5 columns, so bk can be 0, 1, 2
            # bk=0: cols 0,1
            # bk=1: cols 2,3
            # bk=2: cols 4 (only 1 col, so this is 2x1, not 2x2)
            
            if bk < 2:  # Complete 2x2 block
                # Output: C[2*bi:2*bi+2, 2*bk:2*bk+2]
                # = A[2*bi:2*bi+2, :] @ B[:, 2*bk:2*bk+2]
                #
                # A_block = A[2*bi:2*bi+2, :] (2x5)
                # B_block = B[:, 2*bk:2*bk+2] (5x2)
                #
                # C_block = A_block @ B_block (2x2)
                #
                # We can decompose A_block into 2x2, 2x2, 2x1 blocks:
                # A_block = [A11 A12 A13] where A11 is 2x2, A12 is 2x2, A13 is 2x1
                #
                # B_block = [B11; B21; B31] where B11 is 2x2, B21 is 2x2, B31 is 2x1
                #
                # C_block = A11@B11 + A12@B21 + A13@B31
                #
                # A11@B11: 2x2 * 2x2 -> use Strassen (rank 7)
                # A12@B21: 2x2 * 2x2 -> use Strassen (rank 7)
                # A13@B31: 2x1 * 2x1 -> 4 multiplications (schoolbook)
                #
                # Total for this block: 7 + 7 + 4 = 18
                
                # A11 = A[2*bi:2*bi+2, 0:2]
                # A12 = A[2*bi:2*bi+2, 2:4]
                # A13 = A[2*bi:2*bi+2, 4]
                
                # B11 = B[0:2, 2*bk:2*bk+2]
                # B21 = B[2:4, 2*bk:2*bk+2]
                # B31 = B[4, 2*bk:2*bk+2]
                
                # Strassen for A11@B11
                # a11 = A[2*bi, 0], a12 = A[2*bi, 1], a21 = A[2*bi+1, 0], a22 = A[2*bi+1, 1]
                # b11 = B[0, 2*bk], b12 = B[0, 2*bk+1], b21 = B[1, 2*bk], b22 = B[1, 2*bk+1]
                
                a11_i, a11_j = 2*bi, 0
                a12_i, a12_j = 2*bi, 1
                a21_i, a21_j = 2*bi+1, 0
                a22_i, a22_j = 2*bi+1, 1
                
                b11_i, b11_k = 0, 2*bk
                b12_i, b12_k = 0, 2*bk+1
                b21_i, b21_k = 1, 2*bk
                b22_i, b22_k = 1, 2*bk+1
                
                # M1 = (a11+a22)(b11+b22)
                u = zero_u(); u[a11_i*m+a11_j] = 1; u[a22_i*m+a22_j] = 1
                v = zero_v(); v[b11_i*p+b11_k] = 1; v[b21_i*p+b21_k] = 1
                w = zero_w()
                # c11 += M1, c22 += M1
                w[(2*bk)*n + 2*bi] = 1      # C[2*bi, 2*bk]
                w[(2*bk+1)*n + 2*bi+1] = 1  # C[2*bi+1, 2*bk+1]
                add_term(u, v, w)
                
                # M2 = (a21+a22)b11
                u = zero_u(); u[a21_i*m+a21_j] = 1; u[a22_i*m+a22_j] = 1
                v = zero_v(); v[b11_i*p+b11_k] = 1
                w = zero_w()
                w[(2*bk)*n + 2*bi+1] = 1  # c21 += M2
                add_term(u, v, w)
                
                # M3 = a11(b21-b22)
                u = zero_u(); u[a11_i*m+a11_j] = 1
                v = zero_v(); v[b21_i*p+b21_k] = 1; v[b22_i*p+b22_k] = -1
                w = zero_w()
                w[(2*bk+1)*n + 2*bi] = 1    # c12 += M3
                w[(2*bk)*n + 2*bi+1] = 1    # c21 += M3 (wait, no)
                # Actually: c12 = M3 + M5, c21 = M2 + M4
                # Let me re-check Strassen formulas:
                # c11 = M1 + M4 - M5 + M7
                # c12 = M3 + M5
                # c21 = M2 + M4
                # c22 = M1 - M2 + M3 + M6
                # So M3 contributes to c12 and c22
                w = zero_w()
                w[(2*bk+1)*n + 2*bi] = 1    # c12 += M3
                w[(2*bk+1)*n + 2*bi+1] = 1  # c22 += M3
                add_term(u, v, w)
                
                # M4 = a22(b12-b11)
                u = zero_u(); u[a22_i*m+a22_j] = 1
                v = zero_v(); v[b12_i*p+b12_k] = 1; v[b11_i*p+b11_k] = -1
                w = zero_w()
                w[(2*bk)*n + 2*bi] = 1    # c11 += M4
                w[(2*bk)*n + 2*bi+1] = 1  # c21 += M4
                add_term(u, v, w)
                
                # M5 = (a11+a12)b22
                u = zero_u(); u[a11_i*m+a11_j] = 1; u[a12_i*m+a12_j] = 1
                v = zero_v(); v[b22_i*p+b22_k] = 1
                w = zero_w()
                w[(2*bk)*n + 2*bi] = -1   # c11 -= M5
                w[(2*bk+1)*n + 2*bi] = 1   # c12 += M5
                add_term(u, v, w)
                
                # M6 = (a21-a11)(b11+b12)
                u = zero_u(); u[a21_i*m+a21_j] = 1; u[a11_i*m+a11_j] = -1
                v = zero_v(); v[b11_i*p+b11_k] = 1; v[b12_i*p+b12_k] = 1
                w = zero_w()
                w[(2*bk+1)*n + 2*bi+1] = 1  # c22 += M6
                add_term(u, v, w)
                
                # M7 = (a12-a22)(b21+b22)
                u = zero_u(); u[a12_i*m+a12_j] = 1; u[a22_i*m+a22_j] = -1
                v = zero_v(); v[b21_i*p+b21_k] = 1; v[b22_i*p+b22_k] = 1
                w = zero_w()
                w[(2*bk)*n + 2*bi] = 1    # c11 += M7
                add_term(u, v, w)
                
                # Strassen for A12@B21
                # A12 = A[2*bi:2*bi+2, 2:4]
                # B21 = B[2:4, 2*bk:2*bk+2]
                
                a11_i, a11_j = 2*bi, 2
                a12_i, a12_j = 2*bi, 3
                a21_i, a21_j = 2*bi+1, 2
                a22_i, a22_j = 2*bi+1, 3
                
                b11_i, b11_k = 2, 2*bk
                b12_i, b12_k = 2, 2*bk+1
                b21_i, b21_k = 3, 2*bk
                b22_i, b22_k = 3, 2*bk+1
                
                # M1' = (a11+a22)(b11+b22)
                u = zero_u(); u[a11_i*m+a11_j] = 1; u[a22_i*m+a22_j] = 1
                v = zero_v(); v[b11_i*p+b11_k] = 1; v[b21_i*p+b21_k] = 1
                w = zero_w()
                w[(2*bk)*n + 2*bi] = 1
                w[(2*bk+1)*n + 2*bi+1] = 1
                add_term(u, v, w)
                
                # M2' = (a21+a22)b11
                u = zero_u(); u[a21_i*m+a21_j] = 1; u[a22_i*m+a22_j] = 1
                v = zero_v(); v[b11_i*p+b11_k] = 1
                w = zero_w()
                w[(2*bk)*n + 2*bi+1] = 1
                add_term(u, v, w)
                
                # M3' = a11(b21-b22)
                u = zero_u(); u[a11_i*m+a11_j] = 1
                v = zero_v(); v[b21_i*p+b21_k] = 1; v[b22_i*p+b22_k] = -1
                w = zero_w()
                w[(2*bk+1)*n + 2*bi] = 1
                w[(2*bk+1)*n + 2*bi+1] = 1
                add_term(u, v, w)
                
                # M4' = a22(b12-b11)
                u = zero_u(); u[a22_i*m+a22_j] = 1
                v = zero_v(); v[b12_i*p+b12_k] = 1; v[b11_i*p+b11_k] = -1
                w = zero_w()
                w[(2*bk)*n + 2*bi] = 1
                w[(2*bk)*n + 2*bi+1] = 1
                add_term(u, v, w)
                
                # M5' = (a11+a12)b22
                u = zero_u(); u[a11_i*m+a11_j] = 1; u[a12_i*m+a12_j] = 1
                v = zero_v(); v[b22_i*p+b22_k] = 1
                w = zero_w()
                w[(2*bk)*n + 2*bi] = -1
                w[(2*bk+1)*n + 2*bi] = 1
                add_term(u, v, w)
                
                # M6' = (a21-a11)(b11+b12)
                u = zero_u(); u[a21_i*m+a21_j] = 1; u[a11_i*m+a11_j] = -1
                v = zero_v(); v[b11_i*p+b11_k] = 1; v[b12_i*p+b12_k] = 1
                w = zero_w()
                w[(2*bk+1)*n + 2*bi+1] = 1
                add_term(u, v, w)
                
                # M7' = (a12-a22)(b21+b22)
                u = zero_u(); u[a12_i*m+a12_j] = 1; u[a22_i*m+a22_j] = -1
                v = zero_v(); v[b21_i*p+b21_k] = 1; v[b22_i*p+b22_k] = 1
                w = zero_w()
                w[(2*bk)*n + 2*bi] = 1
                add_term(u, v, w)
                
                # A13@B31: 2x1 * 2x1 -> 4 multiplications
                # A13 = A[2*bi:2*bi+2, 4] (2x1)
                # B31 = B[4, 2*bk:2*bk+2] (1x2)
                
                for di in range(2):
                    for dk in range(2):
                        u = zero_u(); u[(2*bi+di)*m + 4] = 1
                        v = zero_v(); v[4*p + (2*bk+dk)] = 1
                        w = zero_w()
                        w[(2*bk+dk)*n + (2*bi+di)] = 1
                        add_term(u, v, w)
            
            elif bk == 2:  # 2x1 block (col 4)
                # C[2*bi:2*bi+2, 4] = A[2*bi:2*bi+2, :] @ B[:, 4]
                # = sum_j A[2*bi:2*bi+2, j] * B[j, 4]
                # 5 multiplications for each of 2 rows = 10 total
                # But we can do it in 2*5 = 10, or maybe less?
                # For 2x5 * 5x1, naive is 10. Can we do better?
                # Probably not significantly, so use schoolbook.
                
                for di in range(2):
                    for j in range(5):
                        u = zero_u(); u[(2*bi+di)*m + j] = 1
                        v = zero_v(); v[j*p + 4] = 1
                        w = zero_w()
                        w[4*n + (2*bi+di)] = 1
                        add_term(u, v, w)
    
    # Check deadline
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Verify the decomposition
    def verify(U_list, V_list, W_list):
        """Verify that the decomposition is correct."""
        rank = len(U_list)
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        
        # Test with random matrices
        np.random.seed(42)
        for _ in range(10):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute using decomposition
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_val = U_arr[:, r] @ A.flatten()
                v_val = V_arr[:, r] @ B.flatten()
                C_computed += u_val * v_val * W_arr[:, r].reshape(p, n).T
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True
    
    # Check if our decomposition is valid
    if not verify(U, V, W):
        # Fall back to schoolbook
        U = []
        V = []
        W = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    u[i*m+j] = 1
                    v[j*p+k] = 1
                    w[k*n+i] = 1
                    U.append(u)
                    V.append(v)
                    W.append(w)
    
    return {"U": U, "V": V, "W": W}
