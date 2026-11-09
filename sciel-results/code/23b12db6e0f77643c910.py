import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = 3, 4, 8
    
    def naive_decomp(n_, m_, p_):
        """Returns U, V, W for naive decomposition of n_ x m_ * m_ x p_"""
        r = n_ * m_ * p_
        U = np.zeros((n_ * m_, r), dtype=Fraction)
        V = np.zeros((m_ * p_, r), dtype=Fraction)
        W = np.zeros((p_ * n_, r), dtype=Fraction)
        col = 0
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    U[i * m_ + j, col] = 1
                    V[j * p_ + k, col] = 1
                    W[k * n_ + i, col] = 1
                    col += 1
        return U, V, W
    
    def combine_decomps(list_of_decomps):
        """Combine a list of (U, V, W) decompositions by concatenating columns"""
        U_list = [d[0] for d in list_of_decomps]
        V_list = [d[1] for d in list_of_decomps]
        W_list = [d[2] for d in list_of_decomps]
        U = np.hstack(U_list)
        V = np.hstack(V_list)
        W = np.hstack(W_list)
        return U, V, W
    
    def verify_decomp(U, V, W, n_, m_, p_, tol=1e-6):
        """Verify that U, V, W correctly decompose n_ x m_ * m_ x p_ matrix multiplication"""
        # The multiplication tensor is: T[i, j, k] = delta_{j,k} * delta_{i,r} where C[i,r] = sum_j A[i,j] B[j,r]
        # In our convention: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # The product sum_s U[i*m+j, s] * V[j*p+k, s] should equal W[k*n+i, s] summed over s
        # Actually, the decomposition says: (A*B)[i,k] = sum_j A[i,j] B[j,k] = sum_s (sum_j U[i*m+j,s] A[i,j]) * (sum_j V[j*p+k,s] B[j,k]) ... no
        # Let me think again. The tensor for matrix mult n x m * m x p -> n x p is:
        # T(i, j, k) where i in [0,n), j in [0,m), k in [0,p)
        # The bilinear form is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor decomposition: C[i,k] = sum_s U[i,j]_s * V[j,k]_s ... 
        # In our format: U has shape (n*m, rank) indexed by (i,j), V has shape (m*p, rank) indexed by (j,k), W has shape (p*n, rank) indexed by (k,i)
        # The decomposition: sum_s U[(i,j),s] * V[(j,k),s] should give the coefficient for (i,k) in the output
        # But W is (p*n, rank) indexed by (k,i)
        # So we need: sum_s U[i*m+j, s] * V[j*p+k, s] = W[k*n+i, s] ... no, that's not right either
        # 
        # The standard tensor decomposition for matrix mult: 
        # The multiplication map is a bilinear map (A,B) -> C
        # C[i,k] = sum_j A[i,j] B[j,k]
        # We write this as C[i,k] = sum_s U_s[i,j] A[i,j] * V_s[j,k] B[j,k] ... no
        # 
        # Actually: C[i,k] = sum_s (sum_j U[i*m+j,s] * A[i,j]) * (sum_j V[j*p+k,s] * B[j,k])
        # No, that's not a tensor decomposition.
        #
        # A rank-1 term in the tensor decomposition is: U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
        # But that's trilinear, not bilinear.
        #
        # Let me re-read the task. "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
        # 
        # The matrix multiplication tensor T is a 3-index tensor T[i, j, k] where:
        # T[i, j, k] = 1 if the indices are "compatible" for the multiplication
        # Actually, the multiplication tensor for (n x m) * (m x p) is:
        # T(a, b, c) where a indexes A (size n*m), b indexes B (size m*p), c indexes C (size n*p)
        # T[(i,j), (j',k), (i',k')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
        # 
        # A rank-1 decomposition: T = sum_s U[:,s] * V[:,s] * W[:,s] (outer products)
        # So T[(i,j),(j',k),(i',k')] = sum_s U[i*m+j, s] * V[j'*p+k, s] * W[i'*n+k', s]
        # 
        # Wait, the output W has shape (p*n, rank) with index k*n+i. So W[k*n+i, s] corresponds to C[i,k].
        # 
        # So: delta_{j,j'} * delta_{i,i'} * delta_{k,k'} = sum_s U[i*m+j, s] * V[j'*p+k, s] * W[k*n+i, s]
        #
        # Let's verify the naive decomposition:
        # For each (i,j,k): U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
        # Then: sum_col U[i*m+j, col] * V[j'*p+k', col] * W[k'*n+i', col]
        # = sum over all (i0,j0,k0) where col corresponds to (i0,j0,k0):
        #   delta_{i,j=i0,j0} * delta_{j',k=j0,k0} * delta_{k',i=k0,n+i0}... 
        #   = delta_{j,j0} * delta_{j',j0} * delta_{k,k0} * delta_{i,i0} * delta_{k',k0} * delta_{i',i0}
        #   Wait, W[k*n+i, s] = 1 when k=k0 and i=i0
        #   So: delta_{j,j0} * delta_{j',j0} * delta_{k,k0} * delta_{i,i0} * delta_{k',k0} * delta_{i',i0}
        #   This is non-zero only when i=i0, j=j0, j'=j0, k=k0, k'=k0, i'=i0
        #   So it equals 1 if i=i', j=j', k=k' and 0 otherwise.
        #   That's exactly delta_{i,i'} * delta_{j,j'} * delta_{k,k'}. Correct!
        
        r = U.shape[1]
        # Check a few random entries
        import random
        random.seed(42)
        for _ in range(50):
            i = random.randint(0, n_ - 1)
            j = random.randint(0, m_ - 1)
            jp = random.randint(0, m_ - 1)
            k = random.randint(0, p_ - 1)
            ip = random.randint(0, n_ - 1)
            kp = random.randint(0, p_ - 1)
            
            expected = 1 if (j == jp and i == ip and k == kp) else 0
            actual = sum(U[i * m_ + j, s] * V[jp * p_ + k, s] * W[k * n_ + i, s] for s in range(r))
            if abs(float(actual) - expected) > tol:
                return False
        return True
    
    # Start with the naive rank-96 decomposition
    U_naive, V_naive, W_naive = naive_decomp(n, m, p)
    best_U, best_V, best_W = U_naive, V_naive, W_naive
    best_rank = 96
    
    if time.time() > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Strategy: Try blocking the multiplication
    # A (3x4) * B (4x8)
    # Split A into rows: A1 (1x4), A2 (2x4)
    # C = A1*B + A2*B where C1 = A1*B (1x8), C2 = A2*B (2x8)
    # C1: 1x4 * 4x8, rank = 1*4*8 = 32
    # C2: 2x4 * 4x8, rank = 2*4*8 = 64
    # Total naive blocked: 32 + 64 = 96 (same as before)
    
    # For C2 (2x4 * 4x8), can we do better?
    # Split B into B1 (4x4), B2 (4x4)
    # C2 = A2*B1 + A2*B2
    # A2*B1: 2x4 * 4x4, rank 32
    # A2*B2: 2x4 * 4x4, rank 32
    # Total: 64. No improvement.
    
    # Try Strassen-like on 2x2 blocks
    # For 2x4 * 4x4:
    # A2 (2x4) = [A21 A22] where A21, A22 are 2x2
    # B1 (4x4) = [B11 B12; B13 B14] where each is 2x2
    # A2*B1 = [A21*B11 + A22*B13, A21*B12 + A22*B14]
    # Each 2x2 * 2x2 multiplication can use Strassen (rank 7) or naive (rank 8)
    # Using Strassen: 2 multiplications * 7 = 14 per 2x2 block
    # But we need to combine them
    
    # Actually, let's try a direct approach: use known rank results
    # R(1,4,8) = 32 (optimal, since min(n,p)*m = 1*8 = 8 but actually m=4, so it's 4*8=32? No)
    # For 1 x m x p: rank = m*p (naive) but can it be better?
    # C[i=0,k] = sum_j A[j] B[j,k]
    # This is a sum of m*p products, and they're all independent, so rank = m*p = 32.
    
    # For 2 x 4 x 8:
    # Let's try to use a tensor decomposition search or known results
    # Known: R(2,2,2) = 7 (Strassen)
    # R(2,2,8): Split p=8 into four 2s. R = 4 * R(2,2,2) = 4*7 = 28
    # Or: 2x2 * 2x8. Split m=2 into 1+1. R(2,1,8) + R(2,1,8). R(2,1,8) = 2*8 = 16. Total 32.
    # So 28 is better.
    # R(2,4,8): Split m=4 into 2+2. R(2,2,8) + R(2,2,8) = 28+28 = 56
    # Split p=8 into 4+4. R(2,4,4) + R(2,4,4). R(2,4,4): split m=4 into 2+2: 2*R(2,2,4). R(2,2,4): split p=4 into 2+2: 2*R(2,2,2) = 14. So R(2,4,4) = 28. R(2,4,8) = 56.
    # Split m=4 into 3+1: R(2,3,8) + R(2,1,8). R(2,1,8) = 16. R(2,3,8): split m=3 into 2+1: R(2,2,8)+R(2,1,8) = 28+16 = 44. Total 60.
    # So 56 seems best for 2x4x8.
    
    # Total for 3x4x8: R(1,4,8) + R(2,4,8) = 32 + 56 = 88
    
    # Let's construct the rank-88 decomposition
    # Part 1: 1x4 * 4x8 (rank 32)
    U1, V1, W1 = naive_decomp(1, 4, 8)
    
    # Part 2: 2x4 * 4x8 (rank 56)
    # Split m=4 into 2+2: two copies of 2x2 * 2x8
    # Each 2x2 * 2x8: split p=8 into four 2s: four copies of 2x2*2x2 (Strassen rank 7)
    # So 2x4*4x8 = 2 * 4 * 7 = 56
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def strassen_222():
        """Returns U, V, W for Strassen 2x2*2x2 decomposition"""
        r = 7
        U = np.zeros((4, r), dtype=Fraction)
        V = np.zeros((4, r), dtype=Fraction)
        W = np.zeros((4, r), dtype=Fraction)
        
        # A indices: 00->0, 01->1, 10->2, 11->3
        # B indices: 00->0, 01->1, 10->2, 11->3
        # C indices: 00->0, 01->1, 10->2, 11->3
        # W index: k*n+i = k*2+i
        
        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1  # A00
        U[3, 0] = 1  # A11
        V[0, 0] = 1  # B00
        V[3, 0] = 1  # B11
        W[0, 0] = 1  # C00
        W[3, 0] = 1  # C11
        
        # M2 = (A10+A11)*B00
        U[2, 1] = 1  # A10
        U[3, 1] = 1  # A11
        V[0, 1] = 1  # B00
        W[2, 1] = 1  # C10
        W[3, 1] = -1 # C11
        
        # M3 = A00*(B10-B11)
        U[0, 2] = 1  # A00
        V[2, 2] = 1  # B10
        V[3, 2] = -1 # B11
        W[1, 2] = 1  # C01
        W[3, 2] = 1  # C11
        
        # M4 = A11*(B01-B10)
        U[3, 3] = 1  # A11
        V[1, 3] = 1  # B01
        V[2, 3] = -1 # B10
        W[0, 3] = 1  # C00
        W[2, 3] = 1  # C10
        
        # M5 = (A00+A01)*B11
        U[0, 4] = 1  # A00
        U[1, 4] = 1  # A01
        V[3, 4] = 1  # B11
        W[0, 4] = -1 # C00
        W[1, 4] = 1  # C01
        
        # M6 = (A10-A00)*(B00+B10)
        U[2, 5] = 1  # A10
        U[0, 5] = -1 # A00
        V[0, 5] = 1  # B00
        V[2, 5] = 1  # B10
        W[3, 5] = 1  # C11
        
        # M7 = (A01-A11)*(B01+B11)
        U[1, 6] = 1  # A01
        U[3, 6] = -1 # A11
        V[1, 6] = 1  # B01
        V[3, 6] = 1  # B11
        W[0, 6] = 1  # C00
        W[1, 6] = 1  # C01
        
        return U, V, W
    
    def expand_222_to_2m2p(m_, p_):
        """Expand Strassen 2x2*2x2 to 2x2*m_ * 2x2*p_ by tiling"""
        r = 7 * m_ * p_
        U = np.zeros((2 * m_, r), dtype=Fraction)
        V = np.zeros((2 * p_, r), dtype=Fraction)
        W = np.zeros((2 * p_, r), dtype=Fraction)
        
        base_U, base_V, base_W = strassen_222()
        
        for ii in range(m_):
            for kk in range(p_):
                col_offset = (ii * p_ + kk) * 7
                for s in range(7):
                    # A index: (2*ii + i_row, 2*kk + j_col) -> row 2*ii + i_row, col 2*kk + j_col
                    # But U is (2*m_, r), indexed by row 0..2*m_-1
                    # For 2x2 block at (ii, kk): A rows 2*ii, 2*ii+1; B cols 2*kk, 2*kk+1
                    # U row: 2*ii + i_row where i_row in {0,1}
                    # V col: 2*kk + j_col where j_col in {0,1}
                    # W index: k*n+i = (2*kk + k_row)*2 + (2*ii + i_row)
                    
                    for i_row in range(2):
                        u_val = base_U[i_row, s]
                        if u_val != 0:
                            U[2*ii + i_row, col_offset + s] += u_val
                    
                    for j_col in range(2):
                        v_val = base_V[j_col, s]
                        if v_val != 0:
                            V[2*kk + j_col, col_offset + s] += v_val
                    
                    for i_row in range(2):
                        for k_row in range(2):
                            w_val = base_W[k_row * 2 + i_row, s]
                            if w_val != 0:
                                W[(2*kk + k_row) * 2 + (2*ii + i_row), col_offset + s] += w_val
        
        return U, V, W
    
    # Construct 2x4 * 4x8 decomposition
    # Split m=4 into two 2s, p=8 into four 2s
    # 2x4 * 4x8 = sum over 2 m-blocks, 4 p-blocks of 2x2 * 2x2
    U2_parts = []
    V2_parts = []
    W2_parts = []
    
    for m_block in range(2):
        for p_block in range(4):
            U_block, V_block, W_block = expand_222_to_2m2p(1, 1)
            # Shift indices
            U_shifted = np.zeros((8, 7), dtype=Fraction)  # 2*4 = 8
            V_shifted = np.zeros((8, 7), dtype=Fraction)  # 4*2 = 8
            W_shifted = np.zeros((8, 7), dtype=Fraction)  # 8*2 = 8
            
            for i in range(2):
                U_shifted[2*m_block + i, :] = U_block[i, :]
            for j in range(2):
                V_shifted[2*p_block + j, :] = V_block[j, :]
            for i in range(2):
                for k in range(2):
                    W_shifted[k * 2 + i, :] = W_block[k * 2 + i, :]
            
            U2_parts.append(U_shifted)
            V2_parts.append(V_shifted)
            W2_parts.append(W_shifted)
    
    U2 = np.hstack(U2_parts)
    V2 = np.hstack(V2_parts)
    W2 = np.hstack(W2_parts)
    
    # Now combine with Part 1 (1x4 * 4x8)
    # Part 1: U1 (4, 32), V1 (32, 32), W1 (8, 32)
    # Part 2: U2 (8, 56), V2 (32, 56), W2 (24, 56)
    
    # We need to embed Part 1 into the full 3x4 * 4x8 problem
    # A is 3x4, B is 4x8, C is 3x8
    # Part 1 handles row 0 of A: A[0,:]*B -> C[0,:]
    # Part 2 handles rows 1,2 of A: A[1:,:]*B -> C[1:,:]
    
    U_full = np.zeros((12, 88), dtype=Fraction)  # 3*4 = 12
    V_full = np.zeros((32, 88), dtype=Fraction)  # 4*8 = 32
    W_full = np.zeros((24, 88), dtype=Fraction)  # 8*3 = 24
    
    # Part 1: row 0
    U_full[0:4, 0:32] = U1  # U1 is (4, 32), for A[0,0:4]
    V_full[:, 0:32] = V1    # V1 is (32, 32)
    W_full[0:8, 0:32] = W1  # W1 is (8, 32), for C[0,0:8], W index = k*3+0 = k
    
    # Part 2: rows 1,2
    U_full[4:12, 32:88] = U2  # U2 is (8, 56), for A[1:3,0:4]
    V_full[:, 32:88] = V2      # V2 is (32, 56)
    # W2 is (8, 56) but we need (24, 56) for C[1:3, 0:8]
    # W2 index: k*2+i where i in {0,1} corresponds to rows 1,2
    # Full W index: k*3+i_full where i_full in {1,2}
    for i_local in range(2):
        for k in range(8):
            W_full[k*3 + 1 + i_local, 32:88] = W2[k*2 + i_local, :]
    
    # Verify the full decomposition
    if verify_decomp(U_full, V_full, W_full, n, m, p):
        best_U, best_V, best_W = U_full, V_full, W_full
        best_rank = 88
        print(f"Found rank-88 decomposition, verified.")
    else:
        print("Rank-88 decomposition failed verification, using naive.")
    
    if time.time() > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to improve further
    # Can we do better than 88?
    # R(1,4,8) = 32 is optimal
    # R(2,4,8) = 56 from our construction
    # Is R(2,4,8) < 56 possible?
    # Lower bound: R(2,4,8) >= max(2*4, 4*8, 2*8) = max(8, 32, 16) = 32
    # Also R(2,4,8) >= 2*R(2,2,8) if we can't share, but we can potentially share
    # Known results: R(2,3,3) = 12? R(2,4,4) = ?
    
    # Let's try a different blocking for 2x4x8
    # Split p=8 into 2+6: R(2,4,2) + R(2,4,6)
    # R(2,4,2): split m=4 into 2+2: 2*R(2,2,2) = 2*7 = 14
    # R(2,4,6): split m=4 into 2+2: 2*R(2,2,6). R(2,2,6): split p=6 into 3+3: 2*R(2,2,3). R(2,2,3): split p=3 into 1+2: R(2,2,1)+R(2,2,2) = 4+7 = 11. So R(2,2,6) = 22. R(2,4,6) = 44. Total 14+44 = 58. Worse.
    
    # Split m=4 into 1+3: R(2,1,8) + R(2,3,8) = 16 + 44 = 60. Worse.
    
    # So 56 seems good for 2x4x8.
    
    # Alternative: don't split n=3 into 1+2, but try a different approach
    # Maybe use a tensor decomposition search for the full 3x4x8
    
    # For now, return the best we have
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
