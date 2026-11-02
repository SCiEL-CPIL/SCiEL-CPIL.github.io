"""Matrix multiplication 4x4 * 4x8 - rank-minimal tensor decomposition via two-level Strassen."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U: (4, 7) - U[i*2+j, c]
    V: (4, 7) - V[j*2+k, c]
    W: (4, 7) - W[k*2+i, c]  (trace convention: k*n+i with n=2)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a11+a22)(b11+b22) -> contributes to c11 and c22
    # a11=U[0], a22=U[3], b11=V[0], b22=V[3]
    # c11=W[0*2+0]=W[0], c22=W[1*2+1]=W[3]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # W[k*2+i]: c11->k=0,i=0->W[0]; c22->k=1,i=1->W[3]
    
    # M2 = (a21+a22)b11 -> contributes to c21
    # a21=U[2], a22=U[3], b11=V[0]
    # c21=W[0*2+1]=W[1]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # W[k*2+i]: c21->k=0,i=1->W[1]
    
    # M3 = a11(b12-b22) -> contributes to c12
    # a11=U[0], b12=V[1], b22=V[3]
    # c12=W[1*2+0]=W[2]
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1  # W[k*2+i]: c12->k=1,i=0->W[2]
    
    # M4 = a22(b21-b11) -> contributes to c11 and c21
    # a22=U[3], b21=V[2], b11=V[0]
    # c11=W[0], c21=W[1]
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (a11+a12)b22 -> contributes to c11 (neg) and c12
    # a11=U[0], a12=U[1], b22=V[3]
    # c11=W[0] (coeff -1), c12=W[2]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (a21-a11)(b11+b12) -> contributes to c22
    # a21=U[2], a11=U[0], b11=V[0], b12=V[1]
    # c22=W[3]
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a12-a22)(b21+b22) -> contributes to c11
    # a12=U[1], a22=U[3], b21=V[2], b22=V[3]
    # c11=W[0]
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x2_2x4(U_s, V_s, W_s):
    """
    Build decomposition for 2x2 * 2x4 -> 2x4.
    U: (4, 14), V: (8, 14), W: (8, 14)
    U[i*2+j, c], V[j*4+k, c], W[k*2+i, c]
    
    Split 2x4 into two 2x2 halves.
    Left half: cols 0-1 of B, cols 0-1 of C
    Right half: cols 2-3 of B, cols 2-3 of C
    """
    U_2x4 = np.zeros((4, 14), dtype=float)
    V_2x4 = np.zeros((8, 14), dtype=float)
    W_2x4 = np.zeros((8, 14), dtype=float)
    
    # Left half (cols 0-1): embed Strassen 2x2*2x2
    # U rows 0-3, V rows 0-3 (j*4+k for j in {0,1}, k in {0,1}), W rows 0-3 (k*2+i for k in {0,1}, i in {0,1})
    for c in range(7):
        for r in range(4):
            if U_s[r, c] != 0:
                U_2x4[r, c] += U_s[r, c]
            if V_s[r, c] != 0:
                # V_s row r corresponds to j*2+k in 2x2, which is j*4+k in 2x4
                V_2x4[r, c] += V_s[r, c]
            if W_s[r, c] != 0:
                # W_s row r corresponds to k*2+i in 2x2
                W_2x4[r, c] += W_s[r, c]
    
    # Right half (cols 2-3): embed Strassen 2x2*2x2
    # U rows 0-3, V rows 4-7 (j*4+k for j in {0,1}, k in {2,3}), W rows 4-7 (k*2+i for k in {2,3}, i in {0,1})
    for c in range(7):
        for r in range(4):
            if U_s[r, c] != 0:
                U_2x4[r, 7 + c] += U_s[r, c]
            if V_s[r, c] != 0:
                # For right half, k is offset by 2: j*4+(k+2)
                # V_s row r = j*2+k_old, where k_old in {0,1}
                # In 2x4: j*4 + (k_old+2) = 4*j + k_old + 2
                # V_s index r = 2*j + k_old, so j = r//2, k_old = r%2
                j = r // 2
                k_old = r % 2
                v_idx = j * 4 + (k_old + 2)
                V_2x4[v_idx, 7 + c] += V_s[r, c]
            if W_s[r, c] != 0:
                # W_s row r = k_old*2 + i, where k_old in {0,1}
                # In 2x4: k_new = k_old + 2, so W index = k_new*2 + i = (k_old+2)*2 + i = k_old*2 + i + 4 = r + 4
                W_2x4[r + 4, 7 + c] += W_s[r, c]
    
    return U_2x4, V_2x4, W_2x4

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = U.shape[1]
    # Check: C[i,k] = sum_j A[i,j] * B[j,k] = sum_c U[i*m+j_idx, c] * V[j*p+k_idx, c] * W[k*n+i_idx, c]
    # More precisely: for each c, the term is (sum_{i,j} U[i*m+j,c] A[i,j]) * (sum_{j,k} V[j*p+k,c] B[j,k])
    # and this contributes W[k*n+i,c] * that to C[i,k]
    
    # Build the implied C tensor and compare with A@B
    # C[i,k] = sum_c W[k*n+i, c] * (sum_{i',j} U[i'*m+j, c] A[i',j]) * (sum_{j',k'} V[j'*p+k', c] B[j',k'])
    # Wait, that's not right. Let me think again.
    
    # The bilinear form: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor decomposition: C[i,k] = sum_c (sum_{i',j'} U[i'*m+j', c] A[i',j']) * (sum_{j'',k''} V[j''*p+k'', c] B[j'',k'']) * W[k*n+i, c]
    # No wait, that's not right either. The correct form is:
    # C[i,k] = sum_c U_i_c * V_j_c * W_k_c where the sums over i,j,k are implicit in the tensor structure.
    
    # Actually the standard form: C[i,k] = sum_{j,c} A[i,j] * B[j,k] * delta
    # Tensor: T[i,j,k] = delta_{ik}... no.
    
    # The matrix multiplication tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the identity in a sense)
    # And the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's verify: for random A, B, check that sum_c (U_vec_c @ A_flat) * (V_vec_c @ B_flat) * W[k*n+i, c] gives C[i,k]
    
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for c in range(rank):
        u_vec = U[:, c]  # shape (n*m,)
        v_vec = V[:, c]  # shape (m*p,)
        w_vec = W[:, c]  # shape (p*n,)
        
        # Reshape u_vec to (n, m), v_vec to (m, p)
        u_mat = u_vec.reshape(n, m)
        v_mat = v_vec.reshape(m, p)
        w_mat = w_vec.reshape(p, n)
        
        # The contribution to C from this rank-1 term:
        # C += (u_mat.T @ A) * (B @ v_mat) ... no
        # Actually: sum_{i,j} U[i*m+j,c] A[i,j] = trace(u_mat.T @ A) = sum(u_mat * A)
        # sum_{j,k} V[j*p+k,c] B[j,k] = sum(v_mat * B)
        # Then C[i,k] += W[k*n+i, c] * sum(u_mat * A) * sum(v_mat * B)
        # That's not right. The bilinear form is:
        # C[i,k] = sum_c W[k*n+i, c] * (sum_{i',j'} U[i'*m+j',c] A[i',j']) * ... 
        # No. The correct interpretation:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_c sum_j A[i,j] B[j,k] * [sum over all indices of the product]
        # The rank-1 term for factor c is:
        # (sum_{i,j} U[i*m+j,c] A[i,j]) * (sum_{j,k} V[j*p+k,c] B[j,k]) * W[k*n+i,c]
        # But this doesn't work because the indices are coupled.
        
        # Correct: The tensor T[i,j,k] for matrix mult is just 1 (for all i,j,k in range).
        # We need: 1 = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] for all i,j,k
        # Wait no. The matrix multiplication "tensor" is:
        # T[i,j,k] such that C[i,k] = sum_j A[i,j] T[i,j,k] B[j,k]
        # So T[i,j,k] = 1 for all i,j,k.
        
        pass
    
    # Direct verification: check the tensor identity
    # For each (i,j,k), verify: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    
    return max_err < 1e-9, max_err

def _build_full_decomposition():
    """
    Build the full 4x4 * 4x8 decomposition using two-level Strassen.
    
    Level 1: Split A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x4.
    C = [[C11, C12], [C21, C22]] where each C_ij is 2x4.
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each A_ij is 2x2, each B_ij is 2x4.
    Each product A_ij * B_kl is a 2x2 * 2x4 -> 2x4 multiplication.
    
    Level 2: Apply Strassen at the block level (2x2 blocks).
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each M_t is a 2x4 matrix (sum of two 2x2*2x4 products).
    Each 2x2*2x4 product has rank 14.
    Each M_t involves 2 such products, so rank 28 per M_t.
    Total rank = 7 * 28 = 196? No...
    
    Actually, each M_t is a sum of 2 terms, each being a 2x2*2x4 product.
    So M_t has rank at most 2*14 = 28.
    But we can combine them: the total rank is 7 * 28 = 196? That's worse than schoolbook.
    
    Let me reconsider. The total rank should be 7 (Strassen blocks) * 14 (per block) = 98.
    Each of the 7 Strassen multiplications M_t is itself a 2x2 * 2x4 multiplication.
    So M_t = (some 2x2 matrix) * (some 2x4 matrix).
    The "some 2x2 matrix" for M1 is (A11+A22), which is a 2x2 matrix.
    The "some 2x4 matrix" for M1 is (B11+B22), which is a 2x4 matrix.
    So M1 is a single 2x2 * 2x4 product, rank 14.
    Total: 7 * 14 = 98.
    """
    n, m, p = 4, 4, 8
    
    # Get base Strassen decomposition
    U_s, V_s, W_s = _strassen_2x2_2x2()
    
    # Get 2x2 * 2x4 decomposition
    U_2x4, V_2x4, W_2x4 = _build_2x2_2x4(U_s, V_s, W_s)
    # U_2x4: (4, 14), V_2x4: (8, 14), W_2x4: (8, 14)
    # U_2x4[i*2+j, c], V_2x4[j*4+k, c], W_2x4[k*2+i, c]
    
    total_rank = 7 * 14  # 98
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    # Block indices:
    # A = [[A11, A12], [A21, A22]], each 2x2
    # A11: rows 0-1, cols 0-1 -> U rows 0-3
    # A12: rows 0-1, cols 2-3 -> U rows 4-7
    # A21: rows 2-3, cols 0-1 -> U rows 8-11
    # A22: rows 2-3, cols 2-3 -> U rows 12-15
    #
    # B = [[B11, B12], [B21, B22]], each 2x4
    # B11: rows 0-1, cols 0-3 -> V rows 0-7
    # B12: rows 0-1, cols 4-7 -> V rows 8-15
    # B21: rows 2-3, cols 0-3 -> V rows 16-23
    # B22: rows 2-3, cols 4-7 -> V rows 24-31
    #
    # C = [[C11, C12], [C21, C22]], each 2x4
    # C11: rows 0-1, cols 0-3 -> W rows 0-7 (W[k*4+i] for k in 0-3, i in 0-1)
    # C12: rows 0-1, cols 4-7 -> W rows 8-15
    # C21: rows 2-3, cols 0-3 -> W rows 16-23
    # C22: rows 2-3, cols 4-7 -> W rows 24-31
    
    # Helper to embed a 2x2*2x4 product
    def embed_product(U_row_offset, V_row_offset, W_row_offset, col_offset, sign_u=1, sign_v=1, sign_w=1):
        """
        Embed the 2x2*2x4 decomposition.
        U_row_offset: starting row in U (4 rows for the 2x2 A block)
        V_row_offset: starting row in V (8 rows for the 2x4 B block)
        W_row_offset: starting row in W (8 rows for the 2x4 C block)
        col_offset: starting column
        sign_u, sign_v, sign_w: multiplicative signs
        """
        for c in range(14):
            for r in range(4):
                if U_2x4[r, c] != 0:
                    U[U_row_offset + r, col_offset + c] += sign_u * U_2x4[r, c]
            for r in range(8):
                if V_2x4[r, c] != 0:
                    V[V_row_offset + r, col_offset + c] += sign_v * V_2x4[r, c]
            for r in range(8):
                if W_2x4[r, c] != 0:
                    W[W_row_offset + r, col_offset + c] += sign_w * W_2x4[r, c]
    
    # M1 = (A11+A22)(B11+B22)
    # This is a sum of 4 products: A11*B11, A11*B22, A22*B11, A22*B22
    # But in Strassen, M1 is computed as a single "multiplication" of (A11+A22) and (B11+B22)
    # The factorization treats (A11+A22) as the A factor and (B11+B22) as the B factor.
    # So we need to embed the 2x2*2x4 decomposition where:
    #   A factor = A11 + A22 (rows 0-3 of U get contributions from both A11 and A22 positions)
    #   B factor = B11 + B22
    #   C factor goes to both C11 and C22 with appropriate signs
    
    # For M1: A_block = A11 + A22, B_block = B11 + B22
    # U part: A11 contributes to U rows 0-3, A22 contributes to U rows 12-15
    # But in the 2x2*2x4 decomposition, U has only 4 rows.
    # We need to "fold" A11 and A22 into the same 4-row U space.
    
    # The way to do this: for the U matrix, we add contributions from both A11 and A22
    # to the same 4 rows (0-3), effectively computing (A11+A22) as the A factor.
    
    # Let me redefine: for each Strassen multiplication, I'll create a "virtual" 2x2 A and 2x4 B
    # and embed the 2x2*2x4 decomposition with the appropriate row mappings.
    
    # M1 = (A11+A22)(B11+B22)
    # col_offset = 0
    # U: A11 at rows 0-3, A22 at rows 12-15, both map to U_2x4 rows 0-3
    # V: B11 at rows 0-7, B22 at rows 24-31, both map to V_2x4 rows 0-7
    # W: C11 at rows 0-7 (sign +1), C22 at rows 24-31 (sign +1)
    # Wait, M1 contributes +1 to C11 and +1 to C22.
    
    col = 0
    # M1: A=(A11+A22), B=(B11+B22), C contributes +C11 +C22
    # For U: rows 0-3 get U_2x4 from both A11 (rows 0-3 of full U) and A22 (rows 12-15 of full U)
    # We embed into U rows 0-3, but need to also account for A22's position.
    # Actually, the U matrix in the full decomposition has 16 rows.
    # For M1, the A factor is A11+A22, so U[i*m+j, c] should be:
    #   For A11 part: U_2x4[i*2+j, c] at U rows 0-3 (i=0,1; j=0,1)
    #   For A22 part: U_2x4[i*2+j, c] at U rows 12-15 (i=2,3; j=2,3)
    
    for c in range(14):
        # U contributions
        for r in range(4):
            if U_2x4[r, c] != 0:
                # A11 part: U rows 0-3
                U[r, col + c] += U_2x4[r, c]
                # A22 part: U rows 12-15 (r maps to same local index)
                U[12 + r, col + c] += U_2x4[r, c]
        # V contributions
        for r in range(8):
            if V_2x4[r, c] != 0:
                # B11 part: V rows 0-7
                V[r, col + c] += V_2x4[r, c]
                # B22 part: V rows 24-31
                V[24 + r, col + c] += V_2x4[r, c]
        # W contributions
        for r in range(8):
            if W_2x4[r, c] != 0:
                # C11 part: W rows 0-7
                W[r, col + c] += W_2x4[r, c]
                # C22 part: W rows 24-31
                W[24 + r, col + c] += W_2x4[r, c]
    col += 14
    
    # M2 = (A21+A22)B11
    # A=(A21+A22), B=B11, C contributes +C21
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                # A21 part: U rows 8-11
                U[8 + r, col + c] += U_2x4[r, c]
                # A22 part: U rows 12-15
                U[12 + r, col + c] += U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                # B11 part: V rows 0-7
                V[r, col + c] += V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                # C21 part: W rows 16-23
                W[16 + r, col + c] += W_2x4[r, c]
    col += 14
    
    # M3 = A11(B12-B22)
    # A=A11, B=(B12-B22), C contributes +C12
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                # A11 part: U rows 0-3
                U[r, col + c] += U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                # B12 part: V rows 8-15, sign +1
                V[8 + r, col + c] += V_2x4[r, c]
                # B22 part: V rows 24-31, sign -1
                V[24 + r, col + c] -= V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                # C12 part: W rows 8-15
                W[8 + r, col + c] += W_2x4[r, c]
    col += 14
    
    # M4 = A22(B21-B11)
    # A=A22, B=(B21-B11), C contributes +C11 +C21
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                # A22 part: U rows 12-15
                U[12 + r, col + c] += U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                # B21 part: V rows 16-23, sign +1
                V[16 + r, col + c] += V_2x4[r, c]
                # B11 part: V rows 0-7, sign -1
                V[r, col + c] -= V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                # C11 part: W rows 0-7
                W[r, col + c] += W_2x4[r, c]
                # C21 part: W rows 16-23
                W[16 + r, col + c] += W_2x4[r, c]
    col += 14
    
    # M5 = (A11+A12)B22
    # A=(A11+A12), B=B22, C contributes -C11 +C12
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                # A11 part: U rows 0-3
                U[r, col + c] += U_2x4[r, c]
                # A12 part: U rows 4-7
                U[4 + r, col + c] += U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                # B22 part: V rows 24-31
                V[24 + r, col + c] += V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                # C11 part: W rows 0-7, sign -1
                W[r, col + c] -= W_2x4[r, c]
                # C12 part: W rows 8-15, sign +1
                W[8 + r, col + c] += W_2x4[r, c]
    col += 14
    
    # M6 = (A21-A11)(B11+B12)
    # A=(A21-A11), B=(B11+B12), C contributes +C22
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                # A21 part: U rows 8-11, sign +1
                U[8 + r, col + c] += U_2x4[r, c]
                # A11 part: U rows 0-3, sign -1
                U[r, col + c] -= U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                # B11 part: V rows 0-7
                V[r, col + c] += V_2x4[r, c]
                # B12 part: V rows 8-15
                V[8 + r, col + c] += V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                # C22 part: W rows 24-31
                W[24 + r, col + c] += W_2x4[r, c]
    col += 14
    
    # M7 = (A12-A22)(B21+B22)
    # A=(A12-A22), B=(B21+B22), C contributes +C11
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                # A12 part: U rows 4-7, sign +1
                U[4 + r, col + c] += U_2x4[r, c]
                # A22 part: U rows 12-15, sign -1
                U[12 + r, col + c] -= U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                # B21 part: V rows 16-23
                V[16 + r, col + c] += V_2x4[r, c]
                # B22 part: V rows 24-31
                V[24 + r, col + c] += V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                # C11 part: W rows 0-7
                W[r, col + c] += W_2x4[r, c]
    col += 14
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 4, 4, 8
    
    try:
        U, V, W = _build_full_decomposition()
        
        # Verify correctness
        valid, max_err = _verify_decomposition(U, V, W, n, m, p)
        
        if valid:
            # Convert to lists with integer entries
            U_list = [[int(U[i][j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
            V_list = [[int(V[i][j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
            W_list = [[int(W[i][j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            # If verification fails, fall back to schoolbook
            raise ValueError(f"Verification failed with max_err={max_err}")
    
    except Exception as e:
        # Fallback to schoolbook (rank 128)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
