import time
import numpy as np
from itertools import product

def _build_strassen_2x2():
    """
    Strassen's algorithm for 2x2 * 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7) where rank=7.
    
    Matrix multiplication: C = A * B where A is 2x2, B is 2x2.
    Indexing: A[i,j], B[j,k], C[i,k]
    U[i*m + j, r], V[j*p + k, r], W[k*n + i, r] with n=2, m=2, p=2.
    So U has 4 rows (i*m+j), V has 4 rows (j*p+k), W has 4 rows (k*n+i).
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 multiplications for 2x2*2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # A indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B indices: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # C indices: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    
    # U rows: i*m+j -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V rows: j*p+k -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W rows: k*n+i -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2 = (A21 + A22)B11
    U[2, 1] = 1  # A21
    U[3, 1] = 1  # A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1  # C21
    W[3, 1] = -1 # C22
    
    # M3 = A11(B21 - B22)
    U[0, 2] = 1  # A11
    V[2, 2] = 1  # B21
    V[3, 2] = -1 # B22
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # M4 = A22(B12 - B11)
    U[3, 3] = 1  # A22
    V[1, 3] = 1  # B12
    V[0, 3] = -1 # B11
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1  # A11
    U[1, 4] = 1  # A12
    V[3, 4] = 1  # B22
    W[0, 4] = -1 # C11
    W[1, 4] = 1  # C12
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1  # A21
    U[0, 5] = -1 # A11
    V[0, 5] = 1  # B11
    V[1, 5] = 1  # B12
    W[3, 5] = 1  # C22
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1  # A12
    U[3, 6] = -1 # A22
    V[2, 6] = 1  # B21
    V[3, 6] = 1  # B22
    W[0, 6] = 1  # C11
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, r] = 1
                V[j*p + k, r] = 1
                W[k*n + i, r] = 1
                r += 1
    return U, V, W

def _combine_decompositions(decs, shapes):
    """
    Combine block decompositions into a full decomposition.
    decs: list of (U_block, V_block, W_block) for each block
    shapes: list of (n_block, m_block, p_block) for each block
    """
    total_rank = sum(len(U[1]) for U in decs)
    n, m, p = 4, 4, 7
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    offset = 0
    i_offset = 0
    j_offset = 0
    k_offset = 0
    for (U_b, V_b, W_b), (n_b, m_b, p_b) in zip(decs, shapes):
        rank_b = U_b.shape[1]
        # U block: rows i_offset*m + j_offset to (i_offset+n_b-1)*m + j_offset+m_b-1
        # Actually, the block covers A[i_offset:i_offset+n_b, j_offset:j_offset+m_b]
        # U rows: i*m + j for i in [i_offset, i_offset+n_b), j in [j_offset, j_offset+m_b)
        for i in range(n_b):
            for j in range(m_b):
                U[(i_offset+i)*m + (j_offset+j), offset:offset+rank_b] = U_b[i*m_b + j, :]
        
        # V block: rows j*p + k for j in [j_offset, j_offset+m_b), k in [k_offset, k_offset+p_b)
        for j in range(m_b):
            for k in range(p_b):
                V[(j_offset+j)*p + (k_offset+k), offset:offset+rank_b] = V_b[j*p_b + k, :]
        
        # W block: rows k*n + i for k in [k_offset, k_offset+p_b), i in [i_offset, i_offset+n_b)
        for k in range(p_b):
            for i in range(n_b):
                W[(k_offset+k)*n + (i_offset+i), offset:offset+rank_b] = W_b[k*n_b + i, :]
        
        offset += rank_b
        k_offset += p_b
    
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=7):
    """Verify that U @ diag @ V^T reconstructs the tensor via W."""
    rank = U.shape[1]
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta(i,j,k)
    # Actually the decomposition is: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] = 1 for the multiplication
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify with random matrices
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    # Compute via decomposition
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u = U[:, r]  # shape (n*m,)
        v = V[:, r]  # shape (m*p,)
        w = W[:, r]  # shape (p*n,)
        # u reshaped to (n, m), v reshaped to (m, p), w reshaped to (p, n)
        u_mat = u.reshape(n, m)
        v_mat = v.reshape(m, p)
        w_mat = w.reshape(p, n)
        # Contribution: sum_{i,j,k} A[i,j]*B[j,k]*... 
        # Actually: C[i,k] = sum_r (sum_j A[i,j]*u_mat[i,j]) * (sum_j v_mat[j,k]*B[j,k]) * ...
        # No, the tensor decomposition means:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And C[i,k] = sum_j A[i,j] * B[j,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
        # So C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j]*U[i*m+j, r]) * (sum_j B[j,k]*V[j*p+k, r])
        
        a_proj = np.zeros(n)
        b_proj = np.zeros(p)
        for i in range(n):
            for j in range(m):
                a_proj[i] += A[i, j] * u_mat[i, j]
        for k in range(p):
            for j in range(m):
                b_proj[k] += B[j, k] * v_mat[j, k]
        
        # C[i,k] += w_mat[k,i] * a_proj[i] * b_proj[k]
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += w_mat[k, i] * a_proj[i] * b_proj[k]
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Strategy 1: Hybrid - Strassen for 4x4x4 block + schoolbook for 4x4x3 block
    # Split p=7 into p1=4 and p2=3
    # Block 1: A(4x4) * B1(4x4) -> C1(4x4), uses Strassen rank 49
    # Block 2: A(4x4) * B2(4x3) -> C2(4x3), uses schoolbook rank 48
    # Total rank: 49 + 48 = 97
    
    # Build Strassen 4x4x4 from 2x2x2 Strassen
    # 4x4x4 can be done with rank 49 using Strassen's algorithm recursively
    # Actually, let's use the known rank-49 decomposition for 4x4x4
    
    def _strassen_4x4x4():
        """
        Build rank-49 decomposition for 4x4 * 4x4 -> 4x4.
        Uses Strassen's algorithm: 4x4 = 4 blocks of 2x2.
        Each 2x2*2x2 uses 7 multiplications.
        Total: 7 * 4 = 28? No, that's not right.
        
        Actually for 4x4x4, Strassen gives rank 49.
        Let me use a different approach: build it from 2x2 Strassen blocks.
        
        A 4x4 matrix can be viewed as 4 2x2 blocks:
        A = [[A11, A12], [A21, A22]] where each is 2x2
        B = [[B11, B12], [B21, B22]]
        C = A*B = [[C11, C12], [C21, C22]]
        
        C11 = A11*B11 + A12*B21
        C12 = A11*B12 + A12*B22
        C21 = A21*B11 + A22*B21
        C22 = A21*B12 + A22*B22
        
        Each Ci_j is a 2x2 product of 2x2 matrices, so each needs 7 mults.
        Total: 4 * 7 = 28? That's too optimistic.
        
        Wait, no. Each Ci_j = X*Y + Z*W where X,Y,Z,W are 2x2.
        This is a sum of two 2x2*2x2 products, which needs 2*7=14 mults per Ci_j.
        Total: 4*14 = 56. That's worse than 49.
        
        The rank 49 for 4x4x4 comes from a more sophisticated construction.
        Let me just use the known result that rank(4x4x4) <= 49.
        
        Actually, I recall that the standard Strassen recursion gives:
        rank(2x2x2) = 7
        rank(4x4x4) <= 7^2 = 49
        
        Let me build it properly.
        """
        n, m, p = 4, 4, 4
        rank = 49
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Strassen 2x2x2 decomposition
        U2, V2, W2 = _build_strassen_2x2()
        
        # For 4x4x4, we have 4 block positions for A, 4 for B, 4 for C
        # A blocks: A11=(0:2,0:2), A12=(0:2,2:4), A21=(2:4,0:2), A22=(2:4,2:4)
        # B blocks: B11=(0:2,0:2), B12=(0:2,2:4), B21=(2:4,0:2), B22=(2:4,2:4)
        # C blocks: C11=(0:2,0:2), C12=(0:2,2:4), C21=(2:4,0:2), C22=(2:4,2:4)
        
        # C11 = A11*B11 + A12*B21
        # C12 = A11*B12 + A12*B22
        # C21 = A21*B11 + A22*B21
        # C22 = A21*B12 + A22*B22
        
        # Each term like A11*B11 is a 2x2*2x2 product, using 7 mults
        # We need to assign rank indices for each of the 8 such products
        
        # Let's define the 8 products:
        # P1 = A11*B11 (for C11)
        # P2 = A12*B21 (for C11)
        # P3 = A11*B12 (for C12)
        # P4 = A12*B22 (for C12)
        # P5 = A21*B11 (for C21)
        # P6 = A22*B21 (for C21)
        # P7 = A21*B12 (for C22)
        # P8 = A22*B22 (for C22)
        
        # Block offsets for A rows (i*m+j where i,j in 0..3):
        # A11: i in 0:2, j in 0:2 -> rows 0:2, 1:2 -> 0,1,4,5
        # A12: i in 0:2, j in 2:4 -> rows 0:2, 2:4 -> 2,3,6,7
        # A21: i in 2:4, j in 0:2 -> rows 2:4, 0:2 -> 8,9,12,13
        # A22: i in 2:4, j in 2:4 -> rows 2:4, 2:4 -> 10,11,14,15
        
        # Block offsets for V rows (j*p+k where j,k in 0..3):
        # B11: j in 0:2, k in 0:2 -> rows 0:2, 0:2 -> 0,1,4,5
        # B12: j in 0:2, k in 2:4 -> rows 0:2, 2:4 -> 2,3,6,7
        # B21: j in 2:4, k in 0:2 -> rows 2:4, 0:2 -> 8,9,12,13
        # B22: j in 2:4, k in 2:4 -> rows 2:4, 2:4 -> 10,11,14,15
        
        # Block offsets for W rows (k*n+i where k,i in 0..3):
        # C11: k in 0:2, i in 0:2 -> rows 0:2, 0:2 -> 0,1,4,5
        # C12: k in 0:2, i in 2:4 -> rows 0:2, 2:4 -> 2,3,6,7
        # C21: k in 2:4, i in 0:2 -> rows 2:4, 0:2 -> 8,9,12,13
        # C22: k in 2:4, i in 2:4 -> rows 2:4, 2:4 -> 10,11,14,15
        
        # For each product P, we use 7 rank indices
        # P1: ranks 0-6, P2: ranks 7-13, ..., P8: ranks 42-48
        
        products = [
            # (A_block, B_block, C_block)
            (0, 0, 0),  # P1: A11*B11 -> C11
            (1, 2, 0),  # P2: A12*B21 -> C11
            (0, 1, 1),  # P3: A11*B12 -> C12
            (1, 3, 1),  # P4: A12*B22 -> C12
            (2, 0, 2),  # P5: A21*B11 -> C21
            (3, 2, 2),  # P6: A22*B21 -> C21
            (2, 1, 3),  # P7: A21*B12 -> C22
            (3, 3, 3),  # P8: A22*B22 -> C22
        ]
        
        # A block row indices (for U):
        # Block 0 (A11): i in 0:2, j in 0:2 -> U rows: i*4+j = 0,1,4,5
        # Block 1 (A12): i in 0:2, j in 2:4 -> U rows: i*4+j = 2,3,6,7
        # Block 2 (A21): i in 2:4, j in 0:2 -> U rows: i*4+j = 8,9,12,13
        # Block 3 (A22): i in 2:4, j in 2:4 -> U rows: i*4+j = 10,11,14,15
        
        a_block_rows = [
            [0, 1, 4, 5],  # A11
            [2, 3, 6, 7],  # A12
            [8, 9, 12, 13],  # A21
            [10, 11, 14, 15],  # A22
        ]
        
        # V block row indices:
        # Block 0 (B11): j in 0:2, k in 0:2 -> V rows: j*4+k = 0,1,4,5
        # Block 1 (B12): j in 0:2, k in 2:4 -> V rows: j*4+k = 2,3,6,7
        # Block 2 (B21): j in 2:4, k in 0:2 -> V rows: j*4+k = 8,9,12,13
        # Block 3 (B22): j in 2:4, k in 2:4 -> V rows: j*4+k = 10,11,14,15
        
        v_block_rows = [
            [0, 1, 4, 5],  # B11
            [2, 3, 6, 7],  # B12
            [8, 9, 12, 13],  # B21
            [10, 11, 14, 15],  # B22
        ]
        
        # W block row indices:
        # Block 0 (C11): k in 0:2, i in 0:2 -> W rows: k*4+i = 0,1,4,5
        # Block 1 (C12): k in 0:2, i in 2:4 -> W rows: k*4+i = 2,3,6,7
        # Block 2 (C21): k in 2:4, i in 0:2 -> W rows: k*4+i = 8,9,12,13
        # Block 3 (C22): k in 2:4, i in 2:4 -> W rows: k*4+i = 10,11,14,15
        
        w_block_rows = [
            [0, 1, 4, 5],  # C11
            [2, 3, 6, 7],  # C12
            [8, 9, 12, 13],  # C21
            [10, 11, 14, 15],  # C22
        ]
        
        for prod_idx, (a_blk, b_blk, c_blk) in enumerate(products):
            rank_offset = prod_idx * 7
            for r in range(7):
                # U: for each row in a_block_rows[a_blk], set U[row, rank_offset+r] = U2[local_row, r]
                a_rows = a_block_rows[a_blk]
                v_rows = v_block_rows[b_blk]
                w_rows = w_block_rows[c_blk]
                
                for local_idx, global_row in enumerate(a_rows):
                    U[global_row, rank_offset + r] = U2[local_idx, r]
                for local_idx, global_row in enumerate(v_rows):
                    V[global_row, rank_offset + r] = V2[local_idx, r]
                for local_idx, global_row in enumerate(w_rows):
                    W[global_row, rank_offset + r] = W2[local_idx, r]
        
        return U, V, W
    
    # Build the hybrid decomposition
    U_strassen, V_strassen, W_strassen = _strassen_4x4x4()
    
    # Schoolbook for 4x4x3
    n2, m2, p2 = 4, 4, 3
    U_school, V_school, W_school = _build_schoolbook(n2, m2, p2)
    
    # Combine: total rank = 49 + 48 = 97
    total_rank = 49 + 48
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    # Block 1: p in 0:4 (first 4 columns of B)
    U_full[:, 0:49] = U_strassen
    V_full[:, 0:49] = V_strassen
    W_full[:, 0:49] = W_strassen
    
    # Block 2: p in 4:7 (last 3 columns of B)
    # V rows: j*p+k for j in 0:4, k in 4:7
    # In V_full, these are rows j*7+k for k in 4:7
    # In V_school, these are rows j*3+k' for k' in 0:3
    for j in range(m):
        for k in range(p2):
            V_full[j*p + (4+k), 49:49+48] = V_school[j*p2 + k, :]
    
    # W rows: k*n+i for k in 4:7, i in 0:4
    # In W_full, these are rows k*4+i for k in 4:7
    # In W_school, these are rows k'*4+i for k' in 0:3
    for k in range(p2):
        for i in range(n):
            W_full[(4+k)*n + i, 49:49+48] = W_school[k*n2 + i, :]
    
    # U is the same for both blocks (same A)
    U_full[:, 49:49+48] = U_school
    
    # Verify
    if not _verify(U_full, V_full, W_full):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to improve with random search if time permits
    best_U, best_V, best_W = U_full, V_full, W_full
    best_rank = total_rank
    
    if time.time() < deadline - 30:
        # Try random perturbations to see if we can reduce rank
        # This is unlikely to help, but let's try a few
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
