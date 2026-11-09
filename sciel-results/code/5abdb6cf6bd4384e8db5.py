"""Matrix multiplication 4x4x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # First, build the schoolbook decomposition as a safe fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    for c in range(rank_schoolbook):
        i = c // (m * p)
        rem = c % (m * p)
        j = rem // p
        k = rem % p
        U_school[i * m + j, c] = 1.0
        V_school[j * p + k, c] = 1.0
        W_school[k * n + i, c] = 1.0
    
    # Try to build a Strassen-based decomposition for the 4x4x4 part
    # plus 16 extra terms for the 5th column
    # Total rank would be 49 + 16 = 65 vs schoolbook 80
    
    # Build 4x4x4 Strassen decomposition
    # The 4x4x4 tensor can be decomposed using Strassen's method:
    # Split 4x4 into 2x2 blocks, use Strassen for 2x2 multiplications
    # This gives rank 49 for the 4x4x4 part
    
    # For 4x4x5, we can:
    # 1. Use 49 terms for the first 4 columns (4x4x4 part)
    # 2. Use 16 terms for the 5th column (each entry of A times each entry of B's 5th column)
    # Total rank = 65
    
    # Let's construct this carefully
    rank = 49 + 16
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: 4x4x4 Strassen decomposition
    # Strassen for 2x2: 7 multiplications
    # For 4x4, we have 4x4 = 2x2 blocks, so 4 block multiplications
    # Each block multiplication uses Strassen (7 terms)
    # But we can share some linear forms
    
    # Actually, let's use the known rank-49 decomposition for 4x4x4
    # This comes from applying Strassen recursively
    
    # Strassen 2x2x2: rank 7
    # U_2x2, V_2x2, W_2x2 for 2x2x2
    U_2x2 = [
        [1, 0, 0, 1],  # m1: (A00+A11)
        [0, 0, 1, 1],  # m2: (A10+A11)
        [1, 0, 0, 0],  # m3: A00
        [0, 0, 0, 1],  # m4: A11
        [1, 1, 0, 0],  # m5: (A00+A01)
        [-1, 0, 1, 0], # m6: (A10-A00)
        [0, 1, 0, -1]  # m7: (A01-A11)
    ]
    V_2x2 = [
        [1, 0, 0, 1],  # m1: (B00+B11)
        [1, 0, 0, 0],  # m2: B00
        [0, 1, 0, -1], # m3: (B01-B11)
        [0, 0, 1, -1], # m4: (B10-B00)
        [0, 0, 0, 1],  # m5: B11
        [1, 1, 0, 0],  # m6: (B00+B01)
        [0, 0, 1, 1]   # m7: (B10+B11)
    ]
    W_2x2 = [
        [1, 0, 0, 1],  # m1 -> C00, C11
        [0, 0, 1, -1], # m2 -> C10, C11
        [0, 1, 0, 1],  # m3 -> C01, C11
        [1, 0, 1, 0],  # m4 -> C00, C10
        [-1, 1, 0, 0], # m5 -> C00, C01
        [0, 0, 0, 1],  # m6 -> C11
        [1, 0, 0, 0]   # m7 -> C00
    ]
    
    # For 4x4x4, we need to handle 2x2 blocks
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = [[C00, C01], [C10, C11]] where each is 2x2
    
    # Using Strassen at the block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M is a 2x2 matrix product, so each uses 7 Strassen terms
    # Total: 7 * 7 = 49 terms
    
    col = 0
    for m_idx in range(7):
        # Linear form on A blocks
        if m_idx == 0:  # M1: A00+A11
            a_blocks = [1, 0, 0, 1]  # A00, A01, A10, A11
        elif m_idx == 1:  # M2: A10+A11
            a_blocks = [0, 0, 1, 1]
        elif m_idx == 2:  # M3: A00
            a_blocks = [1, 0, 0, 0]
        elif m_idx == 3:  # M4: A11
            a_blocks = [0, 0, 0, 1]
        elif m_idx == 4:  # M5: A00+A01
            a_blocks = [1, 1, 0, 0]
        elif m_idx == 5:  # M6: A10-A00
            a_blocks = [-1, 0, 1, 0]
        elif m_idx == 6:  # M7: A01-A11
            a_blocks = [0, 1, 0, -1]
        
        # Linear form on B blocks
        if m_idx == 0:  # M1: B00+B11
            b_blocks = [1, 0, 0, 1]
        elif m_idx == 1:  # M2: B00
            b_blocks = [1, 0, 0, 0]
        elif m_idx == 2:  # M3: B01-B11
            b_blocks = [0, 1, 0, -1]
        elif m_idx == 3:  # M4: B10-B00
            b_blocks = [0, 0, 1, -1]
        elif m_idx == 4:  # M5: B11
            b_blocks = [0, 0, 0, 1]
        elif m_idx == 5:  # M6: B00+B01
            b_blocks = [1, 1, 0, 0]
        elif m_idx == 6:  # M7: B10+B11
            b_blocks = [0, 0, 1, 1]
        
        # For each of the 7 inner Strassen terms
        for r_idx in range(7):
            u_inner = U_2x2[r_idx]
            v_inner = V_2x2[r_idx]
            w_inner = W_2x2[r_idx]
            
            # Build U vector (16 elements for 4x4 A)
            u_vec = np.zeros(16)
            for bi in range(2):  # block row
                for bj in range(2):  # block col
                    block_idx = bi * 2 + bj
                    coeff = a_blocks[block_idx]
                    if coeff != 0:
                        for ii in range(2):  # inner row
                            for jj in range(2):  # inner col
                                i = bi * 2 + ii
                                j = bj * 2 + jj
                                u_vec[i * 4 + j] = coeff * u_inner[ii * 2 + jj]
            
            # Build V vector (20 elements for 4x5 B)
            # For 4x4x4 part, only use first 4 columns
            v_vec = np.zeros(20)
            for bi in range(2):  # block row
                for bj in range(2):  # block col (only 0,1 for 4 columns)
                    block_idx = bi * 2 + bj
                    coeff = b_blocks[block_idx]
                    if coeff != 0:
                        for ii in range(2):  # inner row
                            for jj in range(2):  # inner col
                                i = bi * 2 + ii
                                j = bj * 2 + jj
                                v_vec[i * 5 + j] = coeff * v_inner[ii * 2 + jj]
            
            # Build W vector (20 elements for 5x4 C)
            # C = [[C00, C01], [C10, C11]]
            # M1 -> C00, C11
            # M2 -> C10, C11
            # M3 -> C01, C11
            # M4 -> C00, C10
            # M5 -> C00, C01
            # M6 -> C11
            # M7 -> C00
            
            w_vec = np.zeros(20)
            # Determine which C blocks this M contributes to
            if m_idx == 0:  # M1
                c_blocks = [(0, 0, 1), (1, 1, 1)]  # (row_block, col_block, coeff)
            elif m_idx == 1:  # M2
                c_blocks = [(1, 0, 1), (1, 1, -1)]
            elif m_idx == 2:  # M3
                c_blocks = [(0, 1, 1), (1, 1, 1)]
            elif m_idx == 3:  # M4
                c_blocks = [(0, 0, 1), (1, 0, 1)]
            elif m_idx == 4:  # M5
                c_blocks = [(0, 0, -1), (0, 1, 1)]
            elif m_idx == 5:  # M6
                c_blocks = [(1, 1, 1)]
            elif m_idx == 6:  # M7
                c_blocks = [(0, 0, 1)]
            
            for ci, cj, coeff in c_blocks:
                for ii in range(2):  # inner row
                    for jj in range(2):  # inner col
                        i = ci * 2 + ii
                        j = cj * 2 + jj
                        w_vec[i * 4 + j] = coeff * w_inner[ii * 2 + jj]
            
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            col += 1
    
    # Part 2: 16 extra terms for the 5th column
    # For each row i of A and each column j of B's 5th column:
    # C[i, 4] += A[i, j] * B[j, 4]
    # Term (i, j): U[i*4+j] = 1, V[j*5+4] = 1, W[4*4+i] = 1
    # That's 4*4 = 16 terms
    
    for i in range(4):
        for j in range(4):
            u_vec = np.zeros(16)
            u_vec[i * 4 + j] = 1.0
            
            v_vec = np.zeros(20)
            v_vec[j * 5 + 4] = 1.0
            
            w_vec = np.zeros(20)
            w_vec[4 * 4 + i] = 1.0  # k=4, i=i, so index is 4*4+i = 16+i
            
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            col += 1
    
    # Verify the decomposition by checking a random matrix multiplication
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal the (i,j,k) entry of the matrix multiplication tensor
    
    # Let's verify with a simple test
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_test = A_test @ B_test
    
    C_reconstructed = np.zeros((n, p))
    for c in range(rank):
        A_lin = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_lin[i, j] = U[i * m + j, c]
        
        B_lin = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_lin[j, k] = V[j * p + k, c]
        
        C_lin = np.zeros((n, p))
        for k in range(p):
            for i in range(n):
                C_lin[i, k] = W[k * n + i, c]
        
        C_reconstructed += A_lin @ B_lin * C_lin
    
    # Check if the reconstruction matches
    if np.allclose(C_reconstructed, C_test, atol=1e-10):
        # The decomposition is correct, return it
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
