"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    num_U, num_V, num_W = n * m, m * p, p * n  # 20, 30, 24
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((num_U, rank_school), dtype=int)
    V_school = np.zeros((num_V, rank_school), dtype=int)
    W_school = np.zeros((num_W, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_factors = (U_school, V_school, W_school)
    best_rank = rank_school

    # Structured Construction using Strassen for 2x2 blocks
    # Partition:
    # I: [0,1] -> I1, [2,3] -> I2
    # J: [0,1] -> J1, [2,3] -> J2, [4] -> J3
    # K: [0,1] -> K1, [2,3] -> K2, [4,5] -> K3
    
    # Each (I_block, K_block) pair: 2x5 * 5x2
    # Decomposed as:
    #   A[:,J1] * B[J1,:] (2x2 * 2x2): rank 7
    #   A[:,J2] * B[J2,:] (2x2 * 2x2): rank 7
    #   A[:,J3] * B[J3,:] (2x1 * 1x2): rank 4
    # Total per block: 18. 6 blocks: 108.
    
    rank_structured = 108
    U_struct = np.zeros((num_U, rank_structured), dtype=int)
    V_struct = np.zeros((num_V, rank_structured), dtype=int)
    W_struct = np.zeros((num_W, rank_structured), dtype=int)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    def add_strassen_2x2(i_offset, j_offset, k_offset, start_col):
        """
        Adds Strassen rank-7 decomposition for 2x2 * 2x2 multiplication.
        A: rows i_offset..i_offset+1, cols j_offset..j_offset+1
        B: rows j_offset..j_offset+1, cols k_offset..k_offset+1
        C: rows i_offset..i_offset+1, cols k_offset..k_offset+1
        Returns number of columns used (7).
        """
        # Strassen 7 multiplications for 2x2*2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # Each M is a product of a linear combo of A entries and a linear combo of B entries.
        # In tensor decomposition, each M corresponds to one rank-1 term where:
        # U has the A coefficients, V has the B coefficients, W has the C contributions.
        
        # Define the 7 terms as (U_coeff, V_coeff, W_coeff)
        # U_coeff: dict mapping (i,j) -> coeff for A[i,j]
        # V_coeff: dict mapping (j,k) -> coeff for B[j,k]
        # W_coeff: dict mapping (k,i) -> coeff for C[i,k] (note: W index is k*n+i)
        
        terms = [
            # M1: (A11+A22)(B11+B22), contributes to C11(+1), C22(+1)
            ({(0,0): 1, (1,1): 1},
             {(0,0): 1, (1,1): 1},
             {(0,0): 1, (1,1): 1}),
            # M2: (A21+A22)B11, contributes to C21(+1), C22(-1)
            ({(1,0): 1, (1,1): 1},
             {(0,0): 1},
             {(0,1): 1, (1,1): -1}),
            # M3: A11(B21-B22), contributes to C12(+1), C22(+1)
            ({(0,0): 1},
             {(1,0): 1, (1,1): -1},
             {(0,1): 1, (1,1): 1}),
            # M4: A22(B12-B11), contributes to C11(+1), C21(+1)
            ({(1,1): 1},
             {(0,1): 1, (0,0): -1},
             {(0,0): 1, (0,1): 1}),
            # M5: (A11+A12)B22, contributes to C11(-1), C12(+1)
            ({(0,0): 1, (0,1): 1},
             {(1,1): 1},
             {(0,0): -1, (0,1): 1}),
            # M6: (A21-A11)(B11+B12), contributes to C22(+1)
            ({(1,0): 1, (0,0): -1},
             {(0,0): 1, (0,1): 1},
             {(1,1): 1}),
            # M7: (A12-A22)(B21+B22), contributes to C11(+1)
            ({(0,1): 1, (1,1): -1},
             {(1,0): 1, (1,1): 1},
             {(0,0): 1}),
        ]
        
        for t_idx, (u_coeff, v_coeff, w_coeff) in enumerate(terms):
            c = start_col + t_idx
            for (di, dj), coeff in u_coeff.items():
                U_struct[u_idx(i_offset + di, j_offset + dj), c] += coeff
            for (dj, dk), coeff in v_coeff.items():
                V_struct[v_idx(j_offset + dj, k_offset + dk), c] += coeff
            for (dk, di), coeff in w_coeff.items():
                W_struct[w_idx(k_offset + dk, i_offset + di), c] += coeff
        
        return 7
    
    def add_outer_2x1x2(i_offset, j_idx, k_offset, start_col):
        """
        Adds rank-4 decomposition for 2x1 * 1x2 multiplication (outer product).
        A: rows i_offset..i_offset+1, col j_idx
        B: row j_idx, cols k_offset..k_offset+1
        C: rows i_offset..i_offset+1, cols k_offset..k_offset+1
        Returns number of columns used (4).
        """
        # C[i,k] = A[i,j] * B[j,k]
        # This is a rank-4 tensor: each (i,k) pair needs its own term
        # Actually, this is the outer product structure:
        # T[i, j, k] = 1 for all i in {0,1}, k in {0,1}, j fixed
        # Rank is 4: each (i,k) combination
        for di in range(2):
            for dk in range(2):
                c = start_col + di * 2 + dk
                U_struct[u_idx(i_offset + di, j_idx), c] = 1
                V_struct[v_idx(j_idx, k_offset + dk), c] = 1
                W_struct[w_idx(k_offset + dk, i_offset + di), c] = 1
        return 4
    
    col = 0
    # I blocks: [0,1], [2,3]
    # K blocks: [0,1], [2,3], [4,5]
    for i_off in [0, 2]:
        for k_off in [0, 2, 4]:
            # J1: [0,1]
            col += add_strassen_2x2(i_off, 0, k_off, col)
            # J2: [2,3]
            col += add_strassen_2x2(i_off, 2, k_off, col)
            # J3: [4]
            col += add_outer_2x1x2(i_off, 4, k_off, col)
    
    # Verify correctness
    def verify(U, V, W):
        """Verify that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if s != 1:
                        return False
        return True
    
    if verify(U_struct, V_struct, W_struct):
        best_factors = (U_struct, V_struct, W_struct)
        best_rank = rank_structured
    else:
        # Fall back to schoolbook
        pass
    
    U, V, W = best_factors
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
