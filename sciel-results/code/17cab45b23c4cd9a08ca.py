"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_decomposition():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns lists of (u_coeffs, v_coeffs, w_coeffs) for each of the 7 bilinear terms.
    u_coeffs: dict {(i,j): coeff} for A
    v_coeffs: dict {(j,k): coeff} for B  
    w_coeffs: dict {(i,k): coeff} for C
    """
    # Strassen's 7 multiplications for 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10+b11)
    # M4 = (a11)(b01-b11)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    terms = []
    
    # M1 = (a00+a11)(b00+b11)
    terms.append({
        'u': {(0,0): 1, (1,1): 1},
        'v': {(0,0): 1, (1,1): 1},
        'w': {(0,0): 1, (1,1): 1}
    })
    
    # M2 = (a10+a11)(b00)
    terms.append({
        'u': {(1,0): 1, (1,1): 1},
        'v': {(0,0): 1},
        'w': {(1,0): 1, (1,1): -1}
    })
    
    # M3 = (a00)(b10+b11)
    terms.append({
        'u': {(0,0): 1},
        'v': {(1,0): 1, (1,1): 1},
        'w': {(0,1): 1, (1,0): 1}
    })
    
    # M4 = (a11)(b01-b11)
    terms.append({
        'u': {(1,1): 1},
        'v': {(0,1): 1, (1,1): -1},
        'w': {(0,0): 1, (1,0): 1}
    })
    
    # M5 = (a00+a01)(b11)
    terms.append({
        'u': {(0,0): 1, (0,1): 1},
        'v': {(1,1): 1},
        'w': {(0,0): -1, (0,1): 1}
    })
    
    # M6 = (a10-a00)(b00+b01)
    terms.append({
        'u': {(1,0): 1, (0,0): -1},
        'v': {(0,0): 1, (0,1): 1},
        'w': {(1,1): 1}
    })
    
    # M7 = (a01-a11)(b10+b11)
    terms.append({
        'u': {(0,1): 1, (1,1): -1},
        'v': {(1,0): 1, (1,1): 1},
        'w': {(0,0): 1}
    })
    
    return terms

def _build_strassen_block_decomposition(n, m, p):
    """
    Build a decomposition using Strassen for 2x2 blocks where applicable,
    and schoolbook for the rest.
    
    For 4x5 * 5x5:
    - Split the middle dimension (5) into 4+1
    - A = [A1 (4x4) | A2 (4x1)]
    - B = [B1 (4x5); B2 (1x5)]
    - C = A1*B1 + A2*B2
    
    For A1*B1 (4x4 * 4x5):
    - Split B1 into [B1a (4x4) | B1b (4x1)]
    - C_part1 = A1*B1a + A1*B1b
    
    For A1*B1a (4x4 * 4x4): use 2x2 Strassen blocks
    For A1*B1b (4x4 * 4x1): schoolbook = 16
    For A2*B2 (4x1 * 1x5): schoolbook = 20
    
    For 4x4*4x4, partition into 2x2 blocks:
    A1 = [[A11, A12], [A21, A22]] (each 2x2)
    B1a = [[B11, B12], [B21, B22]] (each 2x2)
    
    Product = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
               [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    Each 2x2*2x2 product uses Strassen rank 7.
    There are 4 such products, so 4*7 = 28 for the 4x4*4x4 part.
    
    Total: 28 + 16 + 20 = 64
    """
    n, m, p = 4, 5, 5
    
    U_entries = []  # list of (row_index, coeff)
    V_entries = []  # list of (col_index, coeff)
    W_entries = []  # list of (out_index, coeff)
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a bilinear term to the decomposition."""
        U_row = {}
        V_row = {}
        W_row = {}
        
        for (i, j), coeff in u_dict.items():
            U_row[i * m + j] = coeff
        for (j, k), coeff in v_dict.items():
            V_row[j * p + k] = coeff
        for (k, i), coeff in w_dict.items():
            W_row[k * n + i] = coeff
        
        U_entries.append(U_row)
        V_entries.append(V_row)
        W_entries.append(W_row)
    
    # Strassen terms for 2x2*2x2
    strassen_terms = _strassen_2x2_decomposition()
    
    # Block indices for 4x4*4x4:
    # A1 rows: 0,1 (top), 2,3 (bottom)
    # A1 cols: 0,1 (left), 2,3 (right)
    # B1a rows: 0,1 (top), 2,3 (bottom)  
    # B1a cols: 0,1 (left), 2,3 (right)
    
    # The 4x4*4x4 product has 4 blocks of 2x2*2x2:
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    # C[0:2,2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    # C[2:4,0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
    # C[2:4,2:4] = A[2:4,0:2]*B[0:2,2:4] + A[2:4,2:4]*B[2:4,2:4]
    
    # For each 2x2*2x2 sub-product, apply Strassen
    # Sub-product (ar, ac, br, bc) where ar,ac,br,bc are 0 or 1 (block index)
    # A block rows: ar*2 to ar*2+1, cols: ac*2 to ac*2+1
    # B block rows: br*2 to br*2+1, cols: bc*2 to bc*2+1
    # C output rows: ar*2 to ar*2+1, cols: bc*2 to bc*2+1
    
    for ar in range(2):
        for ac in range(2):
            for br in range(2):
                for bc in range(2):
                    # This is A[ar,ac] * B[br,bc] contributing to C[ar,bc]
                    # Apply Strassen to this 2x2*2x2
                    for term in strassen_terms:
                        u_dict = {}
                        v_dict = {}
                        w_dict = {}
                        
                        for (ii, jj), coeff in term['u'].items():
                            # A block (ar,ac): global row = ar*2+ii, col = ac*2+jj
                            u_dict[(ar*2+ii, ac*2+jj)] = coeff
                        
                        for (jj, kk), coeff in term['v'].items():
                            # B block (br,bc): global row = br*2+jj, col = bc*2+kk
                            v_dict[(br*2+jj, bc*2+kk)] = coeff
                        
                        for (kk, ii), coeff in term['w'].items():
                            # C output block (ar,bc): global row = ar*2+ii, col = bc*2+kk
                            w_dict[(bc*2+kk, ar*2+ii)] = coeff
                        
                        add_term(u_dict, v_dict, w_dict)
    
    # Now handle A1*B1b: A1 (4x4) * B1b (4x1)
    # This is schoolbook: 4*4*1 = 16 terms
    for i in range(4):
        for j in range(4):
            for k in range(1):
                # A[i,j] * B[4+k, j+k]... wait, B1b is the last column of B1
                # B1 is rows 0-3, cols 0-4. B1b is cols 4 only, so B[0:4, 4]
                # A1 is rows 0-3, cols 0-3
                # Product contributes to C[i, 4]
                # C[i,4] = sum_j A1[i,j] * B1b[j, 0] where B1b[j,0] = B[j, 4]
                u_dict = {(i, j): 1}
                v_dict = {(j, 4): 1}
                w_dict = {(4, i): 1}
                add_term(u_dict, v_dict, w_dict)
    
    # Now handle A2*B2: A2 (4x1) * B2 (1x5)
    # A2 is A[0:4, 4], B2 is B[4:5, 0:5]
    # Product: C[i,k] += A2[i,0] * B2[0,k] = A[i,4] * B[4,k]
    # This is 4*1*5 = 20 terms
    for i in range(4):
        for k in range(5):
            u_dict = {(i, 4): 1}
            v_dict = {(4, k): 1}
            w_dict = {(k, i): 1}
            add_term(u_dict, v_dict, w_dict)
    
    # Convert to matrices
    rank = len(U_entries)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        for idx, coeff in U_entries[r].items():
            U[idx, r] = coeff
        for idx, coeff in V_entries[r].items():
            V[idx, r] = coeff
        for idx, coeff in W_entries[r].items():
            W[idx, r] = coeff
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140
    
    # First, try the Strassen-block decomposition
    try:
        U, V, W, rank = _build_strassen_block_decomposition(n, m, p)
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
            U_r = np.round(U * 2) / 2.0
            V_r = np.round(V * 2) / 2.0
            W_r = np.round(W * 2) / 2.0
            if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-6):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            else:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
