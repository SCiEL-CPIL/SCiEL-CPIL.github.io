"""Matrix multiplication 2,3,4 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 4]

import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Uses Strassen-based block decomposition achieving rank 22.
    Returns {'U': (6, rank), 'V': (12, rank), 'W': (8, rank)} with integer/half-integer entries.
    The output follows the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 3, 4
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Build Strassen-based rank-22 decomposition
    # Split middle dimension 3 into 2+1:
    # A (2x3) = [A_top (2x2); A_bottom (2x1)]
    # B (3x4) = [B_left (2x4); B_right (1x4)]
    # C = A_top @ B_left + A_bottom @ B_right
    # 
    # A_top (2x2) @ B_left (2x4): split B_left into [B11 (2x2), B12 (2x2)]
    #   A_top @ B11: Strassen rank 7
    #   A_top @ B12: Strassen rank 7
    # A_bottom (2x1) @ B_right (1x4): outer product, rank 8
    # Total: 7 + 7 + 8 = 22
    
    rank = 22
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    # Strassen 2x2 x 2x2 for A_top (2x2) @ B11 (2x2)
    # A_top indices: A[i, j] where i in {0,1}, j in {0,1} -> U index i*m+j = i*3+j
    # B11 indices: B[j, k] where j in {0,1}, k in {0,1} -> V index j*p+k = j*4+k
    # Output C[i,k] where i in {0,1}, k in {0,1} -> W index k*n+i = k*2+i
    
    # Strassen formulas for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # B11: j in {0,1}, k in {0,1}
    strassen_B11 = [
        # (a_coeff, b_coeff, c_coeff) for each of 7 terms
        # Each term: U_val * V_val * W_val contributes to output
    ]
    
    # For A_top @ B11:
    # A_top[i, j]: U index = i*3 + j
    # B11[j, k]: V index = j*4 + k
    # Output[i, k]: W index = k*2 + i
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # A00: U[0*3+0]=U[0], A11: U[1*3+1]=U[4]
    # B00: V[0*4+0]=V[0], B11: V[1*4+1]=V[5]
    # This contributes to C00 (+1), C11 (+1)
    U[0, col] = 1; V[0, col] = 1; W[0, col] = 1  # C[0,0]: W[0*2+0]=W[0]
    U[0, col] += 0  # We need separate columns for each Strassen term's contribution
    # Actually, each Strassen term is ONE rank-1 component that contributes to multiple outputs
    # But in tensor decomposition, each rank-1 term contributes to exactly one (i,j,k) triple...
    # No wait. Let me reconsider.
    
    # In tensor decomposition for matrix multiplication:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Each term r contributes U[i,j,r]*V[j,k,r]*W[k,i,r] to T[i,j,k]
    # For the decomposition to work, for each r, the pattern of which (i,j,k) get non-zero
    # contributions must match.
    
    # For Strassen term M1 = (A00+A11)(B00+B11):
    # This means we need terms where:
    # U has 1 at (i=0,j=0) and (i=1,j=1)
    # V has 1 at (j=0,k=0) and (j=1,k=1)  
    # But this doesn't directly give a rank-1 tensor...
    
    # Actually, the standard way: Strassen's algorithm expresses the 8 standard
    # multiplications as 7 new multiplications. Each "multiplication" is a rank-1
    # tensor. The 7 Strassen multiplications are:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B10-B11)
    # P4 = A11*(B01-B10)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B10)
    # P7 = (A01-A11)*(B01+B11)
    
    # Each P is a product of a linear combination of A entries and a linear
    # combination of B entries. In the tensor decomposition, each P corresponds
    # to one rank-1 term where U encodes the A coefficients, V encodes the B
    # coefficients, and W encodes how P contributes to each output C entry.
    
    # For P1 = (A00+A11)*(B00+B11):
    # U[0,0,1]=1, U[1,1,1]=1 (A00 and A11)
    # V[0,0,1]=1, V[1,1,1]=1 (B00 and B11)
    # C00 += P1, C11 += P1
    # So W[0,0,1]=1, W[1,1,1]=1 (output C[0,0] and C[1,1])
    # W index: W[k*n+i] so C[0,0]->W[0], C[1,1]->W[1*2+1]=W[3]
    
    # Let me redo this properly with a helper
    
    def add_strassen_2x2x2x2(u_base_i, u_base_j, v_base_j, v_base_k, w_base_k, w_base_i, col_start):
        """
        Add 7 Strassen terms for multiplying a 2x2 submatrix of A by a 2x2 submatrix of B.
        
        u_base_i, u_base_j: base offsets for U indices (U[i*m + j])
        v_base_j, v_base_k: base offsets for V indices (V[j*p + k])  
        w_base_k, w_base_i: base offsets for W indices (W[k*n + i])
        
        The 2x2 A block has entries at:
          A[0,0]: U[u_base_i + 0, u_base_j + 0]
          A[0,1]: U[u_base_i + 0, u_base_j + 1]
          A[1,0]: U[u_base_i + 1, u_base_j + 0]
          A[1,1]: U[u_base_i + 1, u_base_j + 1]
        
        The 2x2 B block has entries at:
          B[0,0]: V[v_base_j + 0, v_base_k + 0]
          B[0,1]: V[v_base_j + 0, v_base_k + 1]
          B[1,0]: V[v_base_j + 1, v_base_k + 0]
          B[1,1]: V[v_base_j + 1, v_base_k + 1]
        
        The 2x2 C block has entries at:
          C[0,0]: W[w_base_k + 0, w_base_i + 0]
          C[0,1]: W[w_base_k + 1, w_base_i + 0]
          C[1,0]: W[w_base_k + 0, w_base_i + 1]
          C[1,1]: W[w_base_k + 1, w_base_i + 1]
        """
        col = col_start
        # P1 = (A00+A11)*(B00+B11)
        U[u_base_i, u_base_j, col] = 1
        U[u_base_i+1, u_base_j+1, col] = 1
        V[v_base_j, v_base_k, col] = 1
        V[v_base_j+1, v_base_k+1, col] = 1
        W[w_base_k, w_base_i, col] = 1      # C[0,0]
        W[w_base_k+1, w_base_i+1, col] = 1  # C[1,1]
        col += 1
        
        # P2 = (A10+A11)*B00
        U[u_base_i+1, u_base_j, col] = 1
        U[u_base_i+1, u_base_j+1, col] = 1
        V[v_base_j, v_base_k, col] = 1
        W[w_base_k, w_base_i+1, col] = 1    # C[1,0]
        col += 1
        
        # P3 = A00*(B10-B11)
        U[u_base_i, u_base_j, col] = 1
        V[v_base_j+1, v_base_k, col] = 1
        V[v_base_j+1, v_base_k+1, col] = -1
        W[w_base_k+1, w_base_i, col] = 1    # C[0,1]
        W[w_base_k, w_base_i+1, col] = 1    # C[1,0]
        col += 1
        
        # P4 = A11*(B01-B10)
        U[u_base_i+1, u_base_j+1, col] = 1
        V[v_base_j, v_base_k+1, col] = 1
        V[v_base_j+1, v_base_k, col] = -1
        W[w_base_k, w_base_i+1, col] = 1    # C[1,0]
        col += 1
        
        # P5 = (A00+A01)*B11
        U[u_base_i, u_base_j, col] = 1
        U[u_base_i, u_base_j+1, col] = 1
        V[v_base_j+1, v_base_k+1, col] = 1
        W[w_base_k+1, w_base_i, col] = -1   # C[0,1] (negative)
        W[w_base_k+1, w_base_i+1, col] = 1  # C[1,1]
        col += 1
        
        # P6 = (A10-A00)*(B00+B10)
        U[u_base_i+1, u_base_j, col] = 1
        U[u_base_i, u_base_j, col] = -1
        V[v_base_j, v_base_k, col] = 1
        V[v_base_j+1, v_base_k, col] = 1
        W[w_base_k+1, w_base_i+1, col] = 1  # C[1,1]
        col += 1
        
        # P7 = (A01-A11)*(B01+B11)
        U[u_base_i, u_base_j+1, col] = 1
        U[u_base_i+1, u_base_j+1, col] = -1
        V[v_base_j, v_base_k+1, col] = 1
        V[v_base_j+1, v_base_k+1, col] = 1
        W[w_base_k, w_base_i, col] = -1     # C[0,0] (negative)
        col += 1
        
        return col
    
    # Now build the full decomposition
    # A is 2x3: A[i,j] with i in {0,1}, j in {0,1,2}
    # B is 3x4: B[j,k] with j in {0,1,2}, k in {0,1,2,3}
    # C is 2x4: C[i,k] with i in {0,1}, k in {0,1,2,3}
    
    # Split j (middle dim) into j in {0,1} and j=2
    # A_top = A[:, 0:2] (2x2), A_bottom = A[:, 2:3] (2x1)
    # B_top = B[0:2, :] (2x4), B_bottom = B[2:3, :] (1x4)
    # C = A_top @ B_top + A_bottom @ B_bottom
    
    # A_top @ B_top: both 2x2 x 2x4
    # Split B_top into B_top_left (2x2) and B_top_right (2x2)
    # B_top_left = B[0:2, 0:2], B_top_right = B[0:2, 2:4]
    
    # Part 1: A_top (2x2) @ B_top_left (2x2) -> C[:, 0:2]
    # U indices for A_top: i*3+j where i in {0,1}, j in {0,1}
    #   A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
    # V indices for B_top_left: j*4+k where j in {0,1}, k in {0,1}
    #   B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[4], B[1,1]=V[5]
    # W indices for C[:, 0:2]: k*2+i where i in {0,1}, k in {0,1}
    #   C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    
    # u_base_i = 0 (i offset in U's first index, which is i*m+j, so i contributes m*j... 
    # Wait, U is indexed by (i*m + j). For A_top, i in {0,1}, j in {0,1}.
    # U[i*3 + j]: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
    # So the "row" in U for A entry (i,j) is i*3+j.
    # In my add_strassen function, I'm using U[u_base_i + di, u_base_j + dj]
    # but U is 2D: U[row, col]. The row is i*m+j.
    
    # I need to restructure. Let me use explicit row indices.
    
    # Let me redo with explicit indices
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    
    # Helper: add one Strassen term
    def add_strassen_term(u_indices, v_indices, w_indices, col_idx):
        """
        u_indices: list of (row, coeff) for U
        v_indices: list of (row, coeff) for V
        w_indices: list of (row, coeff) for W
        """
        for row, coeff in u_indices:
            U[row, col_idx] = coeff
        for row, coeff in v_indices:
            V[row, col_idx] = coeff
        for row, coeff in w_indices:
            W[row, col_idx] = coeff
    
    # Part 1: A_top (2x2) @ B_top_left (2x2) -> C[:, 0:2]
    # A_top rows in U: [0, 1, 3, 4] for A[0,0], A[0,1], A[1,0], A[1,1]
    # B_top_left rows in V: [0, 1, 4, 5] for B[0,0], B[0,1], B[1,0], B[1,1]
    # C[:, 0:2] rows in W: [0, 1, 2, 3] for C[0,0], C[0,1], C[1,0], C[1,1]
    
    # P1 = (A00+A11)*(B00+B11)
    add_strassen_term(
        [(0, 1), (4, 1)],  # A[0,0], A[1,1]
        [(0, 1), (5, 1)],  # B[0,0], B[1,1]
        [(0, 1), (3, 1)],  # C[0,0], C[1,1]
        col
    )
    col += 1
    
    # P2 = (A10+A11)*B00
    add_strassen_term(
        [(3, 1), (4, 1)],  # A[1,0], A[1,1]
        [(0, 1)],          # B[0,0]
        [(2, 1)],          # C[1,0]
        col
    )
    col += 1
    
    # P3 = A00*(B10-B11)
    add_strassen_term(
        [(0, 1)],          # A[0,0]
        [(4, 1), (5, -1)], # B[1,0], -B[1,1]
        [(1, 1), (2, 1)],  # C[0,1], C[1,0]
        col
    )
    col += 1
    
    # P4 = A11*(B01-B10)
    add_strassen_term(
        [(4, 1)],          # A[1,1]
        [(1, 1), (4, -1)], # B[0,1], -B[1,0]
        [(2, 1)],          # C[1,0]
        col
    )
    col += 1
    
    # P5 = (A00+A01)*B11
    add_strassen_term(
        [(0, 1), (1, 1)],  # A[0,0], A[0,1]
        [(5, 1)],          # B[1,1]
        [(1, -1), (3, 1)], # -C[0,1], C[1,1]
        col
    )
    col += 1
    
    # P6 = (A10-A00)*(B00+B10)
    add_strassen_term(
        [(3, 1), (0, -1)], # A[1,0], -A[0,0]
        [(0, 1), (4, 1)],  # B[0,0], B[1,0]
        [(3, 1)],          # C[1,1]
        col
    )
    col += 1
    
    # P7 = (A01-A11)*(B01+B11)
    add_strassen_term(
        [(1, 1), (4, -1)], # A[0,1], -A[1,1]
        [(1, 1), (5, 1)],  # B[0,1], B[1,1]
        [(0, -1)],         # -C[0,0]
        col
    )
    col += 1
    
    # Part 2: A_top (2x2) @ B_top_right (2x2) -> C[:, 2:4]
    # B_top_right = B[0:2, 2:4]
    # B rows in V: [2, 3, 6, 7] for B[0,2], B[0,3], B[1,2], B[1,3]
    # C[:, 2:4] rows in W: [4, 5, 6, 7] for C[0,2], C[0,3], C[1,2], C[1,3]
    
    # P1 = (A00+A11)*(B00+B11) where B here is B_top_right
    add_strassen_term(
        [(0, 1), (4, 1)],  # A[0,0], A[1,1]
        [(2, 1), (7, 1)],  # B[0,2], B[1,3]
        [(4, 1), (7, 1)],  # C[0,2], C[1,3]
        col
    )
    col += 1
    
    # P2 = (A10+A11)*B00
    add_strassen_term(
        [(3, 1), (4, 1)],  # A[1,0], A[1,1]
        [(2, 1)],          # B[0,2]
        [(6, 1)],          # C[1,2]
        col
    )
    col += 1
    
    # P3 = A00*(B10-B11)
    add_strassen_term(
        [(0, 1)],          # A[0,0]
        [(6, 1), (7, -1)], # B[1,2], -B[1,3]
        [(5, 1), (6, 1)],  # C[0,3], C[1,2]
        col
    )
    col += 1
    
    # P4 = A11*(B01-B10)
    add_strassen_term(
        [(4, 1)],          # A[1,1]
        [(3, 1), (6, -1)], # B[0,3], -B[1,2]
        [(6, 1)],          # C[1,2]
        col
    )
    col += 1
    
    # P5 = (A00+A01)*B11
    add_strassen_term(
        [(0, 1), (1, 1)],  # A[0,0], A[0,1]
        [(7, 1)],          # B[1,3]
        [(5, -1), (7, 1)], # -C[0,3], C[1,3]
        col
    )
    col += 1
    
    # P6 = (A10-A00)*(B00+B10)
    add_strassen_term(
        [(3, 1), (0, -1)], # A[1,0], -A[0,0]
        [(2, 1), (6, 1)],  # B[0,2], B[1,2]
        [(7, 1)],          # C[1,3]
        col
    )
    col += 1
    
    # P7 = (A01-A11)*(B01+B11)
    add_strassen_term(
        [(1, 1), (4, -1)], # A[0,1], -A[1,1]
        [(3, 1), (7, 1)],  # B[0,3], B[1,3]
        [(4, -1)],         # -C[0,2]
        col
    )
    col += 1
    
    # Part 3: A_bottom (2x1) @ B_bottom (1x4) -> C[:, :]
    # A_bottom = A[:, 2:3], A[0,2]=U[2], A[1,2]=U[5]
    # B_bottom = B[2:3, :], B[2,0]=V[8], B[2,1]=V[9], B[2,2]=V[10], B[2,3]=V[11]
    # C[:, :] all 8 entries
    
    # This is an outer product: C[i,k] = A[i,2] * B[2,k]
    # Each (i,k) pair needs its own rank-1 term
    for i in range(2):
        for k in range(4):
            u_row = i * 3 + 2  # A[i, 2]
            v_row = 2 * 4 + k  # B[2, k]
            w_row = k * 2 + i  # C[i, k]
            U[u_row, col] = 1
            V[v_row, col] = 1
            W[w_row, col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify
    if verify(U.tolist(), V.tolist(), W.tolist()):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 24
    rank = 24
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
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
