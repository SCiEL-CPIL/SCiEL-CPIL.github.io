"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit

    def verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
        rank = U.shape[1]
        # Check all valid (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def strassen_2x2x2():
        """Return U, V, W for rank-7 decomposition of 2x2x2 matrix multiplication.
        
        Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        where i,k are output indices (0,1) and j is the contraction index (0,1)
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00: U[0,0]=1, A11: U[3,0]=1
        # B00: V[0,0]=1, B11: V[3,0]=1
        # C00: W[0,0]=1, C01: W[1,0]=1, C10: W[2,0]=1, C11: W[3,0]=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10 + A11)B00
        # A10: U[2,1]=1, A11: U[3,1]=1
        # B00: V[0,1]=1
        # C10: W[2,1]=1, C11: W[3,1]=1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A00(B10 - B11)
        # A00: U[0,2]=1
        # B10: V[2,2]=1, B11: V[3,2]=-1
        # C00: W[0,2]=1, C01: W[1,2]=1
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[1, 2] = 1
        
        # M4 = A11(B00 - B01)
        # A11: U[3,3]=1
        # B00: V[0,3]=1, B01: V[1,3]=-1
        # C10: W[2,3]=1, C11: W[3,3]=1
        U[3, 3] = 1
        V[0, 3] = 1; V[1, 3] = -1
        W[2, 3] = 1; W[3, 3] = 1
        
        # M5 = (A00 + A01)B11
        # A00: U[0,4]=1, A01: U[1,4]=1
        # B11: V[3,4]=1
        # C00: W[0,4]=1, C01: W[1,4]=1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[1, 4] = 1
        
        # M6 = (A10 - A00)(B00 + B01)
        # A10: U[2,5]=1, A00: U[0,5]=-1
        # B00: V[0,5]=1, B01: V[1,5]=1
        # C10: W[2,5]=1, C11: W[3,5]=1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[3, 5] = 1
        
        # M7 = (A01 - A11)(B10 + B11)
        # A01: U[1,6]=1, A11: U[3,6]=-1
        # B10: V[2,6]=1, B11: V[3,6]=1
        # C00: W[0,6]=1, C01: W[1,6]=1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[1, 6] = 1
        
        return U, V, W

    def build_rank98():
        """Construct a rank-98 decomposition for 4x5x5 using Strassen on 2x2x2 blocks.
        
        Block structure:
        A (4x5) split as:
          A = [[A11 (2x2), A12 (2x3)],
               [A21 (2x2), A22 (2x3)]]
        B (5x5) split as:
          B = [[B11 (2x2), B12 (2x3)],
               [B21 (3x2), B22 (3x3)]]
        C (4x5) split as:
          C = [[C11 (2x2), C12 (2x3)],
               [C21 (2x2), C22 (2x3)]]
        
        C11 = A11*B11 + A12*B21  (2x2)
        C12 = A11*B12 + A12*B22  (2x3)
        C21 = A21*B11 + A22*B21  (2x2)
        C22 = A21*B12 + A22*B22  (2x3)
        
        For C11: use Strassen on A11*B11 (rank 7) + schoolbook on A12*B21 (rank 6) = 13
        For C12: use Strassen on A11*B12_col0 (rank 7) + schoolbook on A11*B12_col1..2 (rank 6) 
                 + schoolbook on A12*B22 (rank 18) = 31... this is getting complex.
        
        Let me use a cleaner approach:
        - For each 2x2x2 sub-problem, use Strassen (rank 7)
        - For each 2x2x3 or 2x3x2 sub-problem, use schoolbook
        - For 2x3x3, use schoolbook
        
        Total:
        C11: A11(2x2)*B11(2x2) via Strassen = 7, A12(2x3)*B21(3x2) via schoolbook = 6, total = 13
        C12: A11(2x2)*B12(2x3) via schoolbook = 6, A12(2x3)*B22(3x3) via schoolbook = 18, total = 24
        C21: A21(2x2)*B11(2x2) via Strassen = 7, A22(2x3)*B21(3x2) via schoolbook = 6, total = 13
        C22: A21(2x2)*B12(2x3) via schoolbook = 6, A22(2x3)*B22(3x3) via schoolbook = 18, total = 24
        
        Grand total: 13 + 24 + 13 + 24 = 74... that's way below 98!
        
        Wait, I need to be more careful. Let me think about what "schoolbook" means here.
        
        For A12(2x3) * B21(3x2) = C12_part (2x2):
        C12_part[i,k] = sum_j A12[i,j] * B21[j,k] for j=0,1,2
        This is 2*3*2 = 12 scalar products in schoolbook.
        
        For A11(2x2) * B12(2x3) = C11_part (2x3):
        C11_part[i,k] = sum_j A11[i,j] * B12[j,k] for j=0,1
        This is 2*2*3 = 12 scalar products in schoolbook.
        
        For A12(2x3) * B22(3x3) = C12_part2 (2x3):
        C12_part2[i,k] = sum_j A12[i,j] * B22[j,k] for j=0,1,2
        This is 2*3*3 = 18 scalar products in schoolbook.
        
        So:
        C11: Strassen(2x2x2) + schoolbook(2x3x2) = 7 + 12 = 19
        C12: schoolbook(2x2x3) + schoolbook(2x3x3) = 12 + 18 = 30
        C21: Strassen(2x2x2) + schoolbook(2x3x2) = 7 + 12 = 19
        C22: schoolbook(2x2x3) + schoolbook(2x3x3) = 12 + 18 = 30
        
        Grand total: 19 + 30 + 19 + 30 = 98! Perfect!
        """
        U = np.zeros((n*m, 98))
        V = np.zeros((m*p, 98))
        W = np.zeros((p*n, 98))
        
        col_offset = 0
        
        # Block A11: rows 0-1, cols 0-1 of A
        # Block A12: rows 0-1, cols 2-4 of A
        # Block A21: rows 2-3, cols 0-1 of A
        # Block A22: rows 2-3, cols 2-4 of A
        
        # Block B11: rows 0-1, cols 0-1 of B
        # Block B12: rows 0-1, cols 2-4 of B
        # Block B21: rows 2-4, cols 0-1 of B
        # Block B22: rows 2-4, cols 2-4 of B
        
        # Block C11: rows 0-1, cols 0-1 of C
        # Block C12: rows 0-1, cols 2-4 of C
        # Block C21: rows 2-3, cols 0-1 of C
        # Block C22: rows 2-3, cols 2-4 of C
        
        # Helper: add Strassen decomposition for a 2x2x2 block
        def add_strassen(u_rows, v_rows, w_rows, col_start):
            """Add Strassen 2x2x2 decomposition starting at col_start.
            u_rows: list of 4 U row indices (for A block)
            v_rows: list of 4 V row indices (for B block)
            w_rows: list of 4 W row indices (for C block)
            Returns new col offset.
            """
            sU, sV, sW = strassen_2x2x2()
            for c in range(7):
                for r in range(4):
                    if sU[r, c] != 0:
                        U[u_rows[r], col_start + c] += sU[r, c]
                    if sV[r, c] != 0:
                        V[v_rows[r], col_start + c] += sV[r, c]
                    if sW[r, c] != 0:
                        W[w_rows[r], col_start + c] += sW[r, c]
            return col_start + 7
        
        # Helper: add schoolbook decomposition for a small block multiplication
        def add_schoolbook_block(u_row_indices, v_row_indices, w_row_indices, col_start):
            """Add schoolbook decomposition for a block multiplication.
            u_row_indices: list of U row indices (len = n_block * m_block)
            v_row_indices: list of V row indices (len = m_block * p_block)
            w_row_indices: list of W row indices (len = p_block * n_block)
            col_start: starting column
            Returns new col offset.
            """
            n_b = len(u_row_indices) // len(v_row_indices)  # This isn't quite right
            # Let me use a different approach: pass the block dimensions
            # Actually, let me just handle specific cases directly
            pass
        
        # Let me handle each block multiplication explicitly
        
        # C11 = A11*B11 (2x2x2, Strassen) + A12*B21 (2x3x2, schoolbook)
        
        # C11 part 1: A11(2x2) * B11(2x2) via Strassen
        # A11 rows in U: [0,1,2,3] (i*5+j for i=0,1 and j=0,1)
        # B11 rows in V: [0,1,5,6] (j*5+k for j=0,1 and k=0,1)
        # C11 rows in W: [0,1,4,5] (k*4+i for k=0,1 and i=0,1)
        a11_u = [0, 1, 2, 3]  # i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1
        b11_v = [0, 1, 5, 6]  # j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1
        c11_w = [0, 1, 4, 5]  # k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1
        col_offset = add_strassen(a11_u, b11_v, c11_w, col_offset)
        
        # C11 part 2: A12(2x3) * B21(3x2) via schoolbook
        # A12: i=0,1; j=2,3,4 -> U rows: [2,3,4, 7,8,9]
        # B21: j=2,3,4; k=0,1 -> V rows: [10,11, 15,16, 20,21]
        # C11: k=0,1; i=0,1 -> W rows: [0,1, 4,5]
        a12_u = [2, 3, 4, 7, 8, 9]  # (i,j): (0,2),(0,3),(0,4),(1,2),(1,3),(1,4)
        b21_v = [10, 11, 15, 16, 20, 21]  # (j,k): (2,0),(2,1),(3,0),(3,1),(4,0),(4,1)
        c11_w = [0, 1, 4, 5]  # (k,i): (0,0),(0,1),(1,0),(1,1)
        
        # Schoolbook for 2x3x2: C[i,k] = sum_j A[i,j]*B[j,k]
        # i in {0,1}, j in {0,1,2}, k in {0,1}
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U[a12_u[i*3+j], col_offset] = 1
                    V[b21_v[j*2+k], col_offset] = 1
                    W[c11_w[k*2+i], col_offset] = 1
                    col_offset += 1
        
        # C12 = A11*B12 (2x2x3, schoolbook) + A12*B22 (2x3x3, schoolbook)
        
        # C12 part 1: A11(2x2) * B12(2x3) via schoolbook
        # A11: i=0,1; j=0,1 -> U rows: [0,1,2,3]
        # B12: j=0,1; k=2,3,4 -> V rows: [2,3,4, 7,8,9]
        # C12: k=2,3,4; i=0,1 -> W rows: [2,3,4, 6,7,8]
        a11_u = [0, 1, 2, 3]
        b12_v = [2, 3, 4, 7, 8, 9]  # (j,k): (0,2),(0,3),(0,4),(1,2),(1,3),(1,4)
        c12_w = [2, 3, 4, 6, 7, 8]  # (k,i): (2,0),(2,1),(2,2),(3,0),(3,1),(3,2)
        
        # Schoolbook for 2x2x3: C[i,k] = sum_j A[i,j]*B[j,k]
        # i in {0,1}, j in {0,1}, k in {0,1,2}
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[a11_u[i*2+j], col_offset] = 1
                    V[b12_v[j*3+k], col_offset] = 1
                    W[c12_w[k*2+i], col_offset] = 1
                    col_offset += 1
        
        # C12 part 2: A12(2x3) * B22(3x3) via schoolbook
        # A12: i=0,1; j=2,3,4 -> U rows: [2,3,4, 7,8,9]
        # B22: j=2,3,4; k=2,3,4 -> V rows: [12,13,14, 17,18,19, 22,23,24]
        # C12: k=2,3,4; i=0,1 -> W rows: [2,3,4, 6,7,8]
        a12_u = [2, 3, 4, 7, 8, 9]
        b22_v = [12, 13, 14, 17, 18, 19, 22, 23, 24]  # (j,k): (2,2)..(2,4),(3,2)..(3,4),(4,2)..(4,4)
        c12_w = [2, 3, 4, 6, 7, 8]
        
        # Schoolbook for 2x3x3: C[i,k] = sum_j A[i,j]*B[j,k]
        # i in {0,1}, j in {0,1,2}, k in {0,1,2}
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U[a12_u[i*3+j], col_offset] = 1
                    V[b22_v[j*3+k], col_offset] = 1
                    W[c12_w[k*2+i], col_offset] = 1
                    col_offset += 1
        
        # C21 = A21*B11 (2x2x2, Strassen) + A22*B21 (2x3x2, schoolbook)
        
        # C21 part 1: A21(2x2) * B11(2x2) via Strassen
        # A21: i=2,3; j=0,1 -> U rows: [10,11,12,13]
        # B11: j=0,1; k=0,1 -> V rows: [0,1,5,6]
        # C21: k=0,1; i=2,3 -> W rows: [8,9,12,13]
        a21_u = [10, 11, 12, 13]  # (i,j): (2,0),(2,1),(3,0),(3,1)
        b11_v = [0, 1, 5, 6]
        c21_w = [8, 9, 12, 13]  # (k,i): (0,2),(0,3),(1,2),(1,3)
        col_offset = add_strassen(a21_u, b11_v, c21_w, col_offset)
        
        # C21 part 2: A22(2x3) * B21(3x2) via schoolbook
        # A22: i=2,3; j=2,3,4 -> U rows: [12,13,14, 17,18,19]
        # B21: j=2,3,4; k=0,1 -> V rows: [10,11, 15,16, 20,21]
        # C21: k=0,1; i=2,3 -> W rows: [8,9, 12,13]
        a22_u = [12, 13, 14, 17, 18, 19]
        b21_v = [10, 11, 15, 16, 20, 21]
        c21_w = [8, 9, 12, 13]
        
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U[a22_u[i*3+j], col_offset] = 1
                    V[b21_v[j*2+k], col_offset] = 1
                    W[c21_w[k*2+i], col_offset] = 1
                    col_offset += 1
        
        # C22 = A21*B12 (2x2x3, schoolbook) + A22*B22 (2x3x3, schoolbook)
        
        # C22 part 1: A21(2x2) * B12(2x3) via schoolbook
        # A21: i=2,3; j=0,1 -> U rows: [10,11,12,13]
        # B12: j=0,1; k=2,3,4 -> V rows: [2,3,4, 7,8,9]
        # C22: k=2,3,4; i=2,3 -> W rows: [10,11,12, 14,15,16]
        a21_u = [10, 11, 12, 13]
        b12_v = [2, 3, 4, 7, 8, 9]
        c22_w = [10, 11, 12, 14, 15, 16]  # (k,i): (2,2),(2,3),(2,4),(3,2),(3,3),(3,4)
        
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[a21_u[i*2+j], col_offset] = 1
                    V[b12_v[j*3+k], col_offset] = 1
                    W[c22_w[k*2+i], col_offset] = 1
                    col_offset += 1
        
        # C22 part 2: A22(2x3) * B22(3x3) via schoolbook
        # A22: i=2,3; j=2,3,4 -> U rows: [12,13,14, 17,18,19]
        # B22: j=2,3,4; k=2,3,4 -> V rows: [12,13,14, 17,18,19, 22,23,24]
        # C22: k=2,3,4; i=2,3 -> W rows: [10,11,12, 14,15,16]
        a22_u = [12, 13, 14, 17, 18, 19]
        b22_v = [12, 13, 14, 17, 18, 19, 22, 23, 24]
        c22_w = [10, 11, 12, 14, 15, 16]
        
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U[a22_u[i*3+j], col_offset] = 1
                    V[b22_v[j*3+k], col_offset] = 1
                    W[c22_w[k*2+i], col_offset] = 1
                    col_offset += 1
        
        return U, V, W, col_offset

    # Try to build the rank-98 decomposition
    U98, V98, W98, total_cols = build_rank98()
    
    if total_cols != 98:
        print(f"Warning: expected 98 columns but got {total_cols}")
    
    # Verify the decomposition
    if verify_decomposition(U98, V98, W98, n, m, p):
        return {"U": U98.tolist(), "V": V98.tolist(), "W": W98.tolist()}
    else:
        # Fall back to schoolbook
        U_sb, V_sb, W_sb = schoolbook(n, m, p)
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
