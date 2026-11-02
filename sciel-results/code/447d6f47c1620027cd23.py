"""Combined program: block-structured Strassen-like decomposition for 3x4 * 4x4."""
import numpy as np
from fractions import Fraction
import time
import itertools

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy einsum."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    
    return np.allclose(recon, expected, atol=1e-9)


def schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def strassen_2x2():
    """
    Return the Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 multiplications for 2x2 * 2x2
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A10)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For the tensor decomposition, we need:
    # sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1 for all i,j,k in {0,1}
    
    # Let's index: i,j,k in {0,1}
    # U has rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V has rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W has rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    # A00: U[0,0]=1, A11: U[3,0]=1
    # B00: V[0,0]=1, B11: V[3,0]=1
    # C00: W[0,0]=1, C11: W[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)*B00
    # A10: U[2,1]=1, A11: U[3,1]=1
    # B00: V[0,1]=1
    # C10: W[2,1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00*(B01-B11)
    # A00: U[0,2]=1
    # B01: V[1,2]=1, B11: V[3,2]=-1
    # C01: W[1,2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11*(B10-B00)
    # A11: U[3,3]=1
    # B10: V[2,3]=1, B00: V[0,3]=-1
    # C00: W[0,3]=1, C10: W[2,3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A10)*B11
    # A00: U[0,4]=1, A10: U[2,4]=1
    # B11: V[3,4]=1
    # C00: W[0,4]=-1, C01: W[1,4]=1
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A01-A10)*(B00+B10)
    # A01: U[1,5]=1, A10: U[2,5]=-1
    # B00: V[0,5]=1, B10: V[2,5]=1
    # C11: W[3,5]=1
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)*(B01+B11)
    # A01: U[1,6]=1, A11: U[3,6]=-1
    # B01: V[1,6]=1, B11: V[3,6]=1
    # C00: W[0,6]=1, C11: W[3,6]=-1
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W


def row_times_2x2_block():
    """
    Decomposition for 1x2 * 2x2 matrix multiplication.
    A is 1x2, B is 2x2, C is 1x2.
    Rank: 4 (schoolbook for this subproblem)
    
    Actually, let's think: C0 = A0*B00 + A1*B10, C1 = A0*B01 + A1*B11
    This is 4 products in schoolbook. Can we do better?
    For 1x2 * 2x2, the tensor has 1*2*2 = 4 nonzero entries, so rank >= 4.
    So schoolbook is optimal: rank 4.
    
    Returns U (2x4), V (4x4), W (2x4) where:
    U[j, c] for j in {0,1} (the two elements of A)
    V[j*2+k, c] for j in {0,1}, k in {0,1} (the 2x2 B)
    W[k, c] for k in {0,1} (the two elements of C)
    """
    rank = 4
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(2)]
    
    col = 0
    for j in range(2):
        for k in range(2):
            U[j][col] = 1
            V[j*2+k][col] = 1
            W[k][col] = 1
            col += 1
    
    return U, V, W


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    Uses block-structured decomposition with Strassen for 2x2*2x2 blocks.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    deadline = time.time() + 130  # Safety margin
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = schoolbook(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p  # 48
    
    # Block decomposition approach:
    # Split A (3x4) into A_top (2x4) and A_bottom (1x4)
    # Split B (4x4) into B_left (4x2) and B_right (4x2)
    # 
    # C = A * B is 3x4
    # C_top (2x4) = A_top (2x4) * B (4x4)
    # C_bottom (1x4) = A_bottom (1x4) * B (4x4)
    #
    # Further split:
    # A_top (2x4) = [A_tl (2x2) | A_tr (2x2)]
    # A_bottom (1x4) = [A_bl (1x2) | A_br (1x2)]
    # B (4x4) = [[B_tl (2x2), B_tr (2x2)], [B_bl (2x2), B_br (2x2)]]
    #
    # C_top_left (2x2) = A_tl * B_tl + A_tr * B_bl  -> two 2x2*2x2 products
    # C_top_right (2x2) = A_tl * B_tr + A_tr * B_br  -> two 2x2*2x2 products
    # C_bottom_left (1x2) = A_bl * B_tl + A_br * B_bl  -> two 1x2*2x2 products
    # C_bottom_right (1x2) = A_bl * B_tr + A_br * B_br  -> two 1x2*2x2 products
    #
    # Rank: 4*7 + 4*4 = 28 + 16 = 44
    
    # Let's build this decomposition.
    # We need to carefully map the indices.
    
    # The full U has shape (n*m, rank) = (12, rank)
    # The full V has shape (m*p, rank) = (16, rank)
    # The full W has shape (p*n, rank) = (12, rank)
    
    # Index conventions:
    # U[i*m + j, c] where i in [0,3), j in [0,4)
    # V[j*p + k, c] where j in [0,4), k in [0,4)
    # W[k*n + i, c] where k in [0,4), i in [0,3)
    
    # Block assignments:
    # A_top: i in {0,1}, j in {0,3}
    # A_bottom: i in {2}, j in {0,3}
    # A_tl: i in {0,1}, j in {0,1}
    # A_tr: i in {0,1}, j in {2,3}
    # A_bl: i in {2}, j in {0,1}
    # A_br: i in {2}, j in {2,3}
    
    # B_tl: j in {0,1}, k in {0,1}
    # B_tr: j in {0,1}, k in {2,3}
    # B_bl: j in {2,3}, k in {0,1}
    # B_br: j in {2,3}, k in {2,3}
    
    # C_top_left: k in {0,1}, i in {0,1}
    # C_top_right: k in {2,3}, i in {0,1}
    # C_bottom_left: k in {0,1}, i in {2}
    # C_bottom_right: k in {2,3}, i in {2}
    
    # We'll build the decomposition by combining Strassen for each 2x2*2x2 block
    # and schoolbook for each 1x2*2x2 block.
    
    # Total rank: 4 * 7 + 4 * 4 = 28 + 16 = 44
    
    rank = 44
    U = [[0] * rank for _ in range(12)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(12)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(U_full, V_full, W_full, i_offset, j_offset, k_offset, col_start):
        """
        Add a Strassen 2x2*2x2 block.
        i_offset, j_offset, k_offset: offsets for the block in the full matrices.
        The block covers:
        A: i in [i_offset, i_offset+2), j in [j_offset, j_offset+2)
        B: j in [j_offset, j_offset+2), k in [k_offset, k_offset+2)
        C: k in [k_offset, k_offset+2), i in [i_offset, i_offset+2)
        """
        U_s, V_s, W_s = strassen_2x2()
        for c in range(7):
            for ii in range(2):
                for jj in range(2):
                    U_full[(i_offset + ii) * 4 + (j_offset + jj)][col_start + c] += U_s[ii * 2 + jj][c]
            for jj in range(2):
                for kk in range(2):
                    V_full[(j_offset + jj) * 4 + (k_offset + kk)][col_start + c] += V_s[jj * 2 + kk][c]
            for kk in range(2):
                for ii in range(2):
                    W_full[(k_offset + kk) * 3 + (i_offset + ii)][col_start + c] += W_s[kk * 2 + ii][c]
    
    # Helper to add a 1x2 * 2x2 block
    def add_row_block(U_full, V_full, W_full, i_idx, j_offset, k_offset, col_start):
        """
        Add a 1x2 * 2x2 block.
        A: i = i_idx, j in [j_offset, j_offset+2)
        B: j in [j_offset, j_offset+2), k in [k_offset, k_offset+2)
        C: k in [k_offset, k_offset+2), i = i_idx
        """
        U_r, V_r, W_r = row_times_2x2_block()
        for c in range(4):
            for jj in range(2):
                U_full[i_idx * 4 + (j_offset + jj)][col_start + c] += U_r[jj][c]
            for jj in range(2):
                for kk in range(2):
                    V_full[(j_offset + jj) * 4 + (k_offset + kk)][col_start + c] += V_r[jj * 2 + kk][c]
            for kk in range(2):
                W_full[(k_offset + kk) * 3 + i_idx][col_start + c] += W_r[kk][c]
    
    # C_top_left (2x2) = A_tl * B_tl + A_tr * B_bl
    # A_tl: i in {0,1}, j in {0,1}; B_tl: j in {0,1}, k in {0,1}
    add_strassen_block(U, V, W, 0, 0, 0, col)
    col += 7
    
    # A_tr: i in {0,1}, j in {2,3}; B_bl: j in {2,3}, k in {0,1}
    add_strassen_block(U, V, W, 0, 2, 0, col)
    col += 7
    
    # C_top_right (2x2) = A_tl * B_tr + A_tr * B_br
    # A_tl: i in {0,1}, j in {0,1}; B_tr: j in {0,1}, k in {2,3}
    add_strassen_block(U, V, W, 0, 0, 2, col)
    col += 7
    
    # A_tr: i in {0,1}, j in {2,3}; B_br: j in {2,3}, k in {2,3}
    add_strassen_block(U, V, W, 0, 2, 2, col)
    col += 7
    
    # C_bottom_left (1x2) = A_bl * B_tl + A_br * B_bl
    # A_bl: i=2, j in {0,1}; B_tl: j in {0,1}, k in {0,1}
    add_row_block(U, V, W, 2, 0, 0, col)
    col += 4
    
    # A_br: i=2, j in {2,3}; B_bl: j in {2,3}, k in {0,1}
    add_row_block(U, V, W, 2, 2, 0, col)
    col += 4
    
    # C_bottom_right (1x2) = A_bl * B_tr + A_br * B_br
    # A_bl: i=2, j in {0,1}; B_tr: j in {0,1}, k in {2,3}
    add_row_block(U, V, W, 2, 0, 2, col)
    col += 4
    
    # A_br: i=2, j in {2,3}; B_br: j in {2,3}, k in {2,3}
    add_row_block(U, V, W, 2, 2, 2, col)
    col += 4
    
    # Verify the decomposition
    if time.time() < deadline:
        if _verify_fast(U, V, W, n, m, p):
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
        else:
            # If verification fails, fall back to schoolbook
            pass
    
    return best_result

# EVOLVE-BLOCK-END
