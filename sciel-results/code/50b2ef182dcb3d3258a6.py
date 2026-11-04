"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Check if the decomposition is exact. Returns True if valid."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Reshape to 3D tensors
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    # result[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    return np.allclose(result, 1.0, atol=1e-9)

def _strassen_2x2():
    """
    Strassen decomposition for 2x2x2 matrix multiplication. Rank 7.
    Returns U (4x7), V (4x7), W (4x7) where:
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # c=0: M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # c=1: M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # c=2: M3 = A11(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # c=3: M4 = A22(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # c=4: M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # c=5: M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # c=6: M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Fallback
    best = _schoolbook()
    
    # Total rank: 8*7 (4x4x4 Strassen) + 2 (4x4x2) + 1 (4x1x6) = 59
    total_rank = 59
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    U_s, V_s, W_s = _strassen_2x2()
    
    # Part 1: 4x4x4 core (i=0..3, j=0..3, k=0..3)
    # 8 blocks of 2x2x2
    for i_block in range(2):
        for j_block in range(2):
            for k_block in range(2):
                i_off = i_block * 2
                j_off = j_block * 2
                k_off = k_block * 2
                for r in range(4):
                    for c in range(7):
                        i_l = r // 2
                        j_l = r % 2
                        # U global index: (i_off + i_l) * m + (j_off + j_l)
                        U[(i_off + i_l) * m + (j_off + j_l)][col + c] += U_s[r][c]
                        # V global index: (j_off + j_l) * p + (k_off + k_l)
                        # But wait, for V, r corresponds to (j_l, k_l) = (r//2, r%2)
                        k_l = r % 2
                        V[(j_off + j_l) * p + (k_off + k_l)][col + c] += V_s[r][c]
                        # W global index: (k_off + k_l) * n + (i_off + i_l)
                        # For W, r corresponds to (k_l, i_l) = (r//2, r%2)
                        # So k_l = r//2, i_l_w = r%2
                        # But i_l for W should match the i from the block
                        # Actually W[k*2+i] so row r gives k_l=r//2, i_l=r%2
                        # But we need i_l to be the same i_l as in the block
                        # This is the issue! W's i_l is r%2, but our block's i_l is from the outer loop
                        # 
                        # Let me reconsider. In the 2x2x2 Strassen:
                        # U rows: 0=A11, 1=A12, 2=A21, 3=A22 -> (i_l, j_l) = (0,0),(0,1),(1,0),(1,1)
                        # V rows: 0=B11, 1=B12, 2=B21, 3=B22 -> (j_l, k_l) = (0,0),(0,1),(1,0),(1,1)
                        # W rows: 0=C11, 1=C12, 2=C21, 3=C22 -> (k_l, i_l) = (0,0),(0,1),(1,0),(1,1)
                        #
                        # So for W, row r gives (k_l, i_l) = (r//2, r%2)
                        # The i_l here is the local i index, which should match the block's i_l
                        # But r%2 for W is not necessarily the same as the i_l from U's mapping
                        #
                        # Actually, in the Strassen formula, all three use the same local indices
                        # The key is that U[i_l*2+j_l], V[j_l*2+k_l], W[k_l*2+i_l]
                        # For a given column c, the non-zero entries determine which (i_l,j_l,k_l) are involved
                        # The row index r in U corresponds to (i_l,j_l), in V to (j_l,k_l), in W to (k_l,i_l)
                        # These are consistent because they all refer to the same 2x2x2 sub-tensor
                        #
                        # So for W, row r: k_l = r//2, i_l_w = r%2
                        # But we need i_l_w to equal the block's i_l... 
                        # No! The i_l in W[k_l*2+i_l] is the LOCAL i index within the 2x2 block
                        # which is determined by which C entry we're computing
                        # 
                        # I think the correct mapping is:
                        # For each (i_l, j_l, k_l) in {0,1}^3, the tensor entry is 1
                        # Strassen decomposes this into 7 rank-1 tensors
                        # Each rank-1 tensor has specific (i_l, j_l) for U, (j_l, k_l) for V, (k_l, i_l) for W
                        # The row index in U/V/W encodes which local indices are active
                        #
                        # So for W, row r encodes (k_l, i_l) = (r//2, r%2)
                        # The global W index should be: (k_off + k_l) * n + (i_off + i_l)
                        # where k_l = r//2 and i_l = r%2
                        k_l_w = r // 2
                        i_l_w = r % 2
                        W[(k_off + k_l_w) * n + (i_off + i_l_w)][col + c] += W_s[r][c]
                col += 7
    
    # Part 2: 4x4x2 (j=0..3, k=4,5) - rank 2
    for k_extra in range(2):
        k_global = 4 + k_extra
        for i in range(n):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + k_global][col] = 1
                W[k_global * n + i][col] = 1
        col += 1
    
    # Part 3: 4x1x6 (j=4, k=0..5) - rank 1
    for i in range(n):
        for k in range(p):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
    col += 1
    
    # Verify
    if _verify(U, V, W):
        best = {"U": U, "V": V, "W": W}
    
    return best
# EVOLVE-BLOCK-END
