"""Matrix multiplication 2x4 by 4x4 via block Strassen decomposition, rank 28."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-28 decomposition of 2x4 * 4x4 -> 2x4.
    
    Strategy: Partition B into 2x2 blocks. For each pair (A_sub, B_sub),
    use Strassen's 7-product formula for 2x2*2x2->2x2 (rank 7).
    Four such sub-products give total rank 28.
    
    Convention: U[i*m+k, c], V[k*p+j, c], W[j*n+i, c]
    where n=2, m=4, p=4.
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    
    def w_idx(i, j):
        """W index for output C[i,j] is j*n + i."""
        return j * n + i
    
    def add_strassen(col_off, A_idx, B_idx, C_coords):
        """
        Add one 2x2 Strassen decomposition (rank 7) starting at column col_off.
        
        A_idx: [a00, a01, a10, a11] - flat indices into U
        B_idx: [b00, b01, b10, b11] - flat indices into V  
        C_coords: [(i0,j0),(i0,j1),(i1,j0),(i1,j1)] - output coordinates
        """
        a00, a01, a10, a11 = A_idx
        b00, b01, b10, b11 = B_idx
        (ci00, cj00), (ci01, cj01), (ci10, cj10), (ci11, cj11) = C_coords
        w00 = w_idx(ci00, cj00)
        w01 = w_idx(ci01, cj01)
        w10 = w_idx(ci10, cj10)
        w11 = w_idx(ci11, cj11)
        
        # P1 = (a00+a11)*(b00+b11)
        U[a00][col_off+0] += 1; U[a11][col_off+0] += 1
        V[b00][col_off+0] += 1; V[b11][col_off+0] += 1
        W[w00][col_off+0] += 1; W[w01][col_off+0] += 1
        W[w10][col_off+0] += 1; W[w11][col_off+0] += 1
        
        # P2 = (a10+a11)*b00
        U[a10][col_off+1] += 1; U[a11][col_off+1] += 1
        V[b00][col_off+1] += 1
        W[w10][col_off+1] += 1; W[w11][col_off+1] -= 1
        
        # P3 = a00*(b01-b11)
        U[a00][col_off+2] += 1
        V[b01][col_off+2] += 1; V[b11][col_off+2] -= 1
        W[w01][col_off+2] += 1; W[w11][col_off+2] += 1
        
        # P4 = a11*(b10-b00)
        U[a11][col_off+3] += 1
        V[b10][col_off+3] += 1; V[b00][col_off+3] -= 1
        W[w00][col_off+3] += 1; W[w10][col_off+3] += 1
        
        # P5 = (a00+a01)*b11
        U[a00][col_off+4] += 1; U[a01][col_off+4] += 1
        V[b11][col_off+4] += 1
        W[w00][col_off+4] -= 1; W[w01][col_off+4] += 1
        
        # P6 = (a10-a00)*(b00+b01)
        U[a10][col_off+5] += 1; U[a00][col_off+5] -= 1
        V[b00][col_off+5] += 1; V[b01][col_off+5] += 1
        W[w11][col_off+5] += 1
        
        # P7 = (a01-a11)*(b10+b11)
        U[a01][col_off+6] += 1; U[a11][col_off+6] -= 1
        V[b10][col_off+6] += 1; V[b11][col_off+6] += 1
        W[w00][col_off+6] += 1; W[w11][col_off+6] -= 1
    
    # A flat indices (2x4 row-major):
    # A[i,k] -> i*4+k
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    
    # B flat indices (4x4 row-major):
    # B[k,j] -> k*4+j
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3
    # B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    # B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    # B[3,0]=12, B[3,1]=13, B[3,2]=14, B[3,3]=15
    
    # A_left = A[:, 0:2] as 2x2: [[A[0,0],A[0,1]],[A[1,0],A[1,1]]] -> [0,1,4,5]
    # A_right = A[:, 2:4] as 2x2: [[A[0,2],A[0,3]],[A[1,2],A[1,3]]] -> [2,3,6,7]
    
    # B blocks:
    # B[0:2,0:2] -> [0,1,4,5]
    # B[0:2,2:4] -> [2,3,6,7]
    # B[2:4,0:2] -> [8,9,12,13]
    # B[2:4,2:4] -> [10,11,14,15]
    
    # C output coordinates for each 2x2 block:
    # C[:,0:2] -> [(0,0),(0,1),(1,0),(1,1)]
    # C[:,2:4] -> [(0,2),(0,3),(1,2),(1,3)]
    
    A_left = [0, 1, 4, 5]
    A_right = [2, 3, 6, 7]
    B00 = [0, 1, 4, 5]
    B01 = [2, 3, 6, 7]
    B10 = [8, 9, 12, 13]
    B11 = [10, 11, 14, 15]
    C_left = [(0,0),(0,1),(1,0),(1,1)]
    C_right = [(0,2),(0,3),(1,2),(1,3)]
    
    # C[:,0:2] = A_left * B00 + A_right * B10
    add_strassen(0, A_left, B00, C_left)
    add_strassen(7, A_right, B10, C_left)
    
    # C[:,2:4] = A_left * B01 + A_right * B11
    add_strassen(14, A_left, B01, C_right)
    add_strassen(21, A_right, B11, C_right)
    
    # Verify correctness
    np.random.seed(12345)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                for c in range(rank):
                    C_computed[i, j] += U[i*m+k][c] * V[k*p+j][c] * W[j*n+i][c]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook rank 32
        rank_s = 32
        U_s = [[0] * rank_s for _ in range(n * m)]
        V_s = [[0] * rank_s for _ in range(m * p)]
        W_s = [[0] * rank_s for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_s[i*m+k][col] = 1
                    V_s[k*p+j][col] = 1
                    W_s[j*n+i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
