import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_fractions(U, V, W, n, m, p):
    """Verify with exact fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0, 1)
                for c in range(rank):
                    s += Fraction(U[i * m + j][c], 1) * Fraction(V[j * p + k][c], 1) * Fraction(W[k * n + i][c], 1)
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    t_start = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Strategy: Use a block decomposition that minimizes rank.
    # The tensor for matrix multiplication M_{n,m,p} has rank at most n*m*p (schoolbook).
    # For 4x5x5, schoolbook gives rank 100.
    # We can do better using Strassen for 2x2x2 blocks and Laderman for 3x3x3 blocks.
    
    # Let's try a clean approach:
    # Partition the 4x5x5 multiplication into blocks.
    # A is 4x5, B is 5x5, C is 4x5.
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in [0,4), j in [0,5), k in [0,5).
    
    # The tensor T[i,j,k] = 1 for all valid i,j,k.
    # We want to decompose T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Key insight: We can use Strassen's algorithm on 2x2x2 blocks.
    # For a 2x2x2 block, Strassen uses rank 7 instead of 8.
    
    # Let's partition:
    # i: [0,2), [2,4)  (two blocks of size 2)
    # j: [0,2), [2,4), [4,5)  (blocks of size 2, 2, 1)
    # k: [0,2), [2,4), [4,5)  (blocks of size 2, 2, 1)
    
    # This gives blocks:
    # For each (i_block, j_block, k_block) combination:
    #   bn, bm, bp are the block sizes
    #   If bn=bm=bp=2: use Strassen (rank 7)
    #   Otherwise: use schoolbook (rank bn*bm*bp)
    
    # Blocks:
    # (0,0): i=[0,2), j=[0,2), k=[0,2) -> 2x2x2 -> Strassen rank 7
    # (0,1): i=[0,2), j=[0,2), k=[2,4) -> 2x2x2 -> Strassen rank 7
    # (0,2): i=[0,2), j=[0,2), k=[4,5) -> 2x2x1 -> schoolbook rank 4
    # (1,0): i=[0,2), j=[2,4), k=[0,2) -> 2x2x2 -> Strassen rank 7
    # (1,1): i=[0,2), j=[2,4), k=[2,4) -> 2x2x2 -> Strassen rank 7
    # (1,2): i=[0,2), j=[2,4), k=[4,5) -> 2x2x1 -> schoolbook rank 4
    # (2,0): i=[0,2), j=[4,5), k=[0,2) -> 2x1x2 -> schoolbook rank 4
    # (2,1): i=[0,2), j=[4,5), k=[2,4) -> 2x1x2 -> schoolbook rank 4
    # (2,2): i=[0,2), j=[4,5), k=[4,5) -> 2x1x1 -> schoolbook rank 2
    # (3,0): i=[2,4), j=[0,2), k=[0,2) -> 2x2x2 -> Strassen rank 7
    # (3,1): i=[2,4), j=[0,2), k=[2,4) -> 2x2x2 -> Strassen rank 7
    # (3,2): i=[2,4), j=[0,2), k=[4,5) -> 2x2x1 -> schoolbook rank 4
    # (4,0): i=[2,4), j=[2,4), k=[0,2) -> 2x2x2 -> Strassen rank 7
    # (4,1): i=[2,4), j=[2,4), k=[2,4) -> 2x2x2 -> Strassen rank 7
    # (4,2): i=[2,4), j=[2,4), k=[4,5) -> 2x2x1 -> schoolbook rank 4
    # (5,0): i=[2,4), j=[4,5), k=[0,2) -> 2x1x2 -> schoolbook rank 4
    # (5,1): i=[2,4), j=[4,5), k=[2,4) -> 2x1x2 -> schoolbook rank 4
    # (5,2): i=[2,4), j=[4,5), k=[4,5) -> 2x1x1 -> schoolbook rank 2
    
    # Total rank: 7*8 + 4*8 + 2*2 = 56 + 32 + 4 = 92
    
    # Alternative: Try using 3x3 blocks where possible.
    # i: [0,3), [3,4)
    # j: [0,3), [3,5)
    # k: [0,3), [3,5)
    
    # Blocks:
    # (0,0): 3x3x3 -> Laderman rank 13
    # (0,1): 3x3x2 -> schoolbook rank 18
    # (0,2): 3x2x3 -> schoolbook rank 18
    # (0,3): 3x2x2 -> schoolbook rank 12
    # (1,0): 1x3x3 -> schoolbook rank 9
    # (1,1): 1x3x2 -> schoolbook rank 6
    # (1,2): 1x2x3 -> schoolbook rank 6
    # (1,3): 1x2x2 -> schoolbook rank 4
    
    # Total rank: 13 + 18 + 18 + 12 + 9 + 6 + 6 + 4 = 86
    
    # This is better! Let's implement this.
    
    # Helper functions
    def _strassen_block(n_full, m_full, p_full, i_off, j_off, k_off):
        """Strassen 2x2x2 block."""
        rank = 7
        U = np.zeros((n_full * m_full, rank))
        V = np.zeros((m_full * p_full, rank))
        W = np.zeros((p_full * n_full, rank))
        
        i0, i1 = i_off, i_off + 1
        j0, j1 = j_off, j_off + 1
        k0, k1 = k_off, k_off + 1
        
        def ui(i, j): return i * m_full + j
        def vj(j, k): return j * p_full + k
        def wk(k, i): return k * n_full + i
        
        # M1 = (A00+A11)(B00+B11)
        U[ui(i0, j0), 0] = 1
        U[ui(i1, j1), 0] = 1
        V[vj(j0, k0), 0] = 1
        V[vj(j1, k1), 0] = 1
        W[wk(k0, i0), 0] = 1
        W[wk(k1, i1), 0] = 1
        
        # M2 = (A10+A11)B00
        U[ui(i1, j0), 1] = 1
        U[ui(i1, j1), 1] = 1
        V[vj(j0, k0), 1] = 1
        W[wk(k0, i1), 1] = 1
        W[wk(k1, i1), 1] = -1
        
        # M3 = A00(B01-B11)
        U[ui(i0, j0), 2] = 1
        V[vj(j0, k1), 2] = 1
        V[vj(j1, k1), 2] = -1
        W[wk(k1, i0), 2] = 1
        W[wk(k1, i1), 2] = 1
        
        # M4 = A11(B10-B00)
        U[ui(i1, j1), 3] = 1
        V[vj(j1, k0), 3] = 1
        V[vj(j0, k0), 3] = -1
        W[wk(k0, i0), 3] = 1
        W[wk(k0, i1), 3] = 1
        
        # M5 = (A00+A01)B11
        U[ui(i0, j0), 4] = 1
        U[ui(i0, j1), 4] = 1
        V[vj(j1, k1), 4] = 1
        W[wk(k0, i0), 4] = -1
        W[wk(k1, i0), 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[ui(i1, j0), 5] = 1
        U[ui(i0, j0), 5] = -1
        V[vj(j0, k0), 5] = 1
        V[vj(j0, k1), 5] = 1
        W[wk(k1, i1), 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[ui(i0, j1), 6] = 1
        U[ui(i1, j1), 6] = -1
        V[vj(j1, k0), 6] = 1
        V[vj(j1, k1), 6] = 1
        W[wk(k0, i0), 6] = 1
        
        return U, V, W
    
    def _laderman_block(n_full, m_full, p_full, i_off, j_off, k_off):
        """Laderman 3x3x3 block."""
        rank = 13
        U = np.zeros((n_full * m_full, rank))
        V = np.zeros((m_full * p_full, rank))
        W = np.zeros((p_full * n_full, rank))
        
        def ui(i, j): return (i_off + i) * m_full + (j_off + j)
        def vj(j, k): return (j_off + j) * p_full + (k_off + k)
        def wk(k, i): return (k_off + k) * n_full + (i_off + i)
        
        # Laderman's 13-term formula
        # M0 = A[0,0] * (B[0,0]+B[1,1]+B[2,2])
        U[ui(0,0), 0] = 1
        V[vj(0,0), 0] = 1
        V[vj(1,1), 0] = 1
        V[vj(2,2), 0] = 1
        W[wk(0,0), 0] = 1
        W[wk(1,1), 0] = 1
        W[wk(2,2), 0] = 1
        
        # M1 = (A[0,0]-A[1,0]) * (B[1,0]+B[2,0])
        U[ui(0,0), 1] = 1
        U[ui(1,0), 1] = -1
        V[vj(1,0), 1] = 1
        V[vj(2,0), 1] = 1
        W[wk(0,0), 1] = 1
        W[wk(1,0), 1] = 1
        
        # M2 = (A[0,0]-A[2,0]) * (B[2,0]+B[2,1])
        U[ui(0,0), 2] = 1
        U[ui(2,0), 2] = -1
        V[vj(2,0), 2] = 1
        V[vj(2,1), 2] = 1
        W[wk(0,0), 2] = 1
        W[wk(2,1), 2] = 1
        
        # M3 = (A[1,1]+A[2,2]-A[0,0]) * (B[0,1]+B[0,2])
        U[ui(1,1), 3] = 1
        U[ui(2,2), 3] = 1
        U[ui(0,0), 3] = -1
        V[vj(0,1), 3] = 1
        V[vj(0,2), 3] = 1
        W[wk(1,0), 3] = 1
        W[wk(2,0), 3] = 1
        
        # M4 = A[1,1] * (B[1,0]+B[2,0]+B[0,1]+B[0,2])
        U[ui(1,1), 4] = 1
        V[vj(1,0), 4] = 1
        V[vj(2,0), 4] = 1
        V[vj(0,1), 4] = 1
        V[vj(0,2), 4] = 1
        W[wk(1,1), 4] = 1
        W[wk(2,1), 4] = 1
        W[wk(0,2), 4] = 1
        W[wk(1,2), 4] = 1
        
        # M5 = (A[1,1]-A[2,1]) * (B[2,0]+B[2,1]+B[0,2])
        U[ui(1,1), 5] = 1
        U[ui(2,1), 5] = -1
        V[vj(2,0), 5] = 1
        V[vj(2,1), 5] = 1
        V[vj(0,2), 5] = 1
        W[wk(1,1), 5] = 1
        W[wk(2,1), 5] = 1
        W[wk(0,2), 5] = 1
        
        # M6 = A[2,2] * (B[1,0]+B[2,1]+B[0,2])
        U[ui(2,2), 6] = 1
        V[vj(1,0), 6] = 1
        V[vj(2,1), 6] = 1
        V[vj(0,2), 6] = 1
        W[wk(2,2), 6] = 1
        W[wk(1,2), 6] = 1
        W[wk(0,2), 6] = 1
        
        # M7 = A[1,2] * (B[0,0]+B[1,1]+B[2,2])
        U[ui(1,2), 7] = 1
        V[vj(0,0), 7] = 1
        V[vj(1,1), 7] = 1
        V[vj(2,2), 7] = 1
        W[wk(1,2), 7] = 1
        W[wk(2,2), 7] = 1
        
        # M8 = (A[0,1]-A[2,1]) * (B[1,0]+B[2,0])
        U[ui(0,1), 8] = 1
        U[ui(2,1), 8] = -1
        V[vj(1,0), 8] = 1
        V[vj(2,0), 8] = 1
        W[wk(0,1), 8] = 1
        W[wk(2,1), 8] = 1
        
        # M9 = (A[0,2]-A[1,2]) * (B[0,1]+B[0,2])
        U[ui(0,2), 9] = 1
        U[ui(1,2), 9] = -1
        V[vj(0,1), 9] = 1
        V[vj(0,2), 9] = 1
        W[wk(0,2), 9] = 1
        W[wk(1,2), 9] = 1
        
        # M10 = (A[0,1]-A[1,1]) * (B[1,0]+B[2,0]+B[0,2])
        U[ui(0,1), 10] = 1
        U[ui(1,1), 10] = -1
        V[vj(1,0), 10] = 1
        V[vj(2,0), 10] = 1
        V[vj(0,2), 10] = 1
        W[wk(0,1), 10] = 1
        W[wk(2,1), 10] = 1
        W[wk(0,2), 10] = 1
        
        # M11 = (A[0,2]-A[2,2]) * (B[0,1]+B[0,2]+B[2,1])
        U[ui(0,2), 11] = 1
        U[ui(2,2), 11] = -1
        V[vj(0,1), 11] = 1
        V[vj(0,2), 11] = 1
        V[vj(2,1), 11] = 1
        W[wk(0,2), 11] = 1
        W[wk(2,2), 11] = 1
        W[wk(2,1), 11] = 1
        
        # M12 = (A[1,2]-A[2,2]) * (B[0,0]+B[1,1]+B[2,2])
        U[ui(1,2), 12] = 1
        U[ui(2,2), 12] = -1
        V[vj(0,0), 12] = 1
        V[vj(1,1), 12] = 1
        V[vj(2,2), 12] = 1
        W[wk(1,2), 12] = 1
        W[wk(2,2), 12] = 1
        
        return U, V, W
    
    def _schoolbook_block(bn, bm, bp, n_full, m_full, p_full, i_off, j_off, k_off):
        """Schoolbook block."""
        rank = bn * bm * bp
        U = np.zeros((n_full * m_full, rank))
        V = np.zeros((m_full * p_full, rank))
        W = np.zeros((p_full * n_full, rank))
        col = 0
        for ii in range(bn):
            for jj in range(bm):
                for kk in range(bp):
                    i = i_off + ii
                    j = j_off + jj
                    k = k_off + kk
                    U[i * m_full + j, col] = 1
                    V[j * p_full + k, col] = 1
                    W[k * n_full + i, col] = 1
                    col += 1
        return U, V, W
    
    # Build the Laderman-based decomposition (rank 86)
    U_list = []
    V_list = []
    W_list = []
    
    # i partitions: [0,3), [3,4)
    # j partitions: [0,3), [3,5)
    # k partitions: [0,3), [3,5)
    
    i_parts = [(0, 3), (3, 4)]
    j_parts = [(0, 3), (3, 5)]
    k_parts = [(0, 3), (3, 5)]
    
    for i_start, i_end in i_parts:
        for j_start, j_end in j_parts:
            for k_start, k_end in k_parts:
                bn = i_end - i_start
                bm = j_end - j_start
                bp = k_end - k_start
                
                if bn == 3 and bm == 3 and bp == 3:
                    U_b, V_b, W_b = _laderman_block(n, m, p, i_start, j_start, k_start)
                else:
                    U_b, V_b, W_b = _schoolbook_block(bn, bm, bp, n, m, p, i_start, j_start, k_start)
                
                U_list.append(U_b)
                V_list.append(V_b)
                W_list.append(W_b)
    
    U_combined = np.hstack(U_list)
    V_combined = np.hstack(V_list)
    W_combined = np.hstack(W_list)
    
    U_check = U_combined.tolist()
    V_check = V_combined.tolist()
    W_check = W_combined.tolist()
    
    # Verify
    if _verify(U_check, V_check, W_check, n, m, p):
        rank = len(U_check[0])
        # Also verify with fractions for exactness
        if _verify_fractions(U_check, V_check, W_check, n, m, p):
            return {"U": U_check, "V": V_check, "W": W_check}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
