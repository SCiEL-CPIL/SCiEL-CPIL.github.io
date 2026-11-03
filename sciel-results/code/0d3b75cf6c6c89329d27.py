"""Matrix multiplication 2,4,7 tensor decomposition using Strassen-like block decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 140  # Safety margin

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify if the decomposition is exact."""
    rank = U.shape[1]
    U_ = U.reshape(n, m, rank)
    V_ = V.reshape(m, p, rank)
    W_ = W.reshape(p, n, rank)
    
    # The tensor T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # We want T_{i,j,k} = 1 for all i,j,k
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        result += np.einsum('ij,jk,ki->ijk', U_[:, :, r], V_[:, :, r], W_[:, :, r])
    
    if np.max(np.abs(result - 1.0)) > tol:
        return False
    return True

def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products for 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U factors (A side): rows are (i,j) for i,j in {0,1}
    # V factors (B side): rows are (j,k) for j,k in {0,1}
    # W factors (C side): rows are (k,i) for k,i in {0,1}
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1  # C00 += M1, C11 -= M1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C10 += M2, C11 -= M2
    
    # M3: A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[2, 2] = 1  # C01 += M3, C11 += M3
    
    # M4: A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C00 += M4, C10 += M4
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C00 -= M5, C01 += M5
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11 += M6
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00 += M7
    
    return U, V, W

def _strassen_2x2x1():
    """Return a rank-4 decomposition for 2x2x1 matrix-vector multiplication."""
    rank = 4
    U = np.zeros((4, rank))
    V = np.zeros((2, rank))
    W = np.zeros((2, rank))
    
    # For C = A * b where A is 2x2, b is 2x1:
    # C0 = A00*b0 + A01*b1
    # C1 = A10*b0 + A11*b1
    # 
    # We can use 4 products:
    # P1 = A00*b0
    # P2 = A01*b1
    # P3 = A10*b0
    # P4 = A11*b1
    # C0 = P1 + P2
    # C1 = P3 + P4
    
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def _construct_strassen_2x4x7(n=2, m=4, p=7):
    """
    Construct a Strassen-like decomposition for 2x4x7.
    
    Split the 4x7 matrix into blocks:
    - Top half: 2x4 (rows 0-1, cols 0-3)
    - Bottom half: 2x4 (rows 2-3, cols 0-3)
    Wait, that doesn't work for 4x7.
    
    Better: Split the middle dimension (size 4) into two halves of size 2.
    Split the right dimension (size 7) into... this is tricky.
    
    Let's think differently. We have:
    A: 2x4
    B: 4x7
    C: 2x7
    
    We can split B into two 4x3.5 blocks? No, we need integer splits.
    
    Alternative: Use the fact that we can decompose the multiplication by
    splitting the inner product dimension.
    
    Actually, let's use a recursive approach:
    - Split the 4 (inner dimension) into 2+2
    - Split the 7 (outer dimension) into 3+4 or similar
    
    For 2x4x7:
    Split inner dim 4 = 2 + 2
    Split outer dim 7 = 3 + 4
    
    This gives us subproblems:
    - 2x2x3
    - 2x2x4
    - 2x2x3 (cross term?)
    
    Actually, the standard block decomposition for (n x m) * (m x p) with
    m = m1 + m2 is:
    C = A1 * B1 + A2 * B2
    where A1 is n x m1, A2 is n x m2, B1 is m1 x p, B2 is m2 x p.
    
    So for 2x4x7 with m=4=2+2:
    C = A1 * B1 + A2 * B2
    where A1, A2 are 2x2, B1, B2 are 2x7.
    
    Each term is a 2x2x7 multiplication.
    
    For 2x2x7, we can further split the 7 into 3+4:
    2x2x7 = 2x2x3 + 2x2x4
    
    For 2x2x3, we can use:
    - Split 3 into 1+2: 2x2x1 + 2x2x2
    - Rank(2x2x1) = 4, Rank(2x2x2) = 7, total = 11
    
    For 2x2x4, we can use:
    - Split 4 into 2+2: two 2x2x2
    - Rank = 7 + 7 = 14
    
    So 2x2x7 has rank <= 11 + 14 = 25
    
    And 2x4x7 has rank <= 2 * 25 = 50 (using two 2x2x7 blocks)
    
    But we can do better by sharing some computations.
    """
    # Let's build this explicitly.
    # 
    # A is 2x4, B is 4x7, C is 2x7
    # Split B's rows into B_top (2x7) and B_bot (2x7)
    # Split A's cols into A_left (2x2) and A_right (2x2)
    #
    # C = A_left * B_top + A_right * B_bot
    #
    # Now for each 2x2x7 multiplication, split the 7 into 3+4:
    # A_left * B_top = A_left * B_top_left + A_left * B_top_right
    # where B_top_left is 2x3, B_top_right is 2x4
    #
    # Similarly for A_right * B_bot.
    #
    # For 2x2x3: split 3 into 1+2
    # A * B_2x3 = A * B_2x1 + A * B_2x2
    # Rank(2x2x1) = 4, Rank(2x2x2) = 7
    #
    # For 2x2x4: split 4 into 2+2
    # A * B_2x4 = A * B_2x2_left + A * B_2x2_right
    # Rank = 7 + 7 = 14
    #
    # Total rank estimate: 2 * (4 + 7 + 7 + 7) = 2 * 25 = 50
    
    # Let's construct this step by step.
    rank = 50
    U = np.zeros((8, rank))   # 2*4 = 8
    V = np.zeros((28, rank))  # 4*7 = 28
    W = np.zeros((14, rank))  # 7*2 = 14
    
    # We'll fill in the factors for each sub-block.
    # Index tracking
    col = 0
    
    # Helper: add a 2x2x2 Strassen block
    def add_strassen_2x2x2(u_offset, v_offset, w_offset, col_start):
        """Add a 2x2x2 Strassen block starting at col_start."""
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2x2()
        # u_offset: (i, j) offset in U
        # v_offset: (j, k) offset in V
        # w_offset: (k, i) offset in W
        
        for r in range(7):
            for ui in range(2):
                for uj in range(2):
                    if U_s[ui * 2 + uj, r] != 0:
                        U[(u_offset[0] + ui) * 2 + (u_offset[1] + uj), col_start + r] = U_s[ui * 2 + uj, r]
            for vj in range(2):
                for vk in range(2):
                    if V_s[vj * 2 + vk, r] != 0:
                        V[(v_offset[0] + vj) * 2 + (v_offset[1] + vk), col_start + r] = V_s[vj * 2 + vk, r]
            for wk in range(2):
                for wi in range(2):
                    if W_s[wk * 2 + wi, r] != 0:
                        W[(w_offset[0] + wk) * 2 + (w_offset[1] + wi), col_start + r] = W_s[wk * 2 + wi, r]
        col = col_start + 7
    
    # Helper: add a 2x2x1 block
    def add_strassen_2x2x1(u_offset, v_offset, w_offset, col_start):
        """Add a 2x2x1 block starting at col_start."""
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2x1()
        
        for r in range(4):
            for ui in range(2):
                for uj in range(2):
                    if U_s[ui * 2 + uj, r] != 0:
                        U[(u_offset[0] + ui) * 2 + (u_offset[1] + uj), col_start + r] = U_s[ui * 2 + uj, r]
            for vj in range(2):
                for vk in range(1):
                    if V_s[vj * 1 + vk, r] != 0:
                        V[(v_offset[0] + vj) * 1 + (v_offset[1] + vk), col_start + r] = V_s[vj * 1 + vk, r]
            for wk in range(1):
                for wi in range(2):
                    if W_s[wk * 2 + wi, r] != 0:
                        W[(w_offset[0] + wk) * 2 + (w_offset[1] + wi), col_start + r] = W_s[wk * 2 + wi, r]
        col = col_start + 4
    
    # Now let's build the full decomposition.
    # 
    # C = A_left * B_top + A_right * B_bot
    #
    # A_left is rows 0-1, cols 0-1 of A (2x2)
    # A_right is rows 0-1, cols 2-3 of A (2x2)
    # B_top is rows 0-1 of B (2x7)
    # B_bot is rows 2-3 of B (2x7)
    #
    # For A_left * B_top (2x2x7):
    # Split B_top into B_top_left (2x3, cols 0-2) and B_top_right (2x4, cols 3-6)
    #
    # A_left * B_top_left (2x2x3):
    # Split B_top_left into B_tl1 (2x1, col 0) and B_tl2 (2x2, cols 1-2)
    #
    # A_left * B_top_right (2x2x4):
    # Split B_top_right into B_tr1 (2x2, cols 3-4) and B_tr2 (2x2, cols 5-6)
    
    # Block 1: A_left * B_tl1 (2x2x1)
    # A_left: u_offset = (0, 0)
    # B_tl1: v_offset = (0, 0) in the 2x7 block, so rows 0-1, col 0
    # W: w_offset = (0, 0) in the 7x2 block, so col 0
    add_strassen_2x2x1((0, 0), (0, 0), (0, 0), col)
    
    # Block 2: A_left * B_tl2 (2x2x2)
    # B_tl2: rows 0-1, cols 1-2
    add_strassen_2x2x2((0, 0), (0, 1), (1, 0), col)
    
    # Block 3: A_left * B_tr1 (2x2x2)
    # B_tr1: rows 0-1, cols 3-4
    add_strassen_2x2x2((0, 0), (0, 3), (3, 0), col)
    
    # Block 4: A_left * B_tr2 (2x2x2)
    # B_tr2: rows 0-1, cols 5-6
    add_strassen_2x2x2((0, 0), (0, 5), (5, 0), col)
    
    # Now for A_right * B_bot (2x2x7):
    # A_right: u_offset = (0, 2)
    # B_bot: rows 2-3
    #
    # Split B_bot into B_bot_left (2x3, cols 0-2) and B_bot_right (2x4, cols 3-6)
    
    # Block 5: A_right * B_bl1 (2x2x1)
    # B_bl1: rows 2-3, col 0
    add_strassen_2x2x1((0, 2), (2, 0), (0, 0), col)
    
    # Block 6: A_right * B_bl2 (2x2x2)
    # B_bl2: rows 2-3, cols 1-2
    add_strassen_2x2x2((0, 2), (2, 1), (1, 0), col)
    
    # Block 7: A_right * B_br1 (2x2x2)
    # B_br1: rows 2-3, cols 3-4
    add_strassen_2x2x2((0, 2), (2, 3), (3, 0), col)
    
    # Block 8: A_right * B_br2 (2x2x2)
    # B_br2: rows 2-3, cols 5-6
    add_strassen_2x2x2((0, 2), (2, 5), (5, 0), col)
    
    # Verify
    if _verify(U, V, W, 2, 4, 7):
        return U, V, W
    
    # If verification fails, fall back to schoolbook
    return None

def _construct_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try Strassen-like decomposition
    result = _construct_strassen_2x4x7(n, m, p)
    if result is not None:
        U, V, W = result
        best_rank = U.shape[1]
    else:
        # Fall back to schoolbook
        U, V, W = _construct_schoolbook(n, m, p)
        best_rank = n * m * p
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
