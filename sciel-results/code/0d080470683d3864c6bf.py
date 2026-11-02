"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    max_err = 0.0
    # Check all valid (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                wc = k * n + i
                val = 0.0
                for c in range(rank):
                    val += U_arr[a, c] * V_arr[b, c] * W_arr[wc, c]
                if abs(val - 1.0) > 1e-9:
                    max_err = max(max_err, abs(val - 1.0))
    
    # Check invalid entries are 0
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j2 = b // p
            k = b % p
            for wc in range(p * n):
                k2 = wc // n
                i2 = wc % n
                if j != j2 or k != k2 or i != i2:
                    val = 0.0
                    for c in range(rank):
                        val += U_arr[a, c] * V_arr[b, c] * W_arr[wc, c]
                    if abs(val) > 1e-9:
                        max_err = max(max_err, abs(val))
    
    return max_err < 1e-9

def _check_half_integer(arr):
    """Check all entries are integer or half-integer."""
    for val in arr.flat:
        if abs(val * 2 - round(val * 2)) > 1e-9:
            return False
    return True

def _to_list(arr):
    """Convert numpy array to list with proper integer/half-integer types."""
    result = []
    for row in arr:
        row_list = []
        for val in row:
            if abs(val - round(val)) < 1e-9:
                row_list.append(int(round(val)))
            elif abs(val * 2 - round(val * 2)) < 1e-9:
                row_list.append(round(val * 2) / 2)
            else:
                row_list.append(val)
        result.append(row_list)
    return result

def _build_strassen_2x2x2():
    """
    Build Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
    A is 2x2 indexed as [A00, A01, A10, A11]
    B is 2x2 indexed as [B00, B01, B10, B11]
    C is 2x2 indexed as [C00, C01, C10, C11]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1  # C00 += 1, C11 += 1
    
    # M2: (A10+A11)B00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1  # C10 += 1
    
    # M3: A00(B10-B11)
    U[0, 2] += 1
    V[2, 2] += 1; V[3, 2] += -1
    W[1, 2] += 1  # C01 += 1
    
    # M4: A11(B00-B10)
    U[3, 3] += 1
    V[0, 3] += 1; V[2, 3] += -1
    W[0, 3] += 1; W[2, 3] += 1  # C00 += 1, C10 += 1
    
    # M5: (A00+A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] += -1; W[1, 4] += 1  # C00 -= 1, C01 += 1
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] += 1; U[0, 5] += -1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1  # C11 += 1
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] += 1; U[3, 6] += -1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1  # C00 += 1
    
    return U, V, W

def _build_4x4x6_strassen():
    """
    Build a rank-84 decomposition for 4x4x6 matrix multiplication
    using Strassen on 2x2x2 sub-blocks.
    
    A (4x4) split into four 2x2 blocks:
    A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    
    B (4x6) split into two 2x6 blocks:
    B_top = B[0:2, :], B_bot = B[2:4, :]
    
    C (4x6):
    C_top = A11*B_top + A12*B_bot  (2x6)
    C_bot = A21*B_top + A22*B_bot  (2x6)
    
    Each 2x2x6 multiplication uses Strassen on three 2x2x2 sub-blocks (rank 21).
    Total rank: 4 * 21 = 84.
    """
    n, m, p = 4, 4, 6
    rank = 84
    
    U = np.zeros((n * m, rank), dtype=float)  # 16 x 84
    V = np.zeros((m * p, rank), dtype=float)  # 24 x 84
    W = np.zeros((p * n, rank), dtype=float)  # 24 x 84
    
    # Get the base Strassen 2x2x2 decomposition
    U_s, V_s, W_s = _build_strassen_2x2x2()
    
    def place_2x2x6(a_r0, a_c0, b_r0, c_r0, rank_off):
        """
        Place a 2x2x6 Strassen decomposition.
        a_r0, a_c0: top-left of A's 2x2 block
        b_r0: top row of B's 2x6 block
        c_r0: top row of C's 2x6 block (which is also the row offset in W)
        rank_off: starting rank index
        """
        for block in range(3):
            # Column offset within the 2x6 block
            col_off = block * 2
            r_off = rank_off + block * 7
            
            # For this 2x2x2 sub-block:
            # A local indices: 0=A00, 1=A01, 2=A10, 3=A11
            # Global A index: (a_r0 + dr) * 4 + (a_c0 + dc)
            a_map = {0: (a_r0, a_c0), 1: (a_r0, a_c0+1),
                     2: (a_r0+1, a_c0), 3: (a_r0+1, a_c0+1)}
            
            # B local indices for 2x2 at column offset col_off:
            # B[0, col_off], B[0, col_off+1], B[1, col_off], B[1, col_off+1]
            # Global B index: (b_r0 + dr) * 6 + (col_off + dc)
            b_map = {0: (b_r0, col_off), 1: (b_r0, col_off+1),
                     2: (b_r0+1, col_off), 3: (b_r0+1, col_off+1)}
            
            # C local indices for 2x2 at column offset col_off:
            # C[0, col_off], C[0, col_off+1], C[1, col_off], C[1, col_off+1]
            # Global C index: (c_r0 + dr) * 6 + (col_off + dc)
            # W index: col * n + row = (col_off + dc) * 4 + (c_r0 + dr)
            c_map = {0: (c_r0, col_off), 1: (c_r0, col_off+1),
                     2: (c_r0+1, col_off), 3: (c_r0+1, col_off+1)}
            
            for t in range(7):
                # U contributions
                for la in range(4):
                    if U_s[la, t] != 0:
                        gr, gc = a_map[la]
                        U[gr * 4 + gc, r_off + t] += U_s[la, t]
                
                # V contributions
                for lb in range(4):
                    if V_s[lb, t] != 0:
                        gr, gc = b_map[lb]
                        V[gr * 6 + gc, r_off + t] += V_s[lb, t]
                
                # W contributions
                for lc in range(4):
                    if W_s[lc, t] != 0:
                        gr, gc = c_map[lc]
                        # W index = col * n + row
                        W[gc * 4 + gr, r_off + t] += W_s[lc, t]
    
    # Four 2x2x6 decompositions:
    # 1. A11 * B_top -> C_top: A[0:2,0:2] * B[0:2,:] -> C[0:2,:]
    place_2x2x6(0, 0, 0, 0, 0)
    
    # 2. A12 * B_bot -> C_top: A[0:2,2:4] * B[2:4,:] -> C[0:2,:]
    place_2x2x6(0, 2, 2, 0, 21)
    
    # 3. A21 * B_top -> C_bot: A[2:4,0:2] * B[0:2,:] -> C[2:4,:]
    place_2x2x6(2, 0, 0, 2, 42)
    
    # 4. A22 * B_bot -> C_bot: A[2:4,2:4] * B[2:4,:] -> C[2:4,:]
    place_2x2x6(2, 2, 2, 2, 63)
    
    return U, V, W

def _build_schoolbook():
    """Build the schoolbook (rank 96) decomposition."""
    n, m, p = 4, 4, 6
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before 150s kill
    
    # Build schoolbook as fallback
    U_sb, V_sb, W_sb = _build_schoolbook()
    best_result = {"U": _to_list(U_sb), "V": _to_list(V_sb), "W": _to_list(W_sb)}
    best_rank = n * m * p
    
    # Try Strassen-based rank-84 decomposition
    try:
        U84, V84, W84 = _build_4x4x6_strassen()
        
        if _verify_decomposition(U84, V84, W84, n, m, p):
            if _check_half_integer(U84) and _check_half_integer(V84) and _check_half_integer(W84):
                return {"U": _to_list(U84), "V": _to_list(V84), "W": _to_list(W84)}
    except Exception:
        pass
    
    # Fallback to schoolbook
    return best_result
# EVOLVE-BLOCK-END
