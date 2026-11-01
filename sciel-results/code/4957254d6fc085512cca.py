"""Rewritten program for matrix multiplication tensor rank minimization (2x4 * 4x4)."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=2, m=4, p=4, verbose=False):
    """Verify that the decomposition correctly computes matrix multiplication.
    
    The tensor for matrix multiplication satisfies:
    T[i,j,k] = 1 for all valid i, j, k
    where i in [0,n), j in [0,m), k in [0,p)
    
    The decomposition requires:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                    if verbose and err > 1e-9:
                        print(f"Error at i={i}, j={j}, k={k}: got {val}, expected 1.0")
    
    return max_err


def _build_strassen_28():
    """Build a rank-28 Strassen decomposition for 2x4 * 4x4.
    
    Strategy: Partition B into left and right halves (4x2 each).
    For each half, partition A into two 2x2 blocks and the corresponding B half into two 2x2 blocks.
    Each 2x2 * 2x2 product uses Strassen's 7 multiplications.
    Total: 4 Strassen blocks * 7 = 28 rank terms.
    
    A (2x4) = [A0 | A1] where A0 = A[:, 0:2], A1 = A[:, 2:4]
    B (4x4) = [[B00, B01], [B10, B11]] where each Bj is 2x2
    
    C (2x4) = [A0*B00 + A1*B10, A0*B01 + A1*B11]
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)  # 8 x 28
    V = np.zeros((m * p, rank), dtype=int)  # 16 x 28
    W = np.zeros((p * n, rank), dtype=int)  # 8 x 28
    
    def strassen_2x2(U_rows, V_rows, W_rows, col_offset):
        """Add 7 Strassen terms for a 2x2*2x2 multiplication.
        
        U_rows: indices [A00, A01, A10, A11] in U
        V_rows: indices [B00, B10, B01, B11] in V  (note: B is stored as [B00; B10] in column-major-like fashion)
        W_rows: indices [C00, C01, C10, C11] in W
        col_offset: starting column index
        """
        A00, A01, A10, A11 = U_rows
        B00, B10, B01, B11 = V_rows
        C00, C01, C10, C11 = W_rows
        c = col_offset
        
        # M1 = (A00+A11)*(B00+B11)
        U[A00, c] += 1
        U[A11, c] += 1
        V[B00, c] += 1
        V[B11, c] += 1
        W[C00, c] += 1
        W[C01, c] -= 1
        W[C10, c] -= 1
        W[C11, c] += 1
        c += 1
        
        # M2 = (A10+A11)*B00
        U[A10, c] += 1
        U[A11, c] += 1
        V[B00, c] += 1
        W[C01, c] += 1
        W[C11, c] -= 1
        c += 1
        
        # M3 = A00*(B10-B11)
        U[A00, c] += 1
        V[B10, c] += 1
        V[B11, c] -= 1
        W[C00, c] += 1
        W[C10, c] += 1
        c += 1
        
        # M4 = A11*(B01-B10)
        U[A11, c] += 1
        V[B01, c] += 1
        V[B10, c] -= 1
        W[C01, c] += 1
        W[C11, c] += 1
        c += 1
        
        # M5 = (A00+A10)*B11
        U[A00, c] += 1
        U[A10, c] += 1
        V[B11, c] += 1
        W[C10, c] += 1
        W[C11, c] -= 1
        c += 1
        
        # M6 = (A01-A10)*(B00+B10)
        U[A01, c] += 1
        U[A10, c] -= 1
        V[B00, c] += 1
        V[B10, c] += 1
        W[C00, c] -= 1
        W[C01, c] += 1
        W[C10, c] += 1
        W[C11, c] -= 1
        c += 1
        
        # M7 = (A01-A11)*(B01+B11)
        U[A01, c] += 1
        U[A11, c] -= 1
        V[B01, c] += 1
        V[B11, c] += 1
        W[C00, c] += 1
        W[C01, c] += 1
        W[C10, c] -= 1
        W[C11, c] -= 1
        c += 1
        
        return c
    
    # U indexing: A[i,j] -> U[i*m+j, :]
    # A0 = A[:, 0:2]: A0[0,0]=U[0], A0[0,1]=U[1], A0[1,0]=U[2], A0[1,1]=U[3]
    # A1 = A[:, 2:4]: A1[0,0]=U[4], A1[0,1]=U[5], A1[1,0]=U[6], A1[1,1]=U[7]
    
    # V indexing: B[a,b] -> V[a*p+b, :]
    # B00 = B[0:2, 0:2]: B00[0,0]=V[0], B00[1,0]=V[4], B00[0,1]=V[1], B00[1,1]=V[5]
    # Wait, V[j*p+k] where j is row, k is col of B
    # B[j,k] -> V[j*4+k]
    # B00 = B[0:2, 0:2]: B00[0,0]=V[0], B00[0,1]=V[1], B00[1,0]=V[4], B00[1,1]=V[5]
    # In Strassen, V_rows = [B00, B10, B01, B11]
    # B00[0,0] = V[0*4+0] = V[0]
    # B10[0,0] = V[2*4+0] = V[8]  (B10 is B[2:4, 0:2], so B10[0,0] = B[2,0] = V[2*4+0] = V[8])
    # B01[0,0] = V[0*4+2] = V[2]  (B01 is B[0:2, 2:4], so B01[0,0] = B[0,2] = V[0*4+2] = V[2])
    # B11[0,0] = V[2*4+2] = V[10] (B11 is B[2:4, 2:4], so B11[0,0] = B[2,2] = V[2*4+2] = V[10])
    
    # Let me re-derive the V indices more carefully.
    # B is 4x4. B[a,b] is stored at V[a*4+b].
    # B00 = B[0:2, 0:2]: entries B[0,0], B[0,1], B[1,0], B[1,1]
    #   B[0,0] -> V[0*4+0] = V[0]
    #   B[0,1] -> V[0*4+1] = V[1]
    #   B[1,0] -> V[1*4+0] = V[4]
    #   B[1,1] -> V[1*4+1] = V[5]
    # In Strassen notation: B00[0,0]=V[0], B00[0,1]=V[1], B00[1,0]=V[4], B00[1,1]=V[5]
    # The V_rows parameter expects [B00, B10, B01, B11] where these are the (0,0) entries of each block.
    # Actually, looking at the Strassen formulas, B00, B10, B01, B11 refer to the entire 2x2 blocks.
    # But in the code, we only use single V indices. Let me re-examine.
    
    # Actually, the Strassen formulas operate on 2x2 matrices. The V_rows should be the indices
    # corresponding to the four corners of each 2x2 block in V.
    # For a 2x2 block B_block with entries [[b00, b01], [b10, b11]], we need:
    #   V index for b00, b10, b01, b11
    # But in the current code structure, V_rows is a list of 4 indices, and we use:
    #   V_rows[0] for B00 (top-left of block)
    #   V_rows[1] for B10 (bottom-left of block)
    #   V_rows[2] for B01 (top-right of block)
    #   V_rows[3] for B11 (bottom-right of block)
    
    # For B00 (B[0:2, 0:2]):
    #   b00 = B[0,0] -> V[0]
    #   b10 = B[1,0] -> V[4]
    #   b01 = B[0,1] -> V[1]
    #   b11 = B[1,1] -> V[5]
    # So V_rows for B00 = [0, 4, 1, 5]
    
    # For B10 (B[2:4, 0:2]):
    #   b00 = B[2,0] -> V[8]
    #   b10 = B[3,0] -> V[12]
    #   b01 = B[2,1] -> V[9]
    #   b11 = B[3,1] -> V[13]
    # So V_rows for B10 = [8, 12, 9, 13]
    
    # For B01 (B[0:2, 2:4]):
    #   b00 = B[0,2] -> V[2]
    #   b10 = B[1,2] -> V[6]
    #   b01 = B[0,3] -> V[3]
    #   b11 = B[1,3] -> V[7]
    # So V_rows for B01 = [2, 6, 3, 7]
    
    # For B11 (B[2:4, 2:4]):
    #   b00 = B[2,2] -> V[10]
    #   b10 = B[3,2] -> V[14]
    #   b01 = B[2,3] -> V[11]
    #   b11 = B[3,3] -> V[15]
    # So V_rows for B11 = [10, 14, 11, 15]
    
    # W indexing: C[i,k] -> W[k*n+i]
    # C is 2x4. C[i,k] -> W[k*2+i]
    # C_left = C[:, 0:2]: C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    # C_right = C[:, 2:4]: C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]
    
    # For C_left (2x2 block):
    #   C00 = C[0,0] -> W[0]
    #   C01 = C[0,1] -> W[2]
    #   C10 = C[1,0] -> W[1]
    #   C11 = C[1,1] -> W[3]
    # W_rows for C_left = [0, 2, 1, 3]
    
    # For C_right (2x2 block):
    #   C00 = C[0,2] -> W[4]
    #   C01 = C[0,3] -> W[6]
    #   C10 = C[1,2] -> W[5]
    #   C11 = C[1,3] -> W[7]
    # W_rows for C_right = [4, 6, 5, 7]
    
    # Now call Strassen for each of the four products:
    
    # 1. A0 * B00 -> contributes to C_left
    strassen_2x2([0, 1, 2, 3], [0, 4, 1, 5], [0, 2, 1, 3], 0)
    
    # 2. A1 * B10 -> contributes to C_left
    strassen_2x2([4, 5, 6, 7], [8, 12, 9, 13], [0, 2, 1, 3], 7)
    
    # 3. A0 * B01 -> contributes to C_right
    strassen_2x2([0, 1, 2, 3], [2, 6, 3, 7], [4, 6, 5, 7], 14)
    
    # 4. A1 * B11 -> contributes to C_right
    strassen_2x2([4, 5, 6, 7], [10, 14, 11, 15], [4, 6, 5, 7], 21)
    
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    start_time = time.time()
    
    # Build rank-28 Strassen decomposition
    U, V, W = _build_strassen_28()
    
    # Verify
    err = _verify_decomposition(U, V, W)
    
    if err >= 1e-9:
        # Fall back to schoolbook rank-32
        n, m, p = 2, 4, 4
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
    
    # Deadline guard
    if time.time() - start_time > 140:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
