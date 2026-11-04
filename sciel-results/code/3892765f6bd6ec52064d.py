"""Rewritten program for Matrix Multiplication 3,5,7."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """
    Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    A is 2x2, B is 2x2, C = A*B is 2x2.
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U: A coefficients
    U[0][0] = 1; U[3][0] = 1      # M1: A00+A11
    U[2][1] = 1; U[3][1] = 1      # M2: A10+A11
    U[0][2] = 1                    # M3: A00
    U[3][3] = 1                    # M4: A11
    U[0][4] = 1; U[1][4] = 1      # M5: A00+A01
    U[2][5] = 1; U[0][5] = -1     # M6: A10-A00
    U[1][6] = 1; U[3][6] = -1     # M7: A01-A11
    
    # V: B coefficients
    V[0][0] = 1; V[3][0] = 1      # M1: B00+B11
    V[0][1] = 1                    # M2: B00
    V[2][2] = 1; V[3][2] = -1     # M3: B10-B11
    V[1][3] = 1; V[0][3] = -1     # M4: B01-B00
    V[3][4] = 1                    # M5: B11
    V[0][5] = 1; V[2][5] = 1      # M6: B00+B10
    V[1][6] = 1; V[3][6] = 1      # M7: B01+B11
    
    # W: C coefficients (W[k*2+i, r])
    # C[0,0] -> W[0*2+0] = W[0]
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C[0,1] -> W[1*2+0] = W[2]
    W[2][2] = 1; W[2][4] = 1
    # C[1,0] -> W[0*2+1] = W[1]
    W[1][1] = 1; W[1][3] = 1
    # C[1,1] -> W[1*2+1] = W[3]
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, verbose=False):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    errors = 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                val = float(np.sum(U_arr[u_row] * V_arr[v_row] * W_arr[w_row]))
                if abs(val - 1) > 1e-10:
                    errors += 1
                    if verbose and errors <= 10:
                        print(f"  Error at ({i},{j},{k}): got {val}, expected 1")
    
    # Check non-matching indices (j != j2)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for j2 in range(m):
                    if j2 == j:
                        continue
                    u_row = i * m + j
                    v_row = j2 * p + k
                    w_row = k * n + i
                    val = float(np.sum(U_arr[u_row] * V_arr[v_row] * W_arr[w_row]))
                    if abs(val) > 1e-10:
                        errors += 1
                        if verbose and errors <= 10:
                            print(f"  Non-zero error at ({i},{j2},{k}): got {val}")
    
    return errors == 0


def construct():
    """
    Construct a low-rank decomposition for 3x5x7 matrix multiplication.
    Uses Strassen for 2x2x2 blocks to reduce rank below schoolbook 105.
    """
    n, m, p = DIMENSIONS  # 3, 5, 7
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Partition: n=3=2+1, m=5=2+2+1, p=7=2+2+2+1
    n_splits = [0, 2, 3]
    m_splits = [0, 2, 4, 5]
    p_splits = [0, 2, 4, 6, 7]
    
    # Compute total rank
    total_rank = 0
    for ni in range(len(n_splits)-1):
        for mj in range(len(m_splits)-1):
            for pk in range(len(p_splits)-1):
                ns = n_splits[ni+1] - n_splits[ni]
                ms = m_splits[mj+1] - m_splits[mj]
                ps = p_splits[pk+1] - p_splits[pk]
                if ns == 2 and ms == 2 and ps == 2:
                    total_rank += 7
                else:
                    total_rank += ns * ms * ps
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    for ni in range(len(n_splits)-1):
        for mj in range(len(m_splits)-1):
            for pk in range(len(p_splits)-1):
                i0, i1 = n_splits[ni], n_splits[ni+1]
                j0, j1 = m_splits[mj], m_splits[mj+1]
                k0, k1 = p_splits[pk], p_splits[pk+1]
                ns = i1 - i0
                ms = j1 - j0
                ps = k1 - k0
                
                if ns == 2 and ms == 2 and ps == 2:
                    U_s, V_s, W_s = _strassen_2x2x2()
                    for r in range(7):
                        for i in range(2):
                            for j in range(2):
                                U[(i0+i) * m + (j0+j)][col+r] += U_s[i*2+j][r]
                        for j in range(2):
                            for k in range(2):
                                V[(j0+j) * p + (k0+k)][col+r] += V_s[j*2+k][r]
                        for k in range(2):
                            for i in range(2):
                                W[(k0+k) * n + (i0+i)][col+r] += W_s[k*2+i][r]
                    col += 7
                else:
                    for i in range(i0, i1):
                        for j in range(j0, j1):
                            for k in range(k0, k1):
                                U[i * m + j][col] = 1
                                V[j * p + k][col] = 1
                                W[k * n + i][col] = 1
                                col += 1
    
    assert col == total_rank, f"Column count mismatch: {col} != {total_rank}"
    
    # Verify
    if time.time() < deadline:
        if not _verify_decomposition(U, V, W, n, m, p, verbose=True):
            print("Verification failed, falling back to schoolbook")
            rank_school = n * m * p
            U_s = [[0] * rank_school for _ in range(n * m)]
            V_s = [[0] * rank_school for _ in range(m * p)]
            W_s = [[0] * rank_school for _ in range(p * n)]
            col_s = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i * m + j][col_s] = 1
                        V_s[j * p + k][col_s] = 1
                        W_s[k * n + i][col_s] = 1
                        col_s += 1
            return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
