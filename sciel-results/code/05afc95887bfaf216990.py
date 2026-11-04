"""Matrix multiplication 4x4 * 4x7 using hierarchical tensor train decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_2x2():
    """Rank-7 decomposition for 2x2 * 2x2 -> 2x2.
    U: 4x7, V: 4x7, W: 4x7
    U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_2x2_2x1():
    """Rank-4 schoolbook for 2x2 * 2x1 -> 2x1.
    U: 4x4, V: 2x4, W: 2x4
    U[i*2+j, c] for A[i,j], V[j*1+k, c] for B[j,k], W[k*2+i, c] for C[i,k]
    Here p=1, so B is 2x1, C is 2x1
    """
    n, m, p = 2, 2, 1
    rank = n * m * p  # 4
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2_2x3():
    """Decomposition for 2x2 * 2x3 -> 2x3.
    Split B into 2x2 + 2x1, use Strassen for first part, schoolbook for second.
    Rank = 7 + 4 = 11.
    U: 4x11, V: 6x11, W: 6x11
    U[i*2+j, c] for A[i,j], V[j*3+k, c] for B[j,k], W[k*2+i, c] for C[i,k]
    """
    sU, sV, sW = _strassen_2x2()
    sbU, sbV, sbW = _schoolbook_2x2_2x1()
    
    rank = 7 + 4  # 11
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(6)]
    
    # Part 1: B cols 0-1 (2x2) -> Strassen
    for c in range(7):
        for i in range(4):
            U[i][c] = sU[i][c]
        for i in range(4):
            V[i][c] = sV[i][c]  # V rows 0-3 are B[0,0],B[1,0],B[0,1],B[1,1]
        for i in range(4):
            k_loc = i // 2
            i_loc = i % 2
            W[k_loc * 2 + i_loc][c] = sW[i][c]  # W rows: k=0: i=0,1; k=1: i=2,3
    
    # Part 2: B col 2 (2x1) -> Schoolbook
    for c in range(4):
        for i in range(4):
            U[i][7+c] = sbU[i][c]
        for i in range(2):
            V[4+i][7+c] = sbV[i][c]  # V rows 4-5 are B[0,2],B[1,2]
        for i in range(2):
            k_loc = i // 2
            i_loc = i % 2
            W[2 + k_loc * 2 + i_loc][7+c] = sbW[i][c]  # W rows: k=2: i=0,1
    
    return U, V, W

def _verify(U, V, W):
    """Verify the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                max_err = max(max_err, abs(val - 1))
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Strategy: Use hierarchical decomposition
    # Split i into 2+2, j into 2+2, k into 4+3
    # For each (i_block, j_block): 2x2 x 2x7
    # 2x2 x 2x7 = 2x2 x (2x4 + 2x3)
    # 2x2 x 2x4: use two 2x2x2 Strassen blocks (rank 7 each) = 14
    # 2x2 x 2x3: use Strassen + schoolbook = 7 + 4 = 11
    # Total per (i_block, j_block): 14 + 11 = 25
    # Total rank: 4 * 25 = 100
    
    # Alternative: Try to find a better decomposition
    # Let's use the 2x2x2x3 helper
    sU, sV, sW = _strassen_2x2()
    sbU, sbV, sbW = _schoolbook_2x2_2x1()
    
    total_rank = 100
    
    U = [[0]*total_rank for _ in range(n*m)]   # 16 x 100
    V = [[0]*total_rank for _ in range(m*p)]   # 28 x 100
    W = [[0]*total_rank for _ in range(p*n)]   # 28 x 100
    
    col = 0
    
    for i_b in range(2):
        for j_b in range(2):
            u_off = i_b * 8 + j_b * 2
            v_base_row = j_b * 2
            w_base_row = i_b * 2
            
            # Part 1: B cols 0-1 (2x2) -> Strassen
            for c in range(7):
                for i in range(4):
                    U[u_off + i][col + c] = sU[i][c]
                for i in range(4):
                    V[v_base_row * 7 + i][col + c] = sV[i][c]
                for i in range(4):
                    k_loc = i // 2
                    i_loc = i % 2
                    W[k_loc * 4 + w_base_row + i_loc][col + c] = sW[i][c]
            col += 7
            
            # Part 2: B cols 2-3 (2x2) -> Strassen
            for c in range(7):
                for i in range(4):
                    U[u_off + i][col + c] = sU[i][c]
                for i in range(4):
                    V[v_base_row * 7 + 2 + i][col + c] = sV[i][c]
                for i in range(4):
                    k_loc = i // 2
                    i_loc = i % 2
                    W[(2 + k_loc) * 4 + w_base_row + i_loc][col + c] = sW[i][c]
            col += 7
            
            # Part 3: B cols 4-5 (2x2) -> Strassen
            for c in range(7):
                for i in range(4):
                    U[u_off + i][col + c] = sU[i][c]
                for i in range(4):
                    V[v_base_row * 7 + 4 + i][col + c] = sV[i][c]
                for i in range(4):
                    k_loc = i // 2
                    i_loc = i % 2
                    W[(4 + k_loc) * 4 + w_base_row + i_loc][col + c] = sW[i][c]
            col += 7
            
            # Part 4: B col 6 (2x1) -> Schoolbook 2x2*2x1
            for c in range(4):
                for i in range(4):
                    U[u_off + i][col + c] = sbU[i][c]
                for i in range(2):
                    V[v_base_row * 7 + 6 + i][col + c] = sbV[i][c]
                for i in range(2):
                    k_loc = i // 2
                    i_loc = i % 2
                    W[(6 + k_loc) * 4 + w_base_row + i_loc][col + c] = sbW[i][c]
            col += 4
    
    assert col == 100, f"Expected rank 100, got {col}"
    
    # Verify
    err = _verify(U, V, W)
    if err > 1e-9:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
