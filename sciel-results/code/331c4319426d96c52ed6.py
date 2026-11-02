import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n=3, m=3, p=4, tol=1e-9):
    """Verify rank-r decomposition."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape != (n * m, rank) or V_arr.shape != (m * p, rank) or W_arr.shape != (p * n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n=3, m=3, p=4):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2_local():
    """
    Return rank-7 Strassen decomposition for 2x2 * 2x2.
    Indices:
    U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    W rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
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
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[2][5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _combine_decompositions(decomps):
    """
    Combine a list of decompositions into one.
    Each decomp is a dict with U, V, W and a mapping function or explicit indices.
    Here we assume we pass in the final global matrices and just add them.
    """
    # This function is a placeholder, we will do the addition in construct()
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Initialize global matrices with rank 34
    total_rank = 34
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a local decomposition to global
    def add_decomp(local_decomp, i_map, j_map, k_map, offset):
        # local_decomp: dict with U, V, W of shapes (4, r), (4, r), (4, r)
        r = len(local_decomp['U'][0])
        for li in range(4):
            gi = i_map[li // 2][li % 2] # li: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Wait, li // 2 is i, li % 2 is j?
            # 0: 0,0 -> 0,0
            # 1: 0,1 -> 0,1
            # 2: 1,0 -> 1,0
            # 3: 1,1 -> 1,1
            # Yes.
            gi = i_map[li // 2][li % 2]
            for c in range(r):
                U[gi * m + 0][offset + c] = 0 # Placeholder, need j too
        # This mapping is tricky because U depends on i and j.
        # Let's do it explicitly for each block.

    # Block 1: A00 B00 -> C00
    # A00: i in {0,1}, j in {0,1}
    # B00: j in {0,1}, k in {0,1}
    # C00: k in {0,1}, i in {0,1}
    # Local index li for U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Global U index: i*3 + j
    # Local index li for V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Global V index: j*4 + k
    # Local index li for W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Global W index: k*3 + i
    
    strassen = _strassen_2x2x2_local()
    r_s = len(strassen['U'][0])
    
    # Map local (i,j) to global U row
    def u_idx(i, j): return i * 3 + j
    def v_idx(j, k): return j * 4 + k
    def w_idx(k, i): return k * 3 + i
    
    # Block 1
    for li in range(4):
        li_i = li // 2
        li_j = li % 2
        gu_row = u_idx(li_i, li_j)
        for c in range(r_s):
            U[gu_row][col_offset + c] = strassen['U'][li][c]
            
    for li in range(4):
        li_j = li // 2
        li_k = li % 2
        gv_row = v_idx(li_j, li_k)
        for c in range(r_s):
            V[gv_row][col_offset + c] = strassen['V'][li][c]
            
    for li in range(4):
        li_k = li // 2
        li_i = li % 2
        gw_row = w_idx(li_k, li_i)
        for c in range(r_s):
            W[gw_row][col_offset + c] = strassen['W'][li][c]
            
    col_offset += r_s
    
    # Block 2: A00 B01 -> C01
    # A00: i in {0,1}, j in {0,1}
    # B01: j in {0,1}, k in {2,3}
    # C01: k in {2,3}, i in {0,1}
    # Local k' in {0,1} maps to Global k = 2 + k'
    
    for li in range(4):
        li_i = li // 2
        li_j = li % 2
        gu_row = u_idx(li_i, li_j)
        for c in range(r_s):
            U[gu_row][col_offset + c] = strassen['U'][li][c]
            
    for li in range(4):
        li_j = li // 2
        li_k = li % 2
        gk = 2 + li_k
        gv_row = v_idx(li_j, gk)
        for c in range(r_s):
            V[gv_row][col_offset + c] = strassen['V'][li][c]
            
    for li in range(4):
        li_k = li // 2
        li_i = li % 2
        gk = 2 + li_k
        gw_row = w_idx(gk, li_i)
        for c in range(r_s):
            W[gw_row][col_offset + c] = strassen['W'][li][c]
            
    col_offset += r_s
    
    # Block 3: A01 B10 -> C00
    # A01: i in {0,1}, j in {2}
    # B10: j in {2}, k in {0,1}
    # C00: k in {0,1}, i in {0,1}
    # Terms: A[i,2] B[2,k] -> C[i,k]
    # 4 terms
    for i in range(2):
        for k in range(2):
            gu_row = u_idx(i, 2)
            gv_row = v_idx(2, k)
            gw_row = w_idx(k, i)
            U[gu_row][col_offset] = 1
            V[gv_row][col_offset] = 1
            W[gw_row][col_offset] = 1
            col_offset += 1
            
    # Block 4: A01 B11 -> C01
    # A01: i in {0,1}, j in {2}
    # B11: j in {2}, k in {2,3}
    # C01: k in {2,3}, i in {0,1}
    # Terms: A[i,2] B[2,k] -> C[i,k]
    # 4 terms
    for i in range(2):
        for k in range(2, 4):
            gu_row = u_idx(i, 2)
            gv_row = v_idx(2, k)
            gw_row = w_idx(k, i)
            U[gu_row][col_offset] = 1
            V[gv_row][col_offset] = 1
            W[gw_row][col_offset] = 1
            col_offset += 1
            
    # Block 5: A10 B00 -> C10
    # A10: i in {2}, j in {0,1}
    # B00: j in {0,1}, k in {0,1}
    # C10: k in {0,1}, i in {2}
    # Terms: A[2,j] B[j,k] -> C[k,2]
    # 4 terms
    for j in range(2):
        for k in range(2):
            gu_row = u_idx(2, j)
            gv_row = v_idx(j, k)
            gw_row = w_idx(k, 2)
            U[gu_row][col_offset] = 1
            V[gv_row][col_offset] = 1
            W[gw_row][col_offset] = 1
            col_offset += 1
            
    # Block 6: A11 B10 -> C10
    # A11: i in {2}, j in {2}
    # B10: j in {2}, k in {0,1}
    # C10: k in {0,1}, i in {2}
    # Terms: A[2,2] B[2,k] -> C[k,2]
    # 2 terms
    for k in range(2):
        gu_row = u_idx(2, 2)
        gv_row = v_idx(2, k)
        gw_row = w_idx(k, 2)
        U[gu_row][col_offset] = 1
        V[gv_row][col_offset] = 1
        W[gw_row][col_offset] = 1
        col_offset += 1
        
    # Block 7: A10 B01 -> C11
    # A10: i in {2}, j in {0,1}
    # B01: j in {0,1}, k in {2,3}
    # C11: k in {2,3}, i in {2}
    # Terms: A[2,j] B[j,k] -> C[k,2]
    # 4 terms
    for j in range(2):
        for k in range(2, 4):
            gu_row = u_idx(2, j)
            gv_row = v_idx(j, k)
            gw_row = w_idx(k, 2)
            U[gu_row][col_offset] = 1
            V[gv_row][col_offset] = 1
            W[gw_row][col_offset] = 1
            col_offset += 1
            
    # Block 8: A11 B11 -> C11
    # A11: i in {2}, j in {2}
    # B11: j in {2}, k in {2,3}
    # C11: k in {2,3}, i in {2}
    # Terms: A[2,2] B[2,k] -> C[k,2]
    # 2 terms
    for k in range(2, 4):
        gu_row = u_idx(2, 2)
        gv_row = v_idx(2, k)
        gw_row = w_idx(k, 2)
        U[gu_row][col_offset] = 1
        V[gv_row][col_offset] = 1
        W[gw_row][col_offset] = 1
        col_offset += 1
        
    if col_offset != total_rank:
        # Fallback to schoolbook if construction logic failed
        return _schoolbook(n, m, p)
        
    result = {"U": U, "V": V, "W": W}
    
    if _verify_decomposition(U, V, W, n, m, p):
        return result
    else:
        return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
