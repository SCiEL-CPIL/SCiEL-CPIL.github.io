import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W form a valid decomposition."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = 0
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(val - 1) > tol:
                    return False
    return True

def _get_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U, V, W are lists of lists (4x7).
    """
    # Strassen coefficients for 2x2x2
    # Terms (u, v, w) are 0..3
    # u = i*2+k, v = k*2+j, w = j*2+i
    # We define 7 rank-1 terms.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices:
    # A: 00(0), 01(1), 10(2), 11(3)
    # B: 00(0), 01(1), 10(2), 11(3)
    # C: 00(0), 01(1), 10(2), 11(3)
    
    # Term 1: (1,1,1) on A11, A22; B11, B22; C11, C22
    # Actually, let's just use the known rank-7 decomposition.
    # U, V, W columns correspond to the 7 terms.
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1: M1
    # A: 11+22 -> U[0,0]=1, U[3,0]=1
    # B: 11+22 -> V[0,0]=1, V[3,0]=1
    # C: 11+22 -> W[0,0]=1, W[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2
    # A: 21+22 -> U[2,1]=1, U[3,1]=1
    # B: 11 -> V[0,1]=1
    # C: 21 -> W[2,1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 3: M3
    # A: 11 -> U[0,2]=1
    # B: 12-22 -> V[1,2]=1, V[3,2]=-1
    # C: 12 -> W[1,2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 4: M4
    # A: 22 -> U[3,3]=1
    # B: 21-11 -> V[2,3]=1, V[0,3]=-1
    # C: 22 -> W[3,3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[3][3] = 1
    
    # Term 5: M5
    # A: 11+12 -> U[0,4]=1, U[1,4]=1
    # B: 22 -> V[3,4]=1
    # C: 12 -> W[1,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1
    
    # Term 6: M6
    # A: 21-11 -> U[2,5]=1, U[0,5]=-1
    # B: 11+12 -> V[0,5]=1, V[1,5]=1
    # C: 22 -> W[3,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7: M7
    # A: 12-22 -> U[1,6]=1, U[3,6]=-1
    # B: 21+22 -> V[2,6]=1, V[3,6]=1
    # C: 12 -> W[1,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1
    
    return U, V, W

def _get_strassen_4x4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication with rank 49.
    Uses 7 Strassen steps, each of rank 7.
    """
    U2, V2, W2 = _get_strassen_2x2x2()
    
    # U, V, W for 4x4x4
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # Strassen steps for 4x4:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]]
    
    # Step 1: M1 = (A11+A22)(B11+B22)
    # Step 2: M2 = (A21+A22)B11
    # Step 3: M3 = A11(B12-B22)
    # Step 4: M4 = A22(B21-B11)
    # Step 5: M5 = (A11+A12)B22
    # Step 6: M6 = (A21-A11)(B11+B12)
    # Step 7: M7 = (A12-A22)(B21+B22)
    
    # We need to map the 2x2 indices to 4x4 indices.
    # 4x4 A indices: 0..15.
    # A11: 0..3, A12: 4..7, A21: 8..11, A22: 12..15
    # Same for B and C.
    
    # For each of the 7 steps, we use the 7 rank-1 terms from 2x2x2.
    # So total 49 columns.
    
    for step in range(7):
        for term in range(7):
            col = step * 7 + term
            
            # Get the 2x2 factors for this term
            u2 = U2[:, term]
            v2 = V2[:, term]
            w2 = W2[:, term]
            
            # Determine which 2x2 blocks of A, B, C are involved and with what signs
            # Based on the step definition
            
            # Helper to add to U
            def add_U(a_block, sign):
                for i in range(4):
                    U[a_block*4 + i][col] += sign * u2[i]
                    
            def add_V(b_block, sign):
                for i in range(4):
                    V[b_block*4 + i][col] += sign * v2[i]
                    
            def add_W(c_block, sign):
                for i in range(4):
                    W[c_block*4 + i][col] += sign * w2[i]

            if step == 0: # M1
                add_U(0, 1); add_U(3, 1) # A11, A22
                add_V(0, 1); add_V(3, 1) # B11, B22
                add_W(0, 1); add_W(3, 1) # C11, C22
            elif step == 1: # M2
                add_U(2, 1); add_U(3, 1) # A21, A22
                add_V(0, 1) # B11
                add_W(2, 1) # C21
            elif step == 2: # M3
                add_U(0, 1) # A11
                add_V(1, 1); add_V(3, -1) # B12, B22
                add_W(1, 1) # C12
            elif step == 3: # M4
                add_U(3, 1) # A22
                add_V(2, 1); add_V(0, -1) # B21, B11
                add_W(3, 1) # C22
            elif step == 4: # M5
                add_U(0, 1); add_U(1, 1) # A11, A12
                add_V(3, 1) # B22
                add_W(1, 1) # C12
            elif step == 5: # M6
                add_U(2, 1); add_U(0, -1) # A21, A11
                add_V(0, 1); add_V(1, 1) # B11, B12
                add_W(3, 1) # C22
            elif step == 6: # M7
                add_U(1, 1); add_U(3, -1) # A12, A22
                add_V(2, 1); add_V(3, 1) # B21, B22
                add_W(1, 1) # C12
                
    return U, V, W

def _get_schoolbook_4x4x1():
    """
    Returns U, V, W for 4x4x1 matrix multiplication with rank 16.
    """
    U = [[0]*16 for _ in range(16)]
    V = [[0]*16 for _ in range(4)]
    W = [[0]*16 for _ in range(4)]
    
    col = 0
    for i in range(4):
        for k in range(4):
            U[i*4 + k][col] = 1
            V[k][col] = 1
            W[i][col] = 1
            col += 1
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Get factors
    U1, V1, W1 = _get_strassen_4x4x4()
    U2, V2, W2 = _get_strassen_4x4x4()
    U3, V3, W3 = _get_schoolbook_4x4x1()
    
    # Combine
    # U: 16 x 49, 16 x 49, 16 x 16 -> 16 x 114
    U = [row1 + row2 + row3 for row1, row2, row3 in zip(U1, U2, U3)]
    
    # V: 16 x 49, 16 x 49, 4 x 16 -> 36 x 114
    # V1 covers B cols 0-3 (indices 0-3, 9-12, 18-21, 27-30)
    # V2 covers B cols 4-7 (indices 4-7, 13-16, 22-25, 31-34)
    # V3 covers B col 8 (indices 8, 17, 26, 35)
    
    V_total = [[0]*114 for _ in range(36)]
    
    # Place V1
    for k in range(4):
        for j in range(4):
            v_idx = k * 9 + j
            for c in range(49):
                V_total[v_idx][c] = V1[k*4 + j][c]
                
    # Place V2
    for k in range(4):
        for j in range(4):
            v_idx = k * 9 + (j + 4)
            for c in range(49, 98):
                V_total[v_idx][c] = V2[k*4 + (j-4)][c-49] # Wait, V2 indices are local 0-3
                # Correction: V2 is for 4x4x4, so its local j is 0-3.
                # In the full V, j is 4-7.
                # So V2[k*4 + j_local] maps to V_total[k*9 + j_global]
                # Let's re-do this carefully.
    
    # Reset V_total
    V_total = [[0]*114 for _ in range(36)]
    
    # V1: Local j 0-3 -> Global j 0-3
    for k in range(4):
        for j_local in range(4):
            j_global = j_local
            v_idx = k * 9 + j_global
            for c in range(49):
                V_total[v_idx][c] = V1[k*4 + j_local][c]
                
    # V2: Local j 0-3 -> Global j 4-7
    for k in range(4):
        for j_local in range(4):
            j_global = 4 + j_local
            v_idx = k * 9 + j_global
            for c in range(49, 98):
                V_total[v_idx][c] = V2[k*4 + j_local][c-49]
                
    # V3: Local j 0 -> Global j 8
    for k in range(4):
        j_global = 8
        v_idx = k * 9 + j_global
        for c in range(98, 114):
            V_total[v_idx][c] = V3[k][c-98]
            
    # W: 16 x 49, 16 x 49, 4 x 16 -> 36 x 114
    # W1 covers C cols 0-3 (rows 0-15)
    # W2 covers C cols 4-7 (rows 16-31)
    # W3 covers C col 8 (rows 32-35)
    
    W_total = [[0]*114 for _ in range(36)]
    
    # W1: Local j 0-3 -> Global j 0-3
    for j_local in range(4):
        for i in range(4):
            w_idx = j_local * 4 + i
            for c in range(49):
                W_total[w_idx][c] = W1[j_local*4 + i][c]
                
    # W2: Local j 0-3 -> Global j 4-7
    for j_local in range(4):
        for i in range(4):
            w_idx = (4 + j_local) * 4 + i
            for c in range(49, 98):
                W_total[w_idx][c] = W2[j_local*4 + i][c-49]
                
    # W3: Local j 0 -> Global j 8
    for i in range(4):
        w_idx = 8 * 4 + i
        for c in range(98, 114):
            W_total[w_idx][c] = W3[i][c-98]
            
    result = {"U": U, "V": V_total, "W": W_total}
    
    # Verify
    if _verify(U, V_total, W_total, n, m, p):
        return result
    else:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U_s = [[0] * rank for _ in range(n * m)]
        V_s = [[0] * rank for _ in range(m * p)]
        W_s = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_s[i * m + k][col] = 1
                    V_s[k * p + j][col] = 1
                    W_s[j * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
