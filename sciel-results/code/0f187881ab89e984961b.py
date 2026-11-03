"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    def schoolbook(n, m, p):
        """Schoolbook decomposition for n x m * m x p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W
    
    def strassen_combine(n, m, p, U_sub, V_sub, W_sub):
        """Combine 7 sub-decompositions using Strassen for n x m * m x p where all even."""
        n2, m2, p2 = n // 2, m // 2, p // 2
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        terms = [
            {'A': [(11, 1), (22, 1)], 'B': [(11, 1), (22, 1)], 'C': [(11, 1), (22, 1)]},
            {'A': [(21, 1), (22, 1)], 'B': [(11, 1)], 'C': [(21, 1), (22, -1)]},
            {'A': [(11, 1)], 'B': [(12, 1), (22, -1)], 'C': [(12, 1), (22, 1)]},
            {'A': [(22, 1)], 'B': [(21, 1), (11, -1)], 'C': [(11, 1), (21, 1)]},
            {'A': [(11, 1), (12, 1)], 'B': [(22, 1)], 'C': [(11, -1), (12, 1)]},
            {'A': [(21, 1), (11, -1)], 'B': [(11, 1), (12, 1)], 'C': [(22, 1)]},
            {'A': [(12, 1), (22, -1)], 'B': [(21, 1), (22, 1)], 'C': [(11, 1)]}
        ]
        
        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_idx in range(n2 * m2):
                    val = U_sub[u_idx][c]
                    if val == 0:
                        continue
                    i = u_idx // m2
                    j = u_idx % m2
                    for block, coeff in term['A']:
                        if block == 11:
                            g_u = i*m + j
                        elif block == 12:
                            g_u = i*m + (j+m2)
                        elif block == 21:
                            g_u = (i+n2)*m + j
                        elif block == 22:
                            g_u = (i+n2)*m + (j+m2)
                        else:
                            continue
                        U[g_u][base_col + c] += val * coeff
                
                for v_idx in range(m2 * p2):
                    val = V_sub[v_idx][c]
                    if val == 0:
                        continue
                    j = v_idx // p2
                    k = v_idx % p2
                    for block, coeff in term['B']:
                        if block == 11:
                            g_v = j*p + k
                        elif block == 12:
                            g_v = j*p + (k+p2)
                        elif block == 21:
                            g_v = (j+m2)*p + k
                        elif block == 22:
                            g_v = (j+m2)*p + (k+p2)
                        else:
                            continue
                        V[g_v][base_col + c] += val * coeff
                
                for w_idx in range(p2 * n2):
                    val = W_sub[w_idx][c]
                    if val == 0:
                        continue
                    k = w_idx // n2
                    i = w_idx % n2
                    for block, coeff in term['C']:
                        if block == 11:
                            g_w = k*n + i
                        elif block == 12:
                            g_w = (k+p2)*n + i
                        elif block == 21:
                            g_w = k*n + (i+n2)
                        elif block == 22:
                            g_w = (k+p2)*n + (i+n2)
                        else:
                            continue
                        W[g_w][base_col + c] += val * coeff
        
        return U, V, W
    
    def build_2x2x3():
        """Build decomposition for 2x2 * 2x3.
        Split B into B_left(2x2) and B_right(2x1).
        C_left = A * B_left: use Strassen for 2x2*2x2 (rank 7)
        C_right = A * B_right: schoolbook for 2x2*2x1 (rank 4)
        Total rank: 11
        """
        # Strassen for 2x2 * 2x2
        U_s, V_s, W_s = schoolbook(2, 2, 2)  # This gives rank 8, but we want Strassen rank 7
        
        # Actually, let's build Strassen 2x2*2x2 directly
        # For 2x2*2x2, Strassen has rank 7
        U_str = [[0]*7 for _ in range(4)]
        V_str = [[0]*7 for _ in range(4)]
        W_str = [[0]*7 for _ in range(4)]
        
        # Strassen terms for 2x2 * 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U indices: 0=A11, 1=A12, 2=A21, 3=A22
        # V indices: 0=B11, 1=B12, 2=B21, 3=B22
        # W indices: 0=C11, 1=C12, 2=C21, 3=C22
        
        # Term 0 (M1): A: 11(+1),22(+1); B: 11(+1),22(+1); C: 11(+1),22(+1)
        U_str[0][0] = 1; U_str[3][0] = 1
        V_str[0][0] = 1; V_str[3][0] = 1
        W_str[0][0] = 1; W_str[3][0] = 1
        
        # Term 1 (M2): A: 21(+1),22(+1); B: 11(+1); C: 21(+1),22(-1)
        U_str[2][1] = 1; U_str[3][1] = 1
        V_str[0][1] = 1
        W_str[2][1] = 1; W_str[3][1] = -1
        
        # Term 2 (M3): A: 11(+1); B: 12(+1),22(-1); C: 12(+1),22(+1)
        U_str[0][2] = 1
        V_str[1][2] = 1; V_str[3][2] = -1
        W_str[1][2] = 1; W_str[3][2] = 1
        
        # Term 3 (M4): A: 22(+1); B: 21(+1),11(-1); C: 11(+1),21(+1)
        U_str[3][3] = 1
        V_str[2][3] = 1; V_str[0][3] = -1
        W_str[0][3] = 1; W_str[2][3] = 1
        
        # Term 4 (M5): A: 11(+1),12(+1); B: 22(+1); C: 11(-1),12(+1)
        U_str[0][4] = 1; U_str[1][4] = 1
        V_str[3][4] = 1
        W_str[0][4] = -1; W_str[1][4] = 1
        
        # Term 5 (M6): A: 21(+1),11(-1); B: 11(+1),12(+1); C: 22(+1)
        U_str[2][5] = 1; U_str[0][5] = -1
        V_str[0][5] = 1; V_str[1][5] = 1
        W_str[3][5] = 1
        
        # Term 6 (M7): A: 12(+1),22(-1); B: 21(+1),22(+1); C: 11(+1)
        U_str[1][6] = 1; U_str[3][6] = -1
        V_str[2][6] = 1; V_str[3][6] = 1
        W_str[0][6] = 1
        
        # Now for 2x2 * 2x3:
        # A is 2x2, B is 2x3 = [B_left(2x2), B_right(2x1)]
        # C is 2x3 = [C_left(2x2), C_right(2x1)]
        # C_left = A * B_left: use the Strassen 2x2*2x2 decomp (rank 7)
        # C_right = A * B_right: schoolbook 2x2*2x1 (rank 4)
        
        # Full U: shape (2*2, 11) = (4, 11)
        # Full V: shape (2*3, 11) = (6, 11)
        # Full W: shape (3*2, 11) = (6, 11)
        
        U_full = [[0]*11 for _ in range(4)]
        V_full = [[0]*11 for _ in range(6)]
        W_full = [[0]*11 for _ in range(6)]
        
        # Copy Strassen part (columns 0-6)
        for i in range(4):
            for c in range(7):
                U_full[i][c] = U_str[i][c]
        
        # V for B_left (2x2): rows 0,1 correspond to B_left row 0,1; cols 0,1
        # V indices for B: j*p + k where j=0,1 and k=0,1,2
        # B_left: j=0,1; k=0,1 -> V indices: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
        # B_right: j=0,1; k=2 -> V indices: 0*3+2=2, 1*3+2=5
        for i in range(4):
            for c in range(7):
                # V_str[i] corresponds to B_left
                # V_str[0]=B11->V[0], V_str[1]=B12->V[1], V_str[2]=B21->V[3], V_str[3]=B22->V[4]
                pass
        # Map V_str indices to V_full indices for B_left
        v_map = {0: 0, 1: 1, 2: 3, 3: 4}
        for vi, vfi in v_map.items():
            for c in range(7):
                V_full[vfi][c] = V_str[vi][c]
        
        # W for C_left (2x2): W indices k*n+i where k=0,1 and i=0,1
        # W_full indices: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3
        # W_str[0]=C11->W[0], W_str[1]=C12->W[1], W_str[2]=C21->W[2], W_str[3]=C22->W[3]
        for wi in range(4):
            for c in range(7):
                W_full[wi][c] = W_str[wi][c]
        
        # Schoolbook part for C_right = A * B_right (2x2 * 2x1, rank 4)
        # Columns 7-10
        # A indices: 0=A11, 1=A12, 2=A21, 3=A22
        # B_right: row 0 col 2 -> V index 0*3+2=2, row 1 col 2 -> V index 1*3+2=5
        # C_right: row 0 col 2 -> W index 2*2+0=4, row 1 col 2 -> W index 2*2+1=5
        
        for i in range(2):
            for j in range(2):
                for k in range(2):  # k is the column of B_right (only 1 col, but we index as 0)
                    col = 7 + i*2 + j  # 4 terms: (0,0),(0,1),(1,0),(1,1)
                    # Wait, 2x2 * 2x1: A is 2x2, B_right is 2x1
                    # Terms: for each output (i, k_out) where k_out=0 (only 1 col)
                    # Actually C_right is 2x1, so outputs are (0,0) and (1,0)
                    # Each output = sum over j: A[i,j] * B_right[j,0]
                    # So 2 outputs, each with 2 terms = 4 terms total
                    pass
        
        # Let me redo this more carefully
        # C_right[i, 0] = sum_j A[i,j] * B_right[j, 0] for i=0,1
        # Term (i,j): U[i*2+j, col] = 1, V[j*3+2, col] = 1, W[2*2+i, col] = 1
        # where col = 7 + i*2 + j
        
        for i in range(2):
            for j in range(2):
                col = 7 + i*2 + j
                U_full[i*2 + j][col] = 1
                V_full[j*3 + 2][col] = 1
                W_full[2*2 + i][col] = 1
        
        return U_full, V_full, W_full
    
    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                    if abs(sum(prod) - 1.0) > 1e-9:
                        return False
        return True
    
    # Build 2x2 * 2x3 decomposition (rank 11)
    U_sub, V_sub, W_sub = build_2x2x3()
    
    # Verify sub-decomposition
    if not verify(U_sub, V_sub, W_sub, 2, 2, 3):
        # Fallback to schoolbook for 2x2*2x3
        U_sub, V_sub, W_sub = schoolbook(2, 2, 3)
    
    # Now combine using Strassen for 4x4 * 4x6
    U, V, W = strassen_combine(4, 4, 6, U_sub, V_sub, W_sub)
    
    # Verify full decomposition
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
