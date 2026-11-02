import time
import numpy as np

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 115.0

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target rank 42
    rank = 42
    
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # Strassen decomposition for 2x2x2
    # Local indices:
    # A: i in {0,1}, j in {0,1} -> idx i*2+j
    # B: j in {0,1}, k in {0,1} -> idx j*2+k
    # C: k in {0,1}, i in {0,1} -> idx k*2+i
    
    # We define the 7 terms (U, V, W) coefficients for local indices
    # Term r, U[idx_u, r], V[idx_v, r], W[idx_w, r]
    
    # M1
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1; W[1, 0] -= 1; W[2, 0] += 1
    
    # M2
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1; W[3, 1] += 1
    
    # M3
    U[0, 2] += 1
    V[2, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1; W[3, 2] += 1
    
    # M4
    U[3, 3] += 1
    V[1, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1; W[2, 3] += 1
    
    # M5
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] -= 1; W[1, 4] += 1
    
    # M6
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1; W[2, 5] += 1
    
    # M7
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1; W[1, 6] += 1
    
    # Now map these to the global problem
    # 3 blocks b=0,1,2
    # For each block, 2 sub-problems: Left (A_L, B_T) and Right (A_R, B_B)
    
    col_offset = 0
    
    for b in range(3):
        # Left part: A_L (cols 0,1 of A) * B_T (rows 0,1 of B)
        # A_L global idx: i*4 + j
        # B_T global idx: j*6 + (2b + k)
        # C global idx: (2b + k)*2 + i
        
        for r in range(7):
            # U contributions
            for u_idx in range(4):
                if U[u_idx, r] != 0:
                    i_loc = u_idx // 2
                    j_loc = u_idx % 2
                    global_u_idx = i_loc * 4 + j_loc
                    U[global_u_idx, col_offset + r] += U[u_idx, r]
            
            # V contributions
            for v_idx in range(4):
                if V[v_idx, r] != 0:
                    j_loc = v_idx // 2
                    k_loc = v_idx % 2
                    global_v_idx = j_loc * 6 + (2*b + k_loc)
                    V[global_v_idx, col_offset + r] += V[v_idx, r]
            
            # W contributions
            for w_idx in range(4):
                if W[w_idx, r] != 0:
                    k_loc = w_idx // 2
                    i_loc = w_idx % 2
                    global_w_idx = (2*b + k_loc) * 2 + i_loc
                    W[global_w_idx, col_offset + r] += W[w_idx, r]
        
        col_offset += 7
        
        # Right part: A_R (cols 2,3 of A) * B_B (rows 2,3 of B)
        # A_R global idx: i*4 + (2+j)
        # B_B global idx: (2+j)*6 + (2b + k)
        # C global idx: (2b + k)*2 + i
        
        for r in range(7):
            # U contributions
            for u_idx in range(4):
                if U[u_idx, r] != 0:
                    i_loc = u_idx // 2
                    j_loc = u_idx % 2
                    global_u_idx = i_loc * 4 + (2 + j_loc)
                    U[global_u_idx, col_offset + r] += U[u_idx, r]
            
            # V contributions
            for v_idx in range(4):
                if V[v_idx, r] != 0:
                    j_loc = v_idx // 2
                    k_loc = v_idx % 2
                    global_v_idx = (2 + j_loc) * 6 + (2*b + k_loc)
                    V[global_v_idx, col_offset + r] += V[v_idx, r]
            
            # W contributions
            for w_idx in range(4):
                if W[w_idx, r] != 0:
                    k_loc = w_idx // 2
                    i_loc = w_idx % 2
                    global_w_idx = (2*b + k_loc) * 2 + i_loc
                    W[global_w_idx, col_offset + r] += W[w_idx, r]
        
        col_offset += 7
        
    # Verify
    if _verify_decomposition(n, m, p, U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return _get_schoolbook(n, m, p)

def _verify_decomposition(n, m, p, U, V, W):
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
                # The decomposition should sum to 1 for the diagonal of the multiplication tensor?
                # No, the tensor T[i,j,k] is 1 if the indices correspond to a multiplication term.
                # For matrix multiplication C_ik = sum_j A_ij B_jk
                # The tensor T is such that T[i,j,k] = 1 if we are multiplying A_ij and B_jk to contribute to C_ik.
                # Wait, the standard tensor definition for matrix multiplication is:
                # T[i,j,k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
                # In the decomposition sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k]
                # T[i,j,k] is 1 for all i,j,k? No.
                # The multiplication tensor T has T[i,j,k] = 1 if the indices are valid.
                # Actually, T[i,j,k] is 1 for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}.
                # Because every A[i,j] and B[j,k] pair contributes to C[i,k].
                # So we expect sum_r ... = 1 for all i,j,k.
                
                if s != 1.0:
                    return False
    return True

def _get_schoolbook(n, m, p):
    nm = n * m
    mp = m * p
    pn = p * n
    rank = n * m * p
    
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
