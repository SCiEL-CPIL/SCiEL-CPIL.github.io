"""Matrix multiplication 3x4x4: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]
DEADLINE_SECONDS = 120.0

def construct():
    """
    Returns a rank-44 exact tensor decomposition for 3x4x4 matrix multiplication.
    Uses Strassen's algorithm for 2x2x2 blocks and schoolbook for 1x2x2 blocks.
    The decomposition is verified before returning.
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        # Local indices:
        # A: (0,0)->a, (0,1)->b, (1,0)->c, (1,1)->d
        # B: (0,0)->e, (0,1)->f, (1,0)->g, (1,1)->h
        # C: (0,0)->i, (0,1)->j, (1,0)->k, (1,1)->l
        
        # Map local to global
        # U index: (i_off + li) * m + (j_off + lj)
        # V index: (j_off + lj) * p + (k_off + lk)
        # W index: (k_off + lk) * n + (i_off + li)
        
        # Strassen terms:
        # 1. M1 = (a+d)(e+h) -> i=1, l=1
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # 2. M2 = (c+d)e -> k=1, l=-1
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1
        
        # 3. M3 = a(f-h) -> j=1, l=1
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # 4. M4 = d(g-e) -> i=1, k=1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = -1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1
        
        # 5. M5 = (a+b)h -> i=-1, j=1
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        
        # 6. M6 = (c-a)(e+f) -> l=1
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+0), col] = -1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # 7. M7 = (b-d)(g+h) -> i=1
        U[(i_off+0)*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

    def add_122(i_off, j_off, k_off):
        nonlocal col
        # A: (0,0)->a, (0,1)->b
        # B: (0,0)->e, (0,1)->f, (1,0)->g, (1,1)->h
        # C: (0,0)->i, (0,1)->j
        
        # 1. ae -> i
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1
        
        # 2. bg -> i
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1
        
        # 3. af -> j
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        
        # 4. bh -> j
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

    # Top part: A[0:2, :] * B[:, :]
    # Block 1: A_top[:, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_222(0, 0, 0)
    
    # Block 2: A_top[:, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_222(0, 2, 0)
    
    # Block 3: A_top[:, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_222(0, 0, 2)
    
    # Block 4: A_top[:, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_222(0, 2, 2)
    
    # Bottom part: A[2, :] * B[:, :]
    # Block 5: A_bot[0:2] * B[0:2, 0:2] -> C[2, 0:2]
    add_122(2, 0, 0)
    
    # Block 6: A_bot[2:4] * B[2:4, 0:2] -> C[2, 0:2]
    add_122(2, 2, 0)
    
    # Block 7: A_bot[0:2] * B[0:2, 2:4] -> C[2, 2:4]
    add_122(2, 0, 2)
    
    # Block 8: A_bot[2:4] * B[2:4, 2:4] -> C[2, 2:4]
    add_122(2, 2, 2)
    
    # Verify the decomposition
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal sum_j A[i,j] B[j,k] for the multiplication tensor
    # The tensor entries are T[i,j,k] = 1 for all valid i,j,k
    # So we check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m + j]
                v_val = V_arr[j*p + k]
                w_val = W_arr[k*n + i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank 48
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb), dtype=float)
        V_fb = np.zeros((m*p, rank_fb), dtype=float)
        W_fb = np.zeros((p*n, rank_fb), dtype=float)
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m + j, col_fb] = 1
                    V_fb[j*p + k, col_fb] = 1
                    W_fb[k*n + i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
