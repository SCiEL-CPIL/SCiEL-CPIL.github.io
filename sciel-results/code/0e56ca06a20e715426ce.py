"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Schoolbook baseline (rank 48)
    U0, V0, W0 = _schoolbook(n, m, p)

    best = None
    best_rank = n * m * p

    # Try the structured 4-block rank-44 scheme
    if time.time() < deadline - 20:
        try:
            U1, V1, W1, r1 = _four_block_decomposition(n, m, p)
            if _verify(n, m, p, U1, V1, W1):
                best = (U1, V1, W1)
                best_rank = r1
        except Exception:
            pass

    # Try to refine the 44-rank solution
    if best is not None and time.time() < deadline - 10:
        try:
            U2, V2, W2, r2 = _refine_decomposition(n, m, p, best[0], best[1], best[2], best_rank, deadline)
            if r2 < best_rank and _verify(n, m, p, U2, V2, W2):
                best = (U2, V2, W2)
                best_rank = r2
        except Exception:
            pass

    # Try a 3-block decomposition (3+3+2)
    if time.time() < deadline - 30:
        try:
            U3, V3, W3, r3 = _three_block_decomposition(n, m, p)
            if U3 is not None and _verify(n, m, p, U3, V3, W3):
                if r3 < best_rank:
                    best = (U3, V3, W3)
                    best_rank = r3
        except Exception:
            pass

    if best is not None:
        U, V, W = best
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    return {"U": _to_list(U0), "V": _to_list(V0), "W": _to_list(W0)}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _four_block_decomposition(n, m, p):
    """
    Split the p=8 columns into 4 blocks of size 2.
    For each 2x3 x 3x2 sub-problem, use a rank-11 decomposition.
    Total rank: 44.
    """
    n, m, p = 2, 3, 8
    block_size = p // 4  # 2
    
    rank_per_block = 11
    total_rank = rank_per_block * 4  # 44
    
    U = np.zeros((n * m, total_rank))  # 6 x 44
    V = np.zeros((m * p, total_rank))  # 24 x 44
    W = np.zeros((p * n, total_rank))  # 16 x 44
    
    for b in range(4):
        k_offset = b * block_size  # 0, 2, 4, 6
        col_offset = b * rank_per_block
        
        # Strassen 7 terms for 2x2 * 2x2 part
        strassen_U = [
            {0: 1, 4: 1},      # M1: (a00+a01)
            {3: 1, 4: 1},      # M2: (a10+a11)  -- wait, need to be careful
            {0: 1},            # M3
            {4: 1},            # M4
            {0: 1, 1: 1},      # M5
            {3: 1, 0: -1},     # M6
            {1: 1, 4: -1},     # M7
        ]
        
        # Let me re-derive Strassen for 2x2 * 2x2
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C[0,0] = a00*b00 + a01*b10
        # C[0,1] = a00*b01 + a01*b11
        # C[1,0] = a10*b00 + a11*b10
        # C[1,1] = a10*b01 + a11*b11
        #
        # Strassen:
        # M1 = (a00+a01)*(b00+b10)
        # M2 = a00*(b10-b00)  -- actually let me use the standard form
        #
        # Standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # In our indexing: A[i*2+j] for i,j in {0,1}
        # A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
        
        strassen_U = [
            {0: 1, 3: 1},      # M1: A00+A11
            {2: 1, 3: 1},      # M2: A10+A11
            {0: 1},            # M3: A00
            {3: 1},            # M4: A11
            {0: 1, 1: 1},      # M5: A00+A01
            {2: 1, 0: -1},     # M6: A10-A00
            {1: 1, 3: -1},     # M7: A01-A11
        ]
        
        # B is 2x2, indexed B[j*2+k] for j,k in {0,1}
        # B[0]=B00, B[1]=B01, B[2]=B10, B[3]=B11
        # For block b, B is at rows 0..1 (of the 2x2 part), columns k_offset..k_offset+1
        # In the full V matrix (24 x rank), row j*8+k where j=0,1 (top 2 rows), k=k_offset,k_offset+1
        
        strassen_V = [
            {0 * 8 + k_offset + 0: 1, 1 * 8 + k_offset + 1: 1},  # M1: B00+B11
            {0 * 8 + k_offset + 0: 1},                           # M2: B00
            {1 * 8 + k_offset + 0: 1, 1 * 8 + k_offset + 1: -1}, # M3: B10-B11
            {0 * 8 + k_offset + 1: 1, 0 * 8 + k_offset + 0: -1}, # M4: B01-B00
            {1 * 8 + k_offset + 1: 1},                           # M5: B11
            {0 * 8 + k_offset + 0: 1, 0 * 8 + k_offset + 1: 1},  # M6: B00+B01
            {1 * 8 + k_offset + 0: 1, 1 * 8 + k_offset + 1: 1},  # M7: B10+B11
        ]
        
        # W: C[i,k] -> W[k*2+i]
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For block b, C is at columns k_offset, k_offset+1
        # C[0, k_offset] = C00, C[0, k_offset+1] = C01
        # C[1, k_offset] = C10, C[1, k_offset+1] = C11
        # W index: k*2+i where k is global column index
        
        strassen_W = [
            {(k_offset + 0) * 2 + 0: 1, (k_offset + 1) * 2 + 1: 1},  # M1: C00, C11
            {(k_offset + 0) * 2 + 1: 1},                             # M2: C10
            {(k_offset + 1) * 2 + 0: 1, (k_offset + 1) * 2 + 1: 1},  # M3: C01, C11
            {(k_offset + 0) * 2 + 0: 1, (k_offset + 0) * 2 + 1: 1},  # M4: C00, C10
            {(k_offset + 0) * 2 + 0: -1, (k_offset + 1) * 2 + 0: 1}, # M5: -C00, +C01
            {(k_offset + 1) * 2 + 1: 1},                             # M6: C11
            {(k_offset + 0) * 2 + 0: 1},                             # M7: C00
        ]
        
        for c in range(7):
            for row, val in strassen_U[c].items():
                U[row, col_offset + c] += val
            for row, val in strassen_V[c].items():
                V[row, col_offset + c] += val
            for row, val in strassen_W[c].items():
                W[row, col_offset + c] += val
        
        # Outer product part: A_right(2x1) * B_bottom(1x2) - rank 4
        # A_right is column 2 of A (rows 0,1), B_bottom is row 2 of B (cols k_offset, k_offset+1)
        # C[i, k] += A[i, 2] * B[2, k]
        for ii in range(2):
            for kk in range(2):
                c = 7 + ii * 2 + kk  # 7,8,9,10
                U[ii * 3 + 2, col_offset + c] = 1
                V[2 * 8 + k_offset + kk, col_offset + c] = 1
                W[(k_offset + kk) * 2 + ii, col_offset + c] = 1
    
    return U, V, W, total_rank


def _verify(n, m, p, U, V, W):
    """
    Verify that the decomposition is exact.
    """
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    rank = U.shape[1]
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    u_vals = np.einsum('ijc,ij->ic', U_reshaped, A)
    v_vals = np.einsum('lkp,lk->kp', V_reshaped, B)
    C_comp = np.einsum('kic,ic,kc->ik', W_reshaped, u_vals, v_vals)
    
    if not np.allclose(C_comp, C_true, atol=1e-9):
        return False
    return True


def _refine_decomposition(n, m, p, U, V, W, current_rank, deadline):
    """
    Try to reduce the rank by removing linearly dependent columns.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W, rank
    
    removed_any = True
    while removed_any and time.time() < deadline - 5:
        removed_any = False
        for c_remove in range(rank - 1, -1, -1):
            if time.time() >= deadline - 5:
                break
            
            mask = [c for c in range(rank) if c != c_remove]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            
            if _verify(n, m, p, U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                rank -= 1
                removed_any = True
                if rank <= 1:
                    return U, V, W, rank
    
    return U, V, W, rank


def _three_block_decomposition(n, m, p):
    """
    Try splitting into blocks of size 3, 3, 2.
    For 2x3 x 3x3: use a decomposition
    For 2x3 x 3x2: use a decomposition
    """
    n, m, p = 2, 3, 8
    
    # For now, fall back to 4-block if we can't find something better
    # This is a placeholder for future improvements
    return None, None, None, 999


def _to_list(arr):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in arr:
        r = []
        for val in row:
            v = round(val * 2) / 2
            if abs(v - round(v)) < 1e-9:
                r.append(int(round(v)))
            else:
                r.append(v)
        result.append(r)
    return result

# EVOLVE-BLOCK-END
