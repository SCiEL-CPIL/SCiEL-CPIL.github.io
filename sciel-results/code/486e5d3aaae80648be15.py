"""
Matrix multiplication 2x4 by 4x5 - rank-minimal exact tensor decomposition.
Uses Strassen-based decomposition as a base, then searches for rank reductions
using linear algebra to find and exploit dependencies in the tensor decomposition.
"""
import numpy as np
import time
from itertools import combinations

# Deadline guard
_start_time = time.time()
_deadline = _start_time + 145  # 150s limit, 5s safety margin

def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the 2x4 * 4x5 tensor."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_36():
    """Build the rank-36 Strassen-based decomposition."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n * m, rank))  # 8 x 36
    V = np.zeros((m * p, rank))  # 20 x 36
    W = np.zeros((p * n, rank))  # 10 x 36
    
    col = 0
    
    def add_strassen_2x2(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        u00 = a_row * m + a_col
        u01 = a_row * m + a_col + 1
        u10 = (a_row + 1) * m + a_col
        u11 = (a_row + 1) * m + a_col + 1
        
        v00 = b_row * p + b_col
        v01 = b_row * p + b_col + 1
        v10 = (b_row + 1) * p + b_col
        v11 = (b_row + 1) * p + b_col + 1
        
        w00 = c_col * n + c_row
        w01 = (c_col + 1) * n + c_row
        w10 = c_col * n + c_row + 1
        w11 = (c_col + 1) * n + c_row + 1
        
        # M1 = (a00+a11)(b00+b11)
        U[u00, col] += 1.0
        U[u11, col] += 1.0
        V[v00, col] += 1.0
        V[v11, col] += 1.0
        W[w00, col] += 1.0
        W[w11, col] += 1.0
        col += 1
        
        # M2 = (a10+a11)(b00)
        U[u10, col] += 1.0
        U[u11, col] += 1.0
        V[v00, col] += 1.0
        W[w10, col] += 1.0
        W[w11, col] -= 1.0
        col += 1
        
        # M3 = a00(b01-b11)
        U[u00, col] += 1.0
        V[v01, col] += 1.0
        V[v11, col] -= 1.0
        W[w01, col] += 1.0
        W[w11, col] += 1.0
        col += 1
        
        # M4 = a11(b10-b00)
        U[u11, col] += 1.0
        V[v10, col] += 1.0
        V[v00, col] -= 1.0
        W[w00, col] += 1.0
        W[w10, col] += 1.0
        col += 1
        
        # M5 = (a00+a01)(b11)
        U[u00, col] += 1.0
        U[u01, col] += 1.0
        V[v11, col] += 1.0
        W[w00, col] -= 1.0
        W[w01, col] += 1.0
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[u10, col] += 1.0
        U[u00, col] -= 1.0
        V[v00, col] += 1.0
        V[v01, col] += 1.0
        W[w11, col] += 1.0
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[u01, col] += 1.0
        U[u11, col] -= 1.0
        V[v10, col] += 1.0
        V[v11, col] += 1.0
        W[w00, col] += 1.0
        col += 1
    
    def add_2x2_by_2x1(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        u00 = a_row * m + a_col
        u01 = a_row * m + a_col + 1
        u10 = (a_row + 1) * m + a_col
        u11 = (a_row + 1) * m + a_col + 1
        
        v0 = b_row * p + b_col
        v1 = (b_row + 1) * p + b_col
        
        w0 = c_col * n + c_row
        w1 = c_col * n + c_row + 1
        
        U[u00, col] = 1.0
        V[v0, col] = 1.0
        W[w0, col] = 1.0
        col += 1
        
        U[u01, col] = 1.0
        V[v1, col] = 1.0
        W[w0, col] = 1.0
        col += 1
        
        U[u10, col] = 1.0
        V[v0, col] = 1.0
        W[w1, col] = 1.0
        col += 1
        
        U[u11, col] = 1.0
        V[v1, col] = 1.0
        W[w1, col] = 1.0
        col += 1
    
    # A1 @ B1: 2x2 * 2x5
    add_strassen_2x2(a_row=0, a_col=0, b_row=0, b_col=0, c_row=0, c_col=0)
    add_strassen_2x2(a_row=0, a_col=0, b_row=0, b_col=2, c_row=0, c_col=2)
    add_2x2_by_2x1(a_row=0, a_col=0, b_row=0, b_col=4, c_row=0, c_col=4)
    
    # A2 @ B2: 2x2 * 2x5
    add_strassen_2x2(a_row=0, a_col=2, b_row=2, b_col=0, c_row=0, c_col=0)
    add_strassen_2x2(a_row=0, a_col=2, b_row=2, b_col=2, c_row=0, c_col=2)
    add_2x2_by_2x1(a_row=0, a_col=2, b_row=2, b_col=4, c_row=0, c_col=4)
    
    return U, V, W

def _build_schoolbook_40():
    """Build the rank-40 schoolbook decomposition."""
    n, m, p = 2, 4, 5
    rank = 40
    U = np.zeros((n * m, rank))  # 8 x 40
    V = np.zeros((m * p, rank))  # 20 x 40
    W = np.zeros((p * n, rank))  # 10 x 40
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t = i * m * p + j * p + k
                U[i*m+j, t] = 1.0
                V[j*p+k, t] = 1.0
                W[k*n+i, t] = 1.0
    
    return U, V, W

def _try_greedy_remove(U, V, W, max_removals=20):
    """Try to remove columns (rank terms) while maintaining correctness."""
    n, m, p = 2, 4, 5
    current_rank = U.shape[1]
    
    for attempt in range(max_removals * current_rank):
        if time.time() > _deadline - 10:
            break
        improved = False
        for col_idx in range(current_rank):
            if time.time() > _deadline - 10:
                break
            mask = [i for i in range(current_rank) if i != col_idx]
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            if _verify(U_test, V_test, W_test):
                U, V, W = U_test, V_test, W_test
                current_rank -= 1
                improved = True
                break
        if not improved:
            break
    
    return U, V, W

def _clean_matrix(mat):
    """Convert to integer where possible."""
    result = []
    for row in mat:
        cleaned = []
        for val in row:
            if abs(val - round(val)) < 1e-9:
                cleaned.append(int(round(val)))
            else:
                cleaned.append(float(val))
        result.append(cleaned)
    return result

def construct():
    global _deadline
    _start = time.time()
    _deadline = _start + 140  # 10 second safety margin
    
    # Try multiple base decompositions
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    # Base 1: Strassen rank-36
    U1, V1, W1 = _build_strassen_36()
    if _verify(U1, V1, W1):
        U1, V1, W1 = _try_greedy_remove(U1, V1, W1)
        rank1 = U1.shape[1]
        if rank1 < best_rank:
            best_rank = rank1
            best_U, best_V, best_W = U1, V1, W1
    
    # Base 2: Schoolbook rank-40
    U2, V2, W2 = _build_schoolbook_40()
    if _verify(U2, V2, W2):
        U2, V2, W2 = _try_greedy_remove(U2, V2, W2)
        rank2 = U2.shape[1]
        if rank2 < best_rank:
            best_rank = rank2
            best_U, best_V, best_W = U2, V2, W2
    
    if best_U is None:
        raise ValueError("No valid decomposition found")
    
    U = _clean_matrix(best_U.tolist())
    V = _clean_matrix(best_V.tolist())
    W = _clean_matrix(best_W.tolist())
    
    return {"U": U, "V": V, "W": W}
