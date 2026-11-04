"""Matrix multiplication 4x4 * 4x5 via Strassen + schoolbook for extra column.
Rank 65 = 49 (Strassen for 4x4*4x4) + 16 (schoolbook for 4x4*4x1).
"""
import time
import numpy as np
from fractions import Fraction

def _strassen_2x2():
    """Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a00+a11)(b00+b11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1  # C00
    W[3,0]=1  # C11
    
    # M2 = (a10+a11)b00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1  # C10
    W[3,1]=-1 # C11
    
    # M3 = a00(b10-b11)
    U[0,2]=1
    V[2,2]=1; V[3,2]=-1
    W[1,2]=1  # C01
    W[3,2]=1  # C11
    
    # M4 = a11(b01-b00)
    U[3,3]=1
    V[1,3]=1; V[0,3]=-1
    W[0,3]=1  # C00
    W[1,3]=1  # C10
    
    # M5 = (a00+a01)b11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1 # C00
    W[2,4]=1  # C01
    
    # M6 = (a10-a00)(b00+b01)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[0,5]=1  # C00
    W[1,5]=1  # C10
    
    # M7 = (a01-a11)(b10+b11)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=-1 # C00
    W[3,6]=1  # C11
    
    return U, V, W

def _build_strassen_4x4_4x4():
    """Build Strassen-based decomposition for 4x4 * 4x4 -> 4x4 with rank 49."""
    n = m = p = 4
    rank = 49
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    sU, sV, sW = _strassen_2x2()
    
    block_products = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(0,0,1),(1,0,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,-1),(1,1,1)]),
    ]
    
    for pi, (a_blocks, b_blocks, c_blocks) in enumerate(block_products):
        for s in range(7):
            col = pi * 7 + s
            
            for (br, bc, sign) in a_blocks:
                for xi in range(4):
                    u_val = sU[xi, s]
                    if u_val == 0: continue
                    i2, j2 = xi // 2, xi % 2
                    row = (br*2 + i2) * 4 + (bc*2 + j2)
                    U[row, col] += u_val * sign
            
            for (br, bc, sign) in b_blocks:
                for yi in range(4):
                    v_val = sV[yi, s]
                    if v_val == 0: continue
                    i2, j2 = yi // 2, yi % 2
                    row = (br*2 + i2) * 4 + (bc*2 + j2)
                    V[row, col] += v_val * sign
            
            for (br, bc, sign) in c_blocks:
                for zi in range(4):
                    w_val = sW[zi, s]
                    if w_val == 0: continue
                    i2, j2 = zi // 2, zi % 2
                    row = (bc*2 + j2) * 4 + (br*2 + i2)
                    W[row, col] += w_val * sign
    
    return U, V, W

def _build_extra_column():
    """Schoolbook for 4x4 * 4x1 -> 4x1, rank 16."""
    n, m, p = 4, 4, 1
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    """Verify: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    return np.allclose(T, 1.0, atol=1e-9)

def _try_rank_reduction(U, V, W, n, m, p, rank, deadline):
    """Try to reduce rank by checking if any rank term is redundant.
    
    For a valid decomposition, if we can find coefficients alpha_c such that
    sum_c alpha_c * (U_c ⊗ V_c ⊗ W_c) = 0 (the zero tensor), then we can
    eliminate one term.
    
    Practically, we check if removing any single column still gives a valid
    decomposition (it won't in general, but we check for exact cancellation).
    
    A more effective approach: try to express one rank term as a linear
    combination of others by solving a linear system.
    """
    if time.time() > deadline:
        return U, V, W, rank
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We want to check if the decomposition is valid and if rank can be reduced.
    
    # For each column c, compute the tensor T_c[i,j,k] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # If T_c = sum_{c' != c} alpha_{c'} T_{c'}, then we can eliminate column c.
    
    # This is a high-dimensional problem. Instead, let's try a simpler heuristic:
    # Check if any two columns are proportional (unlikely but easy to check).
    
    # Alternative: use the fact that the tensor rank is the minimum number of
    # rank-1 tensors needed. We can try to find a lower rank by solving an
    # optimization problem, but that's expensive.
    
    # For now, just verify the decomposition is valid and return as-is.
    if _verify(U, V, W, n, m, p, rank):
        return U, V, W, rank
    
    return None

def construct():
    n, m, p = 4, 4, 5
    deadline = time.time() + 130
    
    # Build Strassen 4x4x4 (rank 49) + schoolbook 4x4x1 (rank 16) = rank 65
    U_s, V_s, W_s = _build_strassen_4x4_4x4()
    U_e, V_e, W_e = _build_extra_column()
    
    total_rank = 49 + 16
    U = np.zeros((n*m, total_rank), dtype=np.int64)
    V = np.zeros((m*p, total_rank), dtype=np.int64)
    W = np.zeros((p*n, total_rank), dtype=np.int64)
    
    # Strassen part: 4x4 * 4x4 -> 4x4 (first 4 columns of B)
    U[:, :49] = U_s
    for j in range(4):
        for k in range(4):
            V[j*p+k, :49] = V_s[j*4+k, :]
    for k in range(4):
        for i in range(n):
            W[k*n+i, :49] = W_s[k*4+i, :]
    
    # Schoolbook part: 4x4 * 4x1 -> 4x1 (5th column of B)
    U[:, 49:] = U_e
    for j in range(m):
        V[j*p+4, 49:] = V_e[j*1+0, :]
    for i in range(n):
        W[4*n+i, 49:] = W_e[0*n+i, :]
    
    # Verify
    if _verify(U, V, W, n, m, p, total_rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook (rank 80)
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
