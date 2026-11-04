"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]
_DEADLINE = time.time() + 140  # 10s safety margin

def _schoolbook(n, m, p):
    """Return U, V, W for schoolbook decomposition of (n,m,p) matrix mult."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return U, V, W for Strassen decomposition of 2x2x2 matrix mult. Rank 7."""
    # Strassen: M1..M7
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+b)(e+f)
    # M2 = (c+d)e
    # M3 = a(g+h)
    # M4 = d(f-h)
    # M5 = (a+d)(e+h)
    # M6 = (b-c)(g+h)
    # M7 = (a-c)(e+f)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U: rows indexed by (i,j) for A, i.e., 0:a, 1:b, 2:c, 3:d
    # V: rows indexed by (j,k) for B, i.e., 0:e, 1:f, 2:g, 3:h
    # W: rows indexed by (k,i) for C, i.e., 0:C00, 1:C01, 2:C10, 3:C11
    #
    # Each M_s is a product of a linear form in A and a linear form in B.
    # U[:, s] gives the coefficients for A in M_s
    # V[:, s] gives the coefficients for B in M_s
    # W[:, s] gives how M_s contributes to C
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+b)(e+f)
    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = -1; W[3, 0] = 1  # C00+=M1, C01+=M1, C10-=M1, C11+=M1
    # Wait, let me recheck. C00 = M1 + M4 - M5 + M7, C01 = M3 + M5, C10 = M2 + M4, C11 = M1 - M2 + M3 + M6
    
    # Let me redo this carefully.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 = (a+b)(e+f): U: a=1,b=1; V: e=1,f=1
    # M2 = (c+d)e: U: c=1,d=1; V: e=1
    # M3 = a(g+h): U: a=1; V: g=1,h=1
    # M4 = d(f-h): U: d=1; V: f=1,h=-1
    # M5 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1
    # M6 = (b-c)(g+h): U: b=1,c=-1; V: g=1,h=1
    # M7 = (a-c)(e+f): U: a=1,c=-1; V: e=1,f=1
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10
    W[3, 1] = -1  # C11
    
    # M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = 1
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4
    U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5
    U[0, 4] = 1; U[3, 4] = 1
    V[0, 4] = 1; V[3, 4] = 1
    W[0, 4] = -1  # C00
    W[1, 4] = 1  # C01
    
    # M6
    U[1, 5] = 1; U[2, 5] = -1
    V[2, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1  # C11
    
    # M7
    U[0, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[1, 6] = 1
    W[0, 6] = 1  # C00
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor T from U, V, W: T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    # Compare with the standard tensor: T_std[i,j,k] = 1 if... actually the standard tensor is:
    # T_std[i,j,k] = delta_{j,j} * delta_{k,k} ... no.
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k? No.
    # Actually, the tensor for matrix mult is: T_{i,j,k} where (C)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in the sense that each triple contributes 1)
    # Wait no. The tensor is defined such that C_{i,k} = sum_{j,s} A_{i,j} U_{i*m+j,s} * B_{j,k} V_{j*p+k,s} * W_{k*n+i,s}
    # For this to equal sum_j A_{i,j} B_{j,k}, we need:
    # sum_s U_{i*m+j,s} V_{j*p+k,s} W_{k*n+i,s} = 1 for all i,j,k
    # Actually no, the correct condition is:
    # For each (i,j,k): sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
    # Wait, that's not right either. Let me think again.
    
    # The decomposition says: T = sum_s U_s ⊗ V_s ⊗ W_s where
    # (U_s ⊗ V_s ⊗ W_s)[i,j,k] = U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    # And T[i,j,k] should be 1 for all i,j,k (since the matrix mult tensor has all entries 1 in the appropriate sense)
    
    # Actually, the matrix multiplication tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Because (AB)_{i,k} = sum_j A_{i,j} B_{j,k}, and each term A_{i,j} * B_{j,k} corresponds to T[i,j,k] = 1.
    
    # So we need: for all i,j,k: sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1
    
    # This is equivalent to: the 3D array S[i,j,k] = sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] equals 1 everywhere.
    
    # Let's verify this.
    S = np.zeros((n, m, p))
    for s in range(rank):
        # U[:, s] is a vector of length n*m, reshape to (n, m)
        U_s = U[:, s].reshape(n, m)
        V_s = V[:, s].reshape(m, p)
        W_s = W[:, s].reshape(p, n)
        # The outer product contribution to S[i,j,k] is U_s[i,j] * V_s[j,k] * W_s[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    S[i, j, k] += U_s[i, j] * V_s[j, k] * W_s[k, i]
    
    return np.allclose(S, 1.0, atol=tol)

def _combine_decompositions(decomps, n, m, p):
    """Combine multiple decompositions that cover disjoint sub-problems.
    Each decomp is (U, V, W, i_range, j_range, k_range) where the sub-problem is
    A[i_range, j_range] @ B[j_range, k_range] -> C[i_range, k_range]
    """
    total_rank = sum(d[0].shape[1] for d in decomps)
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col = 0
    for U, V, W, i_range, j_range, k_range in decomps:
        r = U.shape[1]
        # U is (len(i_range)*len(j_range), r), needs to be placed into U_full at rows (i*m+j) for i in i_range, j in j_range
        # V is (len(j_range)*len(k_range), r), placed into V_full at rows (j*p+k) for j in j_range, k in k_range
        # W is (len(k_range)*len(i_range), r), placed into W_full at rows (k*n+i) for k in k_range, i in i_range
        
        for idx_u, (ii, jj) in enumerate(product(i_range, j_range)):
            U_full[ii * m + jj, col:col+r] = U[idx_u, :]
        for idx_v, (jj, kk) in enumerate(product(j_range, k_range)):
            V_full[jj * p + kk, col:col+r] = V[idx_v, :]
        for idx_w, (kk, ii) in enumerate(product(k_range, i_range)):
            W_full[kk * n + ii, col:col+r] = W[idx_w, :]
        
        col += r
    
    return U_full, V_full, W_full

def _block_decomp(n, m, p, i_range, j_range, k_range):
    """Get schoolbook decomposition for a sub-block."""
    ni = len(i_range)
    nj = len(j_range)
    nk = len(k_range)
    U, V, W = _schoolbook(ni, nj, nk)
    return U, V, W, i_range, j_range, k_range

def _strassen_decomp(i_range, j_range, k_range):
    """Get Strassen decomposition for a 2x2x2 sub-block."""
    U, V, W = _strassen_2x2x2()
    return U, V, W, i_range, j_range, k_range

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strategy: Split 5 into 2+3
    # A (2x5) = [A1 (2x2) | A2 (2x3)]
    # B (5x5) = [[B11 (2x2), B12 (2x3)], [B21 (3x2), B22 (3x3)]]
    # C (2x5) = [C1 (2x2) | C2 (2x3)]
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    
    # Sub-problems:
    # 1. (2,2,2): A1*B11 -> C1_left  [i:0-1, j:0-1, k:0-1]
    # 2. (2,3,2): A2*B21 -> C1_right [i:0-1, j:2-4, k:0-1]
    # 3. (2,2,3): A1*B12 -> C2_left  [i:0-1, j:0-1, k:2-4]
    # 4. (2,3,3): A2*B22 -> C2_right [i:0-1, j:2-4, k:2-4]
    
    # Use Strassen for (2,2,2), schoolbook for others
    # Rank: 7 + 12 + 12 + 18 = 49
    
    decomps = []
    
    # 1. (2,2,2) with Strassen
    i1, j1, k1 = [0,1], [0,1], [0,1]
    decomps.append(_strassen_decomp(i1, j1, k1))
    
    # 2. (2,3,2) schoolbook
    i2, j2, k2 = [0,1], [2,3,4], [0,1]
    decomps.append(_block_decomp(n, m, p, i2, j2, k2))
    
    # 3. (2,2,3) schoolbook
    i3, j3, k3 = [0,1], [0,1], [2,3,4]
    decomps.append(_block_decomp(n, m, p, i3, j3, k3))
    
    # 4. (2,3,3) schoolbook
    i4, j4, k4 = [0,1], [2,3,4], [2,3,4]
    decomps.append(_block_decomp(n, m, p, i4, j4, k4))
    
    U, V, W = _combine_decompositions(decomps, n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    # Try to improve by searching for better decompositions of sub-blocks
    # For now, return what we have
    elapsed = time.time() - start_time
    if elapsed > 120:
        # Emergency: return schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    # Convert to lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
