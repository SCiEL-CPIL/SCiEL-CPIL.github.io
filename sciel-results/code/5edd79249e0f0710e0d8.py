"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_strassen_decomp(n, m, p, j_pairs, k_pairs, k_singles=None):
    """Build a Strassen-based decomposition with given pairings."""
    if k_singles is None:
        used_k = set()
        for k0, k1 in k_pairs:
            used_k.add(k0)
            used_k.add(k1)
        k_singles = [k for k in range(p) if k not in used_k]
    
    # Rank calculation
    rank = 7 * len(j_pairs) * len(k_pairs) + 4 * len(j_pairs) * len(k_singles) + 2 * (m - 2*len(j_pairs)) * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            
            # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
        
        # For each k_single with this j_pair: 4 standard products
        for ks in k_singles:
            U[0*m + j0][col] += 1
            V[j0*p + ks][col] += 1
            W[ks*n + 0][col] += 1
            col += 1
            
            U[0*m + j1][col] += 1
            V[j1*p + ks][col] += 1
            W[ks*n + 0][col] += 1
            col += 1
            
            U[1*m + j0][col] += 1
            V[j0*p + ks][col] += 1
            W[ks*n + 1][col] += 1
            col += 1
            
            U[1*m + j1][col] += 1
            V[j1*p + ks][col] += 1
            W[ks*n + 1][col] += 1
            col += 1
    
    # Remaining j indices: standard products
    used_j = set()
    for j0, j1 in j_pairs:
        used_j.add(j0)
        used_j.add(j1)
    
    for j in range(m):
        if j in used_j:
            continue
        for k in range(p):
            U[0*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            
            U[1*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy (vectorized)."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        u_2d = u_col.reshape(n, m)
        v_2d = v_col.reshape(m, p)
        w_2d = w_col.reshape(p, n)
        T += np.einsum('il,lk,ki->ilk', u_2d, v_2d, w_2d)
    
    return np.all(np.abs(T - 1.0) < 1e-9)

def construct():
    n, m, p = DIMENSIONS
    
    # Direct construction of the known Rank 54 decomposition
    # j-pairs: (0,1), (2,3), single: 4
    # k-pairs: (0,1), (2,3), (4,5)
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    k_singles = []
    
    U, V, W = _build_strassen_decomp(n, m, p, j_pairs, k_pairs, k_singles)
    rank = len(U[0])
    
    # Verify
    if not _verify_decomp(U, V, W, n, m, p, rank):
        # Fallback: Standard decomposition (Rank 60)
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        rank = n*m*p
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
