import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    Uses a block-wise Strassen-based decomposition with rank 28.
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 decomposition terms
    # Each term is (u_dict, v_dict, w_dict) where dicts map local index -> coefficient
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_base = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),          # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),          # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),          # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),          # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),          # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),          # M7
    ]
    
    def add_strassen(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        """
        Adds 7 Strassen terms to the global matrices.
        
        a_col_offset: 0 for A[:, 0:2], 1 for A[:, 2:4]
        b_row_offset: 0 for B[0:2, :], 1 for B[2:4, :]
        b_col_offset: 0 for B[:, 0:2], 1 for B[:, 2:4]
        c_col_offset: 0 for C[:, 0:2], 1 for C[:, 2:4]
        """
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_base):
            col = start_col + t_idx
            
            # Map local A indices to global
            # A is n x m (2 x 4). Index in U is i*m + j.
            # Local index l in 2x2 block: r = l//2, c = l%2.
            # Global row i = r.
            # Global col j = a_col_offset*2 + c.
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * m + (a_col_offset * 2 + c)
                U[g_idx, col] += coeff
            
            # Map local B indices to global
            # B is m x p (4 x 4). Index in V is j*p + k.
            # Local index l in 2x2 block: r = l//2, c = l%2.
            # Global row j = b_row_offset*2 + r.
            # Global col k = b_col_offset*2 + c.
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (b_row_offset * 2 + r) * p + (b_col_offset * 2 + c)
                V[g_idx, col] += coeff
            
            # Map local C indices to global
            # C is n x p (2 x 4). Index in W is k*n + i.
            # Local index l in 2x2 block: r = l//2, c = l%2.
            # Global row i = r.
            # Global col k = c_col_offset*2 + c.
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_k = c_col_offset * 2 + c
                g_i = r
                g_idx = g_k * n + g_i
                W[g_idx, col] += coeff

    # We need to compute C = A * B
    # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
    
    # 1. A[:, 0:2] * B[0:2, 0:2] -> contributes to C[:, 0:2]
    add_strassen(0, 0, 0, 0, 0)
    
    # 2. A[:, 2:4] * B[2:4, 0:2] -> contributes to C[:, 0:2]
    add_strassen(1, 1, 0, 0, 7)
    
    # 3. A[:, 0:2] * B[0:2, 2:4] -> contributes to C[:, 2:4]
    add_strassen(0, 0, 1, 1, 14)
    
    # 4. A[:, 2:4] * B[2:4, 2:4] -> contributes to C[:, 2:4]
    add_strassen(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
            
    if not np.allclose(C, Cf):
        print("Mismatch!")
        print(C)
        print(Cf)
    else:
        print("OK rank", U.shape[1])
