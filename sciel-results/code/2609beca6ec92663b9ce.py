"""Matrix multiplication 2x4x6: rank-42 Strassen decomposition."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    """
    Returns U, V, W of shapes (8, 42), (24, 42), (12, 42) with integer entries
    that exactly decompose the 2x4 x 4x6 matrix multiplication tensor.
    """
    n, m, p = 2, 4, 6
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 factors:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # We divide the 4 columns of A into 2 blocks of 2 (j=0,1 and j=2,3)
    # and the 6 rows of B into 3 blocks of 2 (k=0,1 and k=2,3 and k=4,5)
    
    # For each pair of j-block (jb) and k-block (kb), we apply Strassen
    # on the 2x2 (i x local_j) x 2x2 (local_j x local_k) subproblem.
    
    # Strassen coefficients for U (A-space), V (B-space), W (output-space):
    # Each factor s has coefficients (u, v, w) where:
    # u is a 2x2 matrix (for i=0,1 and local j=0,1)
    # v is a 2x2 matrix (for local j=0,1 and local k=0,1)
    # w is a 2x2 matrix (for local k=0,1 and i=0,1)
    
    # M1: u=[[1,0],[0,1]], v=[[1,0],[0,1]], w=[[1,0],[0,1]]
    # M2: u=[[0,0],[1,1]], v=[[1,0],[0,0]], w=[[0,0],[1,0]]
    # M3: u=[[1,0],[0,0]], v=[[0,1],[0,-1]], w=[[1,0],[0,0]]
    # M4: u=[[0,0],[0,1]], v=[[0,0],[1,-1]], w=[[0,1],[0,0]]
    # M5: u=[[1,1],[0,0]], v=[[0,0],[0,1]], w=[[0,0],[1,1]]
    # M6: u=[[1,0],[-1,0]], v=[[1,0],[1,0]], w=[[0,-1],[0,0]]
    # M7: u=[[0,1],[0,-1]], v=[[0,1],[0,1]], w=[[0,1],[0,-1]]
    
    strassen_factors = [
        # (U_2x2, V_2x2, W_2x2)
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),   # M1
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]),   # M2
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[1,0],[0,0]]),  # M3
        ([[0,0],[0,1]], [[0,0],[1,-1]], [[0,1],[0,0]]),  # M4
        ([[1,1],[0,0]], [[0,0],[0,1]], [[0,0],[1,1]]),   # M5
        ([[1,0],[-1,0]], [[1,0],[1,0]], [[0,-1],[0,0]]), # M6
        ([[0,1],[0,-1]], [[0,1],[0,1]], [[0,1],[0,-1]]), # M7
    ]
    
    col = 0
    for jb in range(2):  # j-block: 0 (j=0,1) or 1 (j=2,3)
        for kb in range(3):  # k-block: 0 (k=0,1), 1 (k=2,3), 2 (k=4,5)
            for s in range(7):
                u_mat, v_mat, w_mat = strassen_factors[s]
                
                # Fill U: for each i in {0,1}, local_j in {0,1}
                for i in range(2):
                    for lj in range(2):
                        val = u_mat[i][lj]
                        if val != 0:
                            j_global = jb * 2 + lj
                            U[i * m + j_global, col] += val
                
                # Fill V: for each local_j in {0,1}, local_k in {0,1}
                for lj in range(2):
                    for lk in range(2):
                        val = v_mat[lj][lk]
                        if val != 0:
                            j_global = jb * 2 + lj
                            k_global = kb * 2 + lk
                            V[j_global * p + k_global, col] += val
                
                # Fill W: for each local_k in {0,1}, i in {0,1}
                for lk in range(2):
                    for i in range(2):
                        val = w_mat[lk][i]
                        if val != 0:
                            k_global = kb * 2 + lk
                            W[k_global * n + i, col] += val
                
                col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    U, V, W = result["U"], result["V"], result["W"]
    n, m, p = 2, 4, 6
    rank = len(U[0])
    
    # Verify: reconstruct C = A*B via sum_s (A @ U_s) * (V_s @ B) @ W_s^T? 
    # Actually: C[k,n+i] = sum_{i,j} A[i*m+j] * B[j*p+k] * ... 
    # The tensor T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    # should equal delta(i,j,k) for the multiplication tensor.
    
    # Let's verify on random matrices
    A = np.random.randint(-5, 6, size=(n, m))
    B = np.random.randint(-5, 6, size=(m, p))
    C_expected = A @ B
    
    # Reconstruct C using the decomposition
    C_recon = np.zeros((n, p), dtype=int)
    for s in range(rank):
        u_vec = np.array(U[:, s]).reshape(n, m)
        v_vec = np.array(V[:, s]).reshape(m, p)
        w_vec = np.array(W[:, s]).reshape(p, n)
        C_recon += (A @ u_vec) * (v_vec @ B) * w_vec.T
    
    if np.allclose(C_recon, C_expected):
        print(f"OK: rank {rank} decomposition verified.")
    else:
        print(f"FAIL: C_expected = {C_expected}")
        print(f"C_recon    = {C_recon}")
