"""Rewritten program using block-wise Strassen decomposition."""
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 6 blocks * 7 multiplications = 42
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for a single 2x2x2 block
    # U contributions: list of (local_i, local_j, coeff)
    # V contributions: list of (local_j, local_k, coeff)
    # W contributions: list of (local_k, local_i, coeff)
    
    strassen_U = [
        [(0, 0, 1), (1, 1, 1)],      # M1
        [(1, 0, 1), (1, 1, 1)],      # M2
        [(0, 0, 1)],                 # M3
        [(1, 1, 1)],                 # M4
        [(0, 0, 1), (0, 1, 1)],      # M5
        [(1, 0, 1), (0, 0, -1)],     # M6
        [(0, 1, 1), (1, 1, -1)]      # M7
    ]
    
    strassen_V = [
        [(0, 0, 1), (1, 1, 1)],      # M1
        [(0, 0, 1)],                 # M2
        [(1, 0, 1), (1, 1, -1)],     # M3
        [(0, 1, 1), (0, 0, -1)],     # M4
        [(1, 1, 1)],                 # M5
        [(0, 0, 1), (0, 1, 1)],      # M6
        [(1, 0, 1), (1, 1, 1)]       # M7
    ]
    
    strassen_W = [
        [(0, 0, 1), (1, 1, 1)],      # M1
        [(0, 1, 1), (1, 1, -1)],     # M2
        [(1, 0, 1), (1, 1, 1)],      # M3
        [(0, 0, 1), (0, 1, 1)],      # M4
        [(0, 0, -1), (1, 0, 1)],     # M5
        [(1, 1, 1)],                 # M6
        [(0, 0, 1)]                  # M7
    ]
    
    # Define the 6 blocks
    # Each block is defined by:
    # j_offset: starting index of j in global A/B
    # k_offset: starting index of k in global B/C
    # The blocks are:
    # Left part (j=0,1): k=0,1; k=2,3; k=4,5
    # Right part (j=2,3): k=0,1; k=2,3; k=4,5
    
    blocks = [
        (0, 0), (0, 2), (0, 4),
        (2, 0), (2, 2), (2, 4)
    ]
    
    col = 0
    for j_off, k_off in blocks:
        for m_idx in range(7):
            # Update U
            for li, lj, coeff in strassen_U[m_idx]:
                global_i = li
                global_j = j_off + lj
                U[global_i * m + global_j, col] += coeff
                
            # Update V
            for lj, lk, coeff in strassen_V[m_idx]:
                global_j = j_off + lj
                global_k = k_off + lk
                V[global_j * p + global_k, col] += coeff
                
            # Update W
            for lk, li, coeff in strassen_W[m_idx]:
                global_k = k_off + lk
                global_i = li
                W[global_k * n + global_i, col] += coeff
                
            col += 1
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Verify
    # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # Expected: A[i,j] B[j,k]
    
    # Random test
    np.random.seed(42)
    A = np.random.randint(-10, 10, size=(n, m))
    B = np.random.randint(-10, 10, size=(m, p))
    C_expected = A @ B
    
    C_computed = np.zeros((n, p), dtype=int)
    for i in range(n):
        for k in range(p):
            s = 0
            for c in range(len(U[0])):
                for j in range(m):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            C_computed[i, k] = s
            
    if np.allclose(C_expected, C_computed):
        print("Decomposition is valid.")
    else:
        print("Decomposition is INVALID.")
        print(C_expected)
        print(C_computed)
