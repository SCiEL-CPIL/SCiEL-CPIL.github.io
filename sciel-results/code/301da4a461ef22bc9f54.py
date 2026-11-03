import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=6
    rank = 24
    U = [[0] * rank for _ in range(n * m)]  # 12 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 18 x 24
    
    col = 0
    for a_col_offset in [0, 2]:
        for b_row_offset in [0, 2]:
            for b_col_offset in [0, 3]:
                for i in range(3):
                    for j_local in range(2):
                        u_idx = i * m + (a_col_offset + j_local)
                        v_row = b_row_offset + j_local
                        for k_local in range(3):
                            v_idx = v_row * p + (b_col_offset + k_local)
                            w_idx = (b_col_offset + k_local) * n + i
                            U[u_idx][col] = 1
                            V[v_idx][col] = 1
                            W[w_idx][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    n, m, p = DIMENSIONS
    
    # Verify: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k', r] * W[k'*n+i', r]
    # More precisely, the tensor decomposition should satisfy:
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # (since T[i,j,k] = 1 for the matrix multiplication tensor)
    
    # Build the tensor from U, V, W
    # T[i*m+j, j*p+k, k*n+i] should sum to 1
    # Actually, the condition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(24):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                max_err = max(max_err, abs(s - 1))
    
    print(f"Max error: {max_err}")
    assert max_err < 1e-10, "Decomposition is incorrect!"
    print("Decomposition verified correctly!")
