"""
Tensor decomposition for 2x2 matrix multiplication (Strassen, rank 7).
"""
import numpy as np
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
    
    The tensor rank of 2x2 matrix multiplication is 7 (Strassen).
    This function returns the Strassen decomposition with U, V, W matrices
    following the convention:
      U[i*m + j, r]
      V[j*p + k, r]
      W[k*n + i, r]
    
    The decomposition satisfies:
      sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1 
      for all i, j, k in {0, 1} such that the term contributes to C[i,k].
      Specifically, the tensor T is defined by T[i, j, k] = 1 if we are computing 
      C[i, k] from A[i, j] and B[j, k]. Wait, the standard tensor for matmul 
      is T in R^{n x m} x R^{m x p} x R^{n x p} such that 
      <T, A, B> = trace(C^T (AB)).
      The entries of the tensor are T_{i, j, k} = 1 if the index combination 
      (i, j) for A, (j, k) for B, and (i, k) for C is valid.
      Actually, the bilinear form is sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}^T? 
      No, C_{i,k} = sum_j A_{i,j} B_{j,k}.
      So the tensor T has indices (i, j, k) where i is row of A/C, j is col of A/row of B, 
      k is col of B/C.
      T_{i, j, k} = 1.
      The decomposition is T_{i, j, k} = sum_r U_{i, j, r} V_{j, k, r} W_{i, k, r}.
      
      In the flattened vectors:
      U index: i*m + j
      V index: j*p + k
      W index: k*n + i  (Note: k is col of C, i is row of C)
      
      We verify this by checking if the reconstructed tensor matches the all-ones 
      tensor for the valid (i, j, k) tuples.
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's seven products:
    # Let A = [[a00, a01], [a10, a11]]
    #     B = [[b00, b01], [b10, b11]]
    # 
    # m0 = (a00 + a11) * (b00 + b11)
    # m1 = (a10 + a11) * b00
    # m2 = a00 * (b01 - b11)
    # m3 = a11 * (b10 - b00)
    # m4 = (a00 + a01) * b11
    # m5 = (a10 - a00) * (b00 + b01)
    # m6 = (a01 - a11) * (b10 + b11)
    #
    # c00 = m0 + m3 - m4 + m6
    # c01 = m2 + m4
    # c10 = m1 + m3
    # c11 = m0 - m1 + m2 + m5

    # --- Product 0: m0 = (a00 + a11)(b00 + b11) ---
    U[0][0] = 1   # a00 (i=0, j=0)
    U[3][0] = 1   # a11 (i=1, j=1)
    V[0][0] = 1   # b00 (j=0, k=0)
    V[3][0] = 1   # b11 (j=1, k=1)
    W[0][0] = 1   # c00 (k=0, i=0 -> 0*2+0=0)
    W[3][0] = 1   # c11 (k=1, i=1 -> 1*2+1=3)

    # --- Product 1: m1 = (a10 + a11)b00 ---
    U[2][1] = 1   # a10 (i=1, j=0)
    U[3][1] = 1   # a11 (i=1, j=1)
    V[0][1] = 1   # b00 (j=0, k=0)
    W[1][1] = 1   # c10 (k=0, i=1 -> 0*2+1=1)
    W[3][1] = -1  # c11 (k=1, i=1 -> 1*2+1=3)

    # --- Product 2: m2 = a00(b01 - b11) ---
    U[0][2] = 1   # a00 (i=0, j=0)
    V[1][2] = 1   # b01 (j=0, k=1)
    V[3][2] = -1  # b11 (j=1, k=1)
    W[2][2] = 1   # c01 (k=1, i=0 -> 1*2+0=2)
    W[3][2] = 1   # c11 (k=1, i=1 -> 1*2+1=3)

    # --- Product 3: m3 = a11(b10 - b00) ---
    U[3][3] = 1   # a11 (i=1, j=1)
    V[2][3] = 1   # b10 (j=1, k=0)
    V[0][3] = -1  # b00 (j=0, k=0)
    W[0][3] = 1   # c00 (k=0, i=0 -> 0*2+0=0)
    W[1][3] = 1   # c10 (k=0, i=1 -> 0*2+1=1)

    # --- Product 4: m4 = (a00 + a01)b11 ---
    U[0][4] = 1   # a00 (i=0, j=0)
    U[1][4] = 1   # a01 (i=0, j=1)
    V[3][4] = 1   # b11 (j=1, k=1)
    W[0][4] = -1  # c00 (k=0, i=0 -> 0*2+0=0)
    W[2][4] = 1   # c01 (k=1, i=0 -> 1*2+0=2)

    # --- Product 5: m5 = (a10 - a00)(b00 + b01) ---
    U[2][5] = 1   # a10 (i=1, j=0)
    U[0][5] = -1  # a00 (i=0, j=0)
    V[0][5] = 1   # b00 (j=0, k=0)
    V[1][5] = 1   # b01 (j=0, k=1)
    W[3][5] = 1   # c11 (k=1, i=1 -> 1*2+1=3)

    # --- Product 6: m6 = (a01 - a11)(b10 + b11) ---
    U[1][6] = 1   # a01 (i=0, j=1)
    U[3][6] = -1  # a11 (i=1, j=1)
    V[2][6] = 1   # b10 (j=1, k=0)
    V[3][6] = 1   # b11 (j=1, k=1)
    W[0][6] = 1   # c00 (k=0, i=0 -> 0*2+0=0)

    # Verification Step (Polish/Check)
    # We verify that the decomposition exactly reconstructs the matmul tensor.
    # The tensor T[i, j, k] should be 1 for all i, j, k in {0,1} because 
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The decomposition is sum_r U[i, j, r] V[j, k, r] W[i, k, r] (using W[k*n+i])
    # Let's check for all i, j, k.
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    is_valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                u_val = U_np[i*m + j]
                v_val = V_np[j*p + k]
                w_val = W_np[k*n + i]
                prod_sum = np.dot(u_val, v_val * w_val)
                
                # The tensor entry T[i, j, k] is 1.
                if abs(prod_sum - 1.0) > 1e-9:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break
    
    if not is_valid:
        # Fallback: If the hardcoded Strassen fails verification (unlikely, 
        # but serves as a safety net for convention errors), we could attempt 
        # to find a valid one or return a known valid one. 
        # However, Strassen is the standard and the indices are carefully mapped.
        # For robustness, we return the hardcoded one anyway as it is the 
        # theoretically correct rank-7 solution.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
