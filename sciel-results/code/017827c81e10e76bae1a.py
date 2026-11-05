"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i
        
        # M1: (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2: (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        
        # M3: A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4: A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5: (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M6: (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7: (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # 6 Strassen blocks for 2x4x6 (k=0..5)
    add_strassen(0, 0, 0)  # A(0:2,0:2), B(0:2,0:2) -> C(0:2,0:2)
    add_strassen(0, 0, 2)  # A(0:2,0:2), B(0:2,2:4) -> C(0:2,2:4)
    add_strassen(0, 2, 0)  # A(0:2,2:4), B(2:4,0:2) -> C(0:2,0:2)
    add_strassen(0, 2, 2)  # A(0:2,2:4), B(2:4,2:4) -> C(0:2,2:4)
    add_strassen(0, 0, 4)  # A(0:2,0:2), B(0:2,4:6) -> C(0:2,4:6)
    add_strassen(0, 2, 4)  # A(0:2,2:4), B(2:4,4:6) -> C(0:2,4:6)
    
    # 8 rank-1 terms for k=6
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
    
    # Verify decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def verify(U_arr, V_arr, W_arr):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = float(U_arr[i*m+j] @ (V_arr[j*p+k] * W_arr[k*n+i]))
                    err = abs(lhs - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    max_err = verify(U_np, V_np, W_np)
    if max_err > 1e-9:
        print(f"Verification failed: max_err={max_err}")
    
    # Try to reduce rank by checking for linear dependencies
    # Build the tensor and check column rank
    # Each rank-1 term is a tensor T_r = u_r ⊗ v_r ⊗ w_r
    # The decomposition is sum_r T_r
    # Check if any column is redundant
    
    # Reshape to get the tensor: for each (i,j,k), the coefficient is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to find if we can remove a column r such that the tensor is unchanged
    
    # Try removing each column and check if the decomposition still holds
    rank = col
    best_rank = rank
    best_U, best_V, best_W = U_np, V_np, W_np
    
    # Check linear dependencies by trying to remove each column
    for r in range(rank):
        if time.time() > DEADLINE - 30:
            break
        U_test = np.delete(U_np, r, axis=1)
        V_test = np.delete(V_np, r, axis=1)
        W_test = np.delete(W_np, r, axis=1)
        err = verify(U_test, V_test, W_test)
        if err < 1e-9:
            best_rank = rank - 1
            best_U, best_V, best_W = U_test, V_test, W_test
            # Reset and try to remove more
            U_np, V_np, W_np = U_test, V_test, W_test
            rank = best_rank
            r = 0  # Restart from beginning since indices shifted
            continue
    
    # Convert back to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
