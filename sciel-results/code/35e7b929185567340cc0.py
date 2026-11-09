"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = delta(i,j) * delta(j,k) in the right form
    # Actually, the decomposition should satisfy:
    # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i,j,k
    # Wait, that's not right. Let me re-read the convention.
    
    # The tensor for matrix multiplication: T_{i,j,k} such that
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition: sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    # should equal the tensor entry which is 1 if the indices are consistent.
    
    # Actually, for the standard tensor decomposition of matrix mult:
    # The tensor is T[i,j,k] = 1 for all valid i,j,k (since each product A[i,j]*B[j,k] contributes to C[i,k])
    # But that's not quite right either.
    
    # Let me think again. The matrix multiplication tensor has entries:
    # T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
    # So for fixed i,k, the sum over j gives C[i,k].
    
    # The decomposition: for each c, we have U[:,c], V[:,c], W[:,c]
    # U has shape (n*m, rank), indexed by (i,j)
    # V has shape (m*p, rank), indexed by (j,k)  
    # W has shape (p*n, rank), indexed by (k,i)
    
    # The condition is: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # (since T[i,j,k] = 1 for all i,j,k in the standard decomposition)
    
    # Wait, that's the schoolbook where each term is separate.
    # In the schoolbook, for each (i,j,k) triple, there's a separate column c with
    # U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1, and all other entries 0.
    # Then sum_c gives 1 for each (i,j,k) since only one term is nonzero.
    
    # So the verification is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)  
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m + j]
                v_val = V_arr[j*p + k]
                w_val = W_arr[k*n + i]
                s = np.dot(u_val, v_val * w_val)  # sum_c U[c]*V[c]*W[c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to reduce rank by attempting to merge columns
    # For now, return schoolbook as baseline
    return result

# EVOLVE-BLOCK-END
