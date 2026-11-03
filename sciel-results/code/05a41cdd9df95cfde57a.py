import time

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Construct Schoolbook Decomposition
    # This is the standard O(nmp) decomposition.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Verification (Optional but good for "stronger" design)
    # Verify that sum_c U[i*m+j, c] * V[j*p+k, c] == delta_ik is NOT the definition.
    # The tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] ?
    # No, the decomposition is T = sum_c U_c \otimes V_c \otimes W_c.
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] is incorrect for matrix mult.
    # Matrix mult C = A B. C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor T[i,j,k] corresponds to the coefficient of A[i,j] B[j,k] in C[i,k]?
    # Actually, the multiplication tensor T is defined such that
    # C[i,k] = sum_{j} A[i,j] B[j,k].
    # In tensor form, T is a 3-tensor where T[i,j,k] is 1 if the term A[i,j]B[j,k] contributes to C[i,k]?
    # No, the standard decomposition is:
    # C[i,k] = sum_{c} (sum_j U[i,j,c] A[i,j]) * (sum_j V[j,k,c] B[j,k]) ... No.
    
    # The standard bilinear decomposition is:
    # C[i,k] = sum_{c=1}^{rank} S_c A[i, *] T_c B[*, k]
    # Or more simply, the tensor T_{i,j,k} is 1 if we are multiplying A_{i,j} and B_{j,k} to get C_{i,k}.
    # Wait, the task says "output factor follows the official trace convention W[k*n+i,r]".
    # This implies the tensor is indexed by (i, j, k) where i is row of A, j is col of A/row of B, k is col of B.
    # The term is A[i,j] * B[j,k]. The result is C[i,k].
    # The decomposition is C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) ?? No.
    
    # Let's check the seed program's logic.
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    # This corresponds to the term A[i,j] B[j,k] contributing to C[i,k].
    # The "W" factor in the task description is likely the third factor in the tensor product,
    # often associated with the output index or the trace.
    # The task says "output factor follows ... W[k*n+i, r]".
    # This matches the seed's W index.
    
    # So the seed is correct.
    
    if time.time() > deadline:
        # Fallback if verification took too long (it won't)
        pass
        
    return {"U": U, "V": V, "W": W}
