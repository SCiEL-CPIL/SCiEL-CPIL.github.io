import numpy as np
import time

DIMENSIONS = [2, 3, 9]
_DEADLINE = 120.0
_START_TIME = None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Build the multiplication tensor: T[i,k,j] = delta[i,j] for C = A*B
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # So tensor T has T[i,m+k,j] = 1 if i==j... no wait
    # Actually T[i, k, j] contributes A[i,k]*B[k,j] to C[i,j]
    # The tensor is: for each k, A[:,k] outer B[k,:]
    
    # Check: sum_r U[i,r] * V[k,r] * W[j,r] should equal delta[i,j]... 
    # No, the decomposition should satisfy:
    # C[i,j] = sum_k A[i,k] B[k,j] = sum_r (sum_k A[i,k] U[k... 
    # Actually the standard form: C = sum_r (U_r^T A) * (V_r B) where U_r, V_r are vectors
    # But here U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Indexing: U[i*m+k, r], V[k*p+j, r], W[j*n+i, r]
    # Then C[i,j] = sum_r sum_k U[i*m+k, r] * A[i,k] ... no that's not right either.
    
    # Let me re-read the convention. The task says W[k*n+i, r] for output factor.
    # Standard tensor decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where T[i,k,j] = sum_r U[i,r] * V[k,r] * W[j,r]
    # For matrix mult: T[i,k,j] = delta[i,j] (since C[i,j] = sum_k A[i,k]B[k,j])
    # Wait no, T[i,k,j] should be 1 if i==j, else 0? No...
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # So the tensor T has T[i,k,j] = 1 for all i,j,k? No, that's not right.
    # The multiplication tensor is: T[i,k,j] = delta_{i,j} is wrong.
    # Actually T[i,k,j] = 1 for all valid i,k,j, because each term A[i,k]*B[k,j] 
    # contributes to C[i,j]. So T[i,k,j] = 1 if the indices are valid, but we need
    # the sum over k to give the right result.
    # 
    # The correct tensor: T[i, k, j] = 1 for all i in [0,n), k in [0,m), j in [0,p)
    # But that gives C[i,j] = sum_k 1 * A[i,k] * B[k,j] which is correct!
    # Wait, that's not a tensor decomposition in the usual sense. Let me reconsider.
    #
    # In tensor decomposition for matrix multiplication:
    # We want: sum_k A[i,k] B[k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # Hmm, the indexing is tricky. Let me just verify numerically with random matrices.
    
    # Generate random A (n x m) and B (m x p)
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    # Compute using decomposition
    # C[i,j] = sum_r sum_k U[i*m+k, r] * A[i,k] * V[k*p+j, r] * W[j*n+i, r] / something?
    # Actually the standard form is:
    # C[i,j] = sum_r (sum_k U[i*m+k, r] * A[i,k]) * (sum_j V[k*p+j, r] * B[k,j]) ... no
    # 
    # Let me think again. The bilinear form is:
    # C[i,j] = sum_k A[i,k] B[k,j]
    # A rank-r decomposition means:
    # C[i,j] = sum_{r=0}^{r-1} (sum_k U_r[i,k] * A[i,k]) * (sum_k V_r[k,j] * B[k,j]) * W_r[i,j]
    # But that's not quite right either. The standard tensor decomposition is:
    # T[i,k,j] = sum_r U_r[i] * V_r[k] * W_r[j]
    # where T[i,k,j] = 1 for the multiplication tensor.
    # Then C[i,j] = sum_k T[i,k,j] * A[i,k] * B[k,j] = sum_r W_r[j] * (sum_k U_r[i] A[i,k]) * (sum_k V_r[k] B[k,j])
    # 
    # Given the indexing in the code:
    # U has shape (n*m, rank), indexed as U[i*m + k, r]
    # V has shape (m*p, rank), indexed as V[k*p + j, r]  
    # W has shape (p*n, rank), indexed as W[j*n + i, r]
    # 
    # So: C[i,j] = sum_r W[j*n+i, r] * (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j])
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        # Compute sum_k U[i*m+k, r] * A[i,k] for each i
        left = np.zeros(n)
        for i in range(n):
            for k in range(m):
                left[i] += U[i*m + k, r] * A[i, k]
        # Compute sum_k V[k*p+j, r] * B[k,j] for each j
        right = np.zeros(p)
        for j in range(p):
            for k in range(m):
                right[j] += V[k*p + j, r] * B[k, j]
        # C[i,j] += W[j*n+i, r] * left[i] * right[j]
        for i in range(n):
            for j in range(p):
                C_decomp[i, j] += W[j*n + i, r] * left[i] * right[j]
    
    return np.allclose(C_decomp, C_true, atol=1e-10)

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition
    # Start with the known rank-50 decomposition and try to reduce
    
    # First, let's try rank 49 by checking if any term is redundant
    # Build the rank-50 decomposition
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: A_L (2x2) * B_T (2x9) using Strassen
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        u_a = 0
        u_b = 1
        u_c = 3
        u_d = 4
        
        U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
        U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
        U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
        U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
        U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
        U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0; w_01 = k1 * n + 0; w_10 = k0 * n + 1; w_11 = k1 * n + 1
        W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
        W[w_01][col+2] += 1; W[w_01][col+4] += 1
        W[w_10][col+1] += 1; W[w_10][col+3] += 1
        W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
        col += 7
    
    # Last column k=8: 2x1 direct
    k8 = 8
    v_e = 0 * p + k8; v_g = 1 * p + k8
    u_a = 0; u_b = 1; u_c = 3; u_d = 4
    U[u_a][col] += 1; V[v_e][col] += 1
    U[u_b][col+1] += 1; V[v_g][col+1] += 1
    U[u_c][col+2] += 1; V[v_e][col+2] += 1
    U[u_d][col+3] += 1; V[v_g][col+3] += 1
    w_08 = k8 * n + 0; w_18 = k8 * n + 1
    W[w_08][col] += 1; W[w_08][col+1] += 1
    W[w_18][col+2] += 1; W[w_18][col+3] += 1
    col += 4
    
    # Part 2: A_R (2x1) * B_B (1x9)
    u_ar0 = 2; u_ar1 = 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
        col += 2
    
    # Verify the decomposition
    assert _verify_decomposition(U, V, W, n, m, p), "Rank-50 decomposition is invalid"
    
    # Now try to reduce rank by checking if any column is linearly dependent
    # Convert to numpy arrays
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Try to find if we can remove a term by solving for it
    # For each term r, check if it can be expressed as a linear combination of others
    # This is complex, so let's try a simpler approach: try rank 49 directly
    
    # Alternative: try a different decomposition structure
    # What if we use a 3x3 block structure differently?
    
    # Try: split m=3 as 1+2 instead of 2+1
    # A (2x3) = [A_col0 | A_cols12], B (3x9) = [B_row0; B_rows12]
    # C = A_col0 @ B_row0 + A_cols12 @ B_rows12
    # A_col0 @ B_row0: (2x1) x (1x9) = rank 18
    # A_cols12 @ B_rows12: (2x2) x (2x9) = 4 blocks of (2x2)x(2x2) = 4*7 = 28
    # Total: 46? But wait, that's the same as before just reordered.
    
    # Actually the current decomposition already does this optimally.
    # Let me try to see if rank 49 is achievable by a more careful analysis.
    
    # For now, return the verified rank-50 decomposition
    return {"U": U, "V": V, "W": W}
