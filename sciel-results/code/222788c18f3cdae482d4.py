"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that U @ V^T == W for all tensor entries."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] should equal W[k*n+i, c]... 
    # Actually the tensor decomposition means:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] ... no
    # The standard form: multiplication tensor T has entries T[i,j,k] = 1 if we multiply (i,j) of A by (j,k) of B to get (i,k) of C.
    # A decomposition of rank r: T = sum_c U_c ⊗ V_c ⊗ W_c
    # where U_c is a vector of size n*m, V_c of size m*p, W_c of size p*n
    # So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But the interface says U, V, W are matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # The condition is: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] = W[k*n+i,c]... no that's not right either.
    
    # Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    
    # The matrix multiplication tensor for A(n×m) × B(m×p) = C(n×p) is a 3-tensor T of shape (n, m, p) where T[i,j,k] = 1.
    # A rank-r decomposition: T[i,j,k] = sum_{c=1}^{r} U_c[i,j] * V_c[j,k] * W_c[k,i]
    # But the output factor follows W[k*n+i, r] convention.
    # So U has shape (n*m, r) indexed as U[i*m+j, c]
    # V has shape (m*p, r) indexed as V[j*p+k, c]  
    # W has shape (p*n, r) indexed as W[k*n+i, c]
    # And the condition is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # Wait, that doesn't seem right for a bilinear algorithm. Let me think again.
    # In a bilinear algorithm for matrix multiplication, we have:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # A rank-r bilinear algorithm expresses this as:
    # C[i,k] = sum_c W_c[i,k] * (sum_j U_c[i,j] * A[i,j]) * (sum_j V_c[j,k] * B[j,k])
    # But that's not quite the standard form either.
    
    # The standard tensor decomposition form: The multiplication tensor T ∈ F^{n×m} ⊗ F^{m×p} ⊗ F^{n×p}
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ F^{n×m}, v_c ∈ F^{m×p}, w_c ∈ F^{n×p}
    # T[i,j,k] = sum_c u_c[i,j] * v_c[j,k] * w_c[i,k]
    
    # But the problem says W has shape (p*n, rank) with W[k*n+i, c]. So w_c is indexed by (k,i) not (i,k).
    # This is the "official trace convention W[k*n+i,r]".
    
    # So the condition should be: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Let's verify this.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the full tensor and check
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # For each c, we have a contribution U3[:,:,c] * V3[:,:,c] * W3[:,:,c]
    # Sum over c
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # This is a bit tricky with numpy. Let's do it element-wise.
    
    # Create index arrays
    I, J, K = np.meshgrid(range(n), range(m), range(p), indexing='ij')
    U_vals = U3[I, J, :]  # shape (n, m, p, rank)
    V_vals = V3[J, K, :]  # shape (n, m, p, rank)
    W_vals = W3[K, I, :]  # shape (n, m, p, rank)
    
    T = np.sum(U_vals * V_vals * W_vals, axis=-1)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Start with schoolbook rank 48
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # Based on the knowledge: rank-44 construction using 16 schoolbook + 4 Strassen blocks
    # Let's try to construct a rank-44 decomposition
    
    # The idea: 2x3x8 = combine 2x2x2 (Strassen, rank 7) with some structure
    # Actually, let's try a direct approach: try ranks from 44 down and see if we can find one
    
    # For now, let's try the known rank-44 construction
    # 2x3x8: We can think of this as multiplying a 2x3 by 3x8.
    # One approach: split the 3x8 into two 3x4 blocks, multiply 2x3 by each 3x4, then combine.
    # 2x3 by 3x4: rank at most 2*3*4 = 24, but we can do better.
    # Actually, let's try a different decomposition.
    
    # Let me try to use the fact that 8 = 2*4, so we can view the 3x8 as two 3x4 matrices.
    # Then C[:, 0:4] = A * B[:, 0:4] and C[:, 4:8] = A * B[:, 4:8]
    # Each 2x3 by 3x4 multiplication can be done with rank at most 2*3*4 = 24.
    # But we can use Strassen-like methods.
    
    # Alternative: view 2x3x8 as (2x2x2) ⊗ (1x3x4) approximately
    # 2x2x2 has Strassen rank 7.
    # 1x3x4: this is just a linear map from 3-dim to 4-dim, which has rank 3 (or 4).
    # Actually, 1x3x4 tensor: T[i,j,k] where i in {0}, j in {0,1,2}, k in {0,1,2,3}.
    # This is just a 3x4 matrix, which has rank 3.
    # So the Kronecker product would have rank 7 * 3 = 21? That seems too low.
    # But the Kronecker product of tensors isn't straightforward for matrix multiplication.
    
    # Let me just try to verify if rank 44 is achievable with a simple construction.
    # I'll use a heuristic search for small improvements.
    
    # For now, return the schoolbook decomposition as a safe baseline.
    # The evaluator will measure the rank.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
