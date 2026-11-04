"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s deadline as per task contract
    n, m, p = DIMENSIONS

    # Helper to verify a decomposition
    def verify(U, V, W):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        rank = U.shape[1]
        # Compute C = A * B where A is n x m, B is m x p
        # Using the tensor decomposition: C[i,k] = sum_j A[i,j] B[j,k]
        # The decomposition gives: C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * V[j*p+k, r] * W[k*n+i, r]
        # Actually, the standard form is:
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
        # Wait, let's use the matrix multiplication tensor form.
        # The tensor is T[i,j,k] = 1.
        # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k?
        # No, that's not right. The decomposition of the multiplication tensor is:
        # T = sum_r U_r ⊗ V_r ⊗ W_r
        # where U_r[i,j] = U[i*m+j, r], V_r[j,k] = V[j*p+k, r], W_r[i,k] = W[k*n+i, r] (note: W index is k*n+i)
        # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i,j,k] = 1 for all i,j,k? No, the multiplication tensor has T[i,j,k] = 1 if we're looking at the coefficient.
        # Actually, the multiplication map is (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k].
        # The tensor is defined such that <T, A ⊗ B> = C.
        # In index notation: T[i,j,k] A[i,j] B[j,k] = C[i,k].
        # So T[i,j,k] should be the coefficient, which is 1 for all i,j,k? No, that would sum over j twice.
        # Let's be precise: C[i,k] = sum_j A[i,j] B[j,k].
        # We can write this as C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])? No.
        # The correct expansion is:
        # C[i,k] = sum_r W[k*n+i, r] * U_r · A * V_r · B
        # where U_r · A = sum_j U[i*m+j, r] A[i,j] (this is a vector for fixed i, r? No, U_r is a matrix)
        # Let's define:
        # For each r, let U_r be an n x m matrix with (U_r)[i,j] = U[i*m+j, r]
        # Let V_r be an m x p matrix with (V_r)[j,k] = V[j*p+k, r]
        # Let W_r be an n x p matrix with (W_r)[i,k] = W[k*n+i, r]
        # Then the decomposition is: C = sum_r W_r * (A * U_r^T) * (V_r * B^T)? No.
        
        # The standard tensor decomposition of the multiplication tensor T_{n,m,p} is:
        # T = sum_{r=1}^{R} U_r ⊗ V_r ⊗ W_r
        # where (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U_r[i,j] V_r[j,k] W_r[i,k] (if we use i,k for W)
        # But the indices in the task are: U has shape (n*m, R), V (m*p, R), W (p*n, R).
        # The trace convention is W[k*n+i, r]. So W_r[i,k] = W[k*n+i, r].
        # The tensor condition is: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{i,i'} delta_{k,k'}? No.
        
        # Let's just test with a random A and B.
        # C = A @ B
        # We need to compute C' = sum_r W_r @ (A @ U_r^T) @ (V_r @ B^T)? This is getting complicated.
        
        # Simpler: The tensor T has T[i,j,k] = 1.
        # The decomposition is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k?
        # Let's check the schoolbook:
        # For each (i,j,k), there is one r. U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
        # Sum_r = 1*1*1 = 1. So yes, the condition is sum_r ... = 1 for all i,j,k.
        
        # So we can verify by checking if the tensor sum equals 1 everywhere.
        # But that's O(n*m*p*R). For n=4,m=5,p=5, that's 100*R.
        # We can do it faster using matrix multiplication.
        # Let's construct the tensor and check.
        
        # Actually, a faster way:
        # The multiplication tensor is the identity tensor in the sense that it represents the bilinear map.
        # We can check: for all i, k, C[i,k] should equal sum_j A[i,j] B[j,k].
        # Let's just compute the sum for a few random matrices.
        
        # But for simplicity and speed, let's just check the tensor equation for all i,j,k if R is small.
        # For R <= 100, n*m*p = 100, so 100*R operations, which is fine.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Try to find a better decomposition using random search with integer/half-integer entries
    # This is a heuristic approach. We'll try to minimize the rank.
    
    # Start with schoolbook
    best_U, best_V, best_W = schoolbook()
    best_rank = best_U.shape[1]
    
    # Time check
    if time.time() > deadline - 30:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to reduce rank by 1, 2, ... using random initialization and local search
    # This is very unlikely to work for small matrices, but we'll try.
    
    # For now, return the schoolbook solution
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
