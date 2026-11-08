"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # safety margin before 150s kill
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        """Check if decomposition is exact."""
        # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # and compare with A[i,j]*B[j,k] pattern
        # The multiplication tensor has T[i,j,k] = 1 if we think of it as A[i,j]*B[j,k]
        # Actually, the tensor is defined such that for matrices A (n x m) and B (m x p),
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T satisfies: T[A, B] = C, i.e., sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k]
        # So T[i,j,k] should be 1 for all i,j,k (since each product A[i,j]*B[j,k] contributes to C[i,k])
        # Wait, no. The bilinear form is: C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor T has T[i,j,k] = 1 for all i,j,k? No, that's not a tensor decomposition.
        
        # Let me reconsider. The matrix multiplication map is:
        # (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
        # This is a bilinear map. We can write it as a sum of rank-1 bilinear maps:
        # C[i,k] = sum_r sum_{j} U_r[i,j] A[i,j] V_r[j,k] B[j,k] W_r[k,i]  -- no, this is wrong.
        
        # Correct formulation: The tensor T of matrix multiplication is a trilinear form:
        # T(A, B, C) = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i]
        # where T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k], else 0.
        # Actually, T[i,j,k] = 1 for all i,j,k, because every pair (A[i,j], B[j,k]) contributes to C[i,k].
        
        # Hmm, I think the standard convention is:
        # The multiplication tensor T_{i,j,k} = 1 for all i,j,k in [n]x[m]x[p]
        # and the decomposition is: T = sum_r U_r ⊗ V_r ⊗ W_r
        # where U_r is n x m, V_r is m x p, W_r is p x n
        # and T[i,j,k] = sum_r U_r[i,j] V_r[j,k] W_r[k,i]
        
        # For schoolbook: each (i,j,k) gives a rank-1 term with U_r[i,j]=1, V_r[j,k]=1, W_r[k,i]=1, others 0.
        
        # So to verify: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1.
        # And for any (i',j',k') != (i,j,k), the sum should be 0? No, that's not right either.
        
        # Actually, the tensor T has T[i,j,k] = 1 for all i,j,k. So we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for ALL i,j,k.
        
        # Let me verify this interpretation. If T[i,j,k] = 1 for all i,j,k, then:
        # C[i,k] = sum_j A[i,j] B[j,k] * T[i,j,k] = sum_j A[i,j] B[j,k] * 1 = sum_j A[i,j] B[j,k] ✓
        
        # So yes, T[i,j,k] = 1 for all i,j,k.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try random rank reductions
    np.random.seed(42)
    for target_rank in range(60, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        # Try to find a decomposition of this rank
        # Initialize randomly
        U = np.random.randint(0, 3, size=(n*m, target_rank))
        V = np.random.randint(0, 3, size=(m*p, target_rank))
        W = np.random.randint(0, 3, size=(p*n, target_rank))
        
        # This is just a placeholder; real optimization needed
        # For now, skip and return schoolbook
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
