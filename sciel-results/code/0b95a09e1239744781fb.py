"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 7]

def _schoolbook(n, m, p):
    """Return U, V, W for naive tensor decomposition of n x m @ m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = U.shape[1]
    # Build the tensor via sum over rank
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We need: for all i,j,k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Actually the convention is:
    # A is n x m, B is m x p, C = A @ B is n x p
    # Tensor T[i,j,k] represents: C[i,k] += A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for the schoolbook (each term is separate)
    # With factors: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T[i,j,k] = 1 for all i,j,k (since in schoolbook each (i,j,k) triple has exactly one 1)
    
    # Let's compute T from factors
    T = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]  # shape (n*m,)
        v_col = V[:, r]  # shape (m*p,)
        w_col = W[:, r]  # shape (p*n,)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
    
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=tol, rtol=0):
        return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook baseline
    U_school, V_school, W_school = _schoolbook(n, m, p)
    school_rank = n * m * p
    
    # Try to find a lower rank decomposition using randomized approaches
    # Strategy: Start with random low-rank tensors and check if they can represent the identity tensor
    # The tensor for matrix multiplication T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # For 2x4x7, schoolbook rank = 56
    # Known bounds: rank >= max(n+m-1, m+p-1, n+p-1) * something... 
    # Actually the Strassen bound doesn't directly apply here.
    # Lower bound: rank >= n*m (for each k), rank >= m*p (for each i), rank >= n*p (for each j)
    # So rank >= max(8, 28, 14) = 28
    
    # Let's try rank values from a lower bound up to schoolbook
    # But we need to be time-conscious
    
    deadline = time.time() + 120  # 120 second limit, leave some margin
    
    # Try a simple approach: use the fact that T[i,j,k] = 1
    # This is equivalent to: the tensor is the all-ones tensor
    # The all-ones tensor of shape n x m x p has a trivial decomposition of rank 1!
    # Wait, that's not right. Let me reconsider.
    
    # Actually, for matrix multiplication A (n x m) @ B (m x p) = C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k
    # We need: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * ... 
    # No, the tensor decomposition is:
    # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_k' B[j,k'] * V[j*p+k', r]) ... 
    # That's not quite right either.
    
    # The correct formulation:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_{j,r} A[i,j] B[j,k] delta_{jr} ... no
    # 
    # In tensor decomposition form:
    # C[i,k] = sum_r U[i*m+?, r] ... 
    # 
    # Actually, the standard formulation is:
    # We want to compute C = A @ B where A is n x m, B is m x p
    # The multiplication tensor T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The rank decomposition means:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # And then: C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r A[i,j] * B[j,k] * T[i,j,k]... 
    # No, that's not how it works.
    #
    # Let me think again. The tensor multiplication formula is:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # 
    # With a rank-r decomposition:
    # C[i,k] = sum_{r=0}^{r-1} (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) ... 
    # That's not right either.
    #
    # The correct way: 
    # Define intermediate quantities:
    # For each r:
    #   x_r = sum_j A[i,j] * U[i*m+j, r]  -- but this depends on i and j
    # 
    # Actually, I think the formulation is:
    # C[i,k] = sum_{r} W[k*n+i, r] * (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r])
    # No, that would be O(r * m * (n*p)) which is too much.
    #
    # Let me look at this more carefully. The matrix multiplication tensor is:
    # T_{ijk} = 1 for all i,j,k
    # 
    # A rank-1 tensor is U_i * V_j * W_k
    # So T_{ijk} = sum_r U_i * V_j * W_k
    #
    # But in our case, the indices are coupled:
    # The tensor is indexed by (i, j, k) but the factor matrices have coupled indices:
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    #
    # So the condition is:
    # 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k
    #
    # And the matrix multiplication is computed as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The way the factors are used is:
    # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j' B[j',k] * V[j'*p+k, r]) * W[k*n+i, r]
    # Hmm, that doesn't look right either because the sums over j and j' are independent.
    #
    # I think the correct interpretation is:
    # For each r, compute:
    #   P_r = A @ diag(U[:,r]) ... no
    #
    # Let me just use the verification function and trust that the schoolbook works.
    # The schoolbook has U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1 for each (i,j,k) triple.
    # Then: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 if (i,j,k) matches the column, else 0.
    # So T[i,j,k] = 1. Correct.
    #
    # Now, can we do better? The all-ones tensor T[i,j,k]=1 can be written as:
    # T[i,j,k] = sum_r (a_r[i]) * (b_r[j]) * (c_r[k])
    # If we could use decoupled indices, rank 1 would suffice: a[i]=1, b[j]=1, c[k]=1.
    # But we have coupled indices, so it's more complex.
    
    # For now, let's just return the schoolbook decomposition.
    # The task says score is the rank, so lower is better.
    # Schoolbook rank = 56.
    
    U = U_school.tolist()
    V = V_school.tolist()
    W = W_school.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
