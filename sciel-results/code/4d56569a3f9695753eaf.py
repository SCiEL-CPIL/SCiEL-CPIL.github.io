"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.linalg import svd
import time
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the multiplication tensor T[i,j,k,l] = delta_{i,k} * delta_{j,l} 
    # where i in [0,n), j in [0,m), k in [0,p), l in [0,m)
    # Actually for (n x m) * (m x p): T[i, j, k, l] = delta_{j,k} * delta_{i,l}
    # where i in [0,n), j in [0,m), k in [0,m), l in [0,p)
    # The decomposition: T[i,j,k,l] = sum_r U[i*m+j, r] * V[k*m+l, r] * W[... ]
    # Wait, let me reconsider the index convention.
    
    # For multiplying A (n x m) by B (m x p) to get C (n x p):
    # C[i, l] = sum_j A[i,j] * B[j,l]
    # Tensor T[i, j, l] = delta_{j,l} ... no
    # T[i, j, l, k] = delta_{j,k} * delta_{i,l}
    # where i in [0,n), j in [0,m), k in [0,m), l in [0,p)
    # U shape: (n*m, rank) - indexed by (i, j)
    # V shape: (m*p, rank) - indexed by (k, l) where k is the shared index, l is column of B
    # W shape: (p*n, rank) - indexed by (l, i) where l is output col, i is output row
    
    # Check: sum_r U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r] should equal delta_{j,k} * delta_{i,l}
    # Wait, that doesn't look right dimensionally. Let me re-read the task.
    
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) where k is the column index of B (which equals row index of output), 
    # and i is the row index of A (which equals row index of output).
    # Hmm, but W shape is (p*n, rank). So W[k*n + i, r] where k in [0,p), i in [0,n).
    
    # Let me think about this differently. The standard tensor decomposition for matrix multiplication:
    # (A B)[i, k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i, j, k, l] = delta_{j,l} * delta_{i,k}
    # where i in [0,n), j in [0,m), k in [0,p), l in [0,m)
    
    # Decomposition: T[i,j,k,l] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # But wait, U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # So U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j,?} ... 
    
    # Actually the correct identity is:
    # (AB)[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, j', k) where we need j = j' for the product to contribute.
    # T[i, j, j', k] = delta_{j,j'} * delta_{i,k}... no, that's not right either.
    
    # Let me use the standard formulation:
    # The matrix multiplication tensor T: T[i, j, k, l] = 1 if j=k and i=l, else 0
    # where i in [0,n), j in [0,m), k in [0,m), l in [0,p)
    # Wait no, for (n x m) * (m x p) -> (n x p):
    # C[i,l] = sum_j A[i,j] B[j,l]
    # So T[i, j, j', l] = delta_{j,j'} * delta_{i,l}
    # where i in [0,n), j in [0,m), j' in [0,m), l in [0,p)
    
    # With the given factor shapes:
    # U: (n*m, rank) -> U[i*m + j, r]
    # V: (m*p, rank) -> V[j'*p + l, r]  (but V should be indexed by the shared index and column)
    # W: (p*n, rank) -> W[l*n + i, r]
    
    # So: sum_r U[i*m+j, r] * V[j'*p+l, r] * W[l*n+i, r] = delta_{j,j'} * delta_{i,l}
    # Hmm, but V is (m*p, rank). If V is indexed by (j', l) where j' in [0,m) and l in [0,p),
    # then V[j'*p + l, r] makes sense.
    
    # But wait, the task says V has shape (m*p, rank). And the shared index is j (row of B, col of A).
    # So V should be indexed by (j, l) where j in [0,m) and l in [0,p).
    # V[j*p + l, r]
    
    # And W is indexed by (l, i) where l in [0,p) and i in [0,n).
    # W[l*n + i, r]
    
    # So the decomposition is:
    # sum_r U[i*m+j, r] * V[j*p+l, r] * W[l*n+i, r] = delta_{j,j} * delta_{i,l}... 
    # That's not quite right. Let me think again.
    
    # Actually, I think the correct identity for the tensor decomposition is:
    # For the multiplication tensor, we want:
    # T[i, j, k, l] = sum_r U[i*m+j, r] * V[k*m+l, r] * W[... ]
    # But V is (m*p, rank), not (m*m, rank).
    
    # I think I'm overcomplicating this. Let me just verify numerically by checking
    # that the reconstructed matrix product matches for random inputs.
    
    # Actually, the simplest verification: for any matrices A (n x m) and B (m x p),
    # the product C = A @ B should equal the reconstruction from U, V, W.
    
    # The reconstruction formula based on the factor shapes:
    # C[i, l] = sum_j sum_r U[i*m+j, r] * V[j*p+l, r] * W[l*n+i, r]
    # Wait, that has an extra sum over j and r, and W depends on i and l.
    
    # I think the correct interpretation is:
    # C[i, l] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[k*p+l, r] * B[k,l]) * W[l*n+i, r]
    # No, that doesn't make sense either because W would be multiplying scalars.
    
    # Let me look at this from the tensor perspective more carefully.
    # The matrix multiplication tensor T is a 4-tensor with T[i,j,k,l] = delta_{j,k} * delta_{i,l}
    # where i in [0,n), j in [0,m), k in [0,m), l in [0,p)
    # 
    # A rank-r decomposition writes T as sum_{s=1}^{r} u_s ⊗ v_s ⊗ w_s ⊗ z_s
    # where u_s is in R^n, v_s in R^m, w_s in R^m, z_s in R^p
    # 
    # But the given factorization has U (n*m, r), V (m*p, r), W (p*n, r)
    # This suggests a different grouping of indices.
    
    # Perhaps: T[i,j,k,l] = sum_s U[i*m+j, s] * V[k*p+l, s] * W[? , s]
    # But that's only 3 factors for a 4-tensor. Unless W is a scalar...
    
    # Oh wait! I think the decomposition is:
    # The multiplication can be written as:
    # C = A @ B
    # C[i,l] = sum_j A[i,j] B[j,l]
    # 
    # This is a bilinear map. The tensor decomposition for bilinear maps:
    # T: R^{n*m} x R^{m*p} -> R^{n*p}
    # T(a, b) = sum_r (U[:,r]^T a) * (V[:,r]^T b) * w_r
    # where w_r is in R^{n*p}
    # 
    # So: C[i,l] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[k*p+l, r] * B[k,l]) * W_flat[i*p+l, r]
    # 
    # But W has shape (p*n, rank), not (n*p, rank). And the convention is W[k*n+i, r].
    # So W_flat would be W[l*n+i, r] if we flatten C as (l, i) order... but C is (n, p) so 
    # flattening row-major gives C[i,l] at position i*p+l.
    # 
    # Hmm, W[k*n+i, r] where k in [0,p) and i in [0,n). So W is indexed by (output_col, output_row).
    # So W_flat[i, l] = W[l*n + i, r]... no, W[l*n + i, r] gives a scalar for each r.
    # 
    # I think the correct formula is:
    # C[i, l] = sum_r U[i*m+j, r] * V[j*p+l, r] * W[l*n+i, r] summed over j
    # No wait, that sums over j but U depends on j and V depends on j.
    
    # Let me just test with the schoolbook decomposition to figure out the convention.
    
    # For schoolbook: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # where col corresponds to (i,j,k)
    
    # So for a given (i,j,k):
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1  
    # W[k*n+i, col] = 1
    
    # The product term for this column is:
    # sum_j' U[i*m+j', col] * V[j'*p+k, col] * W[k*n+i, col]
    # = sum_j' delta_{j',j} * 1 * 1  (if V is indexed correctly)
    # 
    # Wait, I think V should be V[j*p+k, col] where j is the shared index.
    # In schoolbook, V[j*p+k, col] = 1 when the column corresponds to (i,j,k).
    
    # So the reconstruction would be:
    # C[i,k] = sum_j sum_col U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
    # For schoolbook, only the column where (i,j,k) matches contributes:
    # C[i,k] = sum_j 1 * 1 * 1 = sum_j delta... 
    # 
    # Actually for a fixed (i,k), the only nonzero terms are when the column has the same i and k.
    # For column (i', j', k'): U[i*m+j, col] = 1 if i'=i and j'=j
    # V[j*p+k, col] = 1 if j'=j and k'=k
    # W[k*n+i, col] = 1 if k'=k and i'=i
    # 
    # So for fixed (i,k), summing over j and col:
    # Only col = (i,j,k) for any j contributes, and only if i'=i, k'=k, j'=j
    # So C[i,k] = sum_j 1 = m
    # 
    # That's not right. We should get C[i,k] = sum_j A[i,j] B[j,k].
    
    # I think the actual formula involves multiplying by the input matrices:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]
    # 
    # For schoolbook with column (i0,j0,k0):
    # sum_j U[i*m+j, col] * A[i,j] = A[i,j0] (only j=j0 contributes)
    # sum_j' V[j'*p+k, col] * B[j',k] = B[j0,k] (only j'=j0 contributes)
    # W[k*n+i, col] = 1 (if i=i0 and k=k0)
    # 
    # So C[i,k] = sum_{i0,j0,k0} A[i,j0] * B[j0,k] * delta_{i,i0} * delta_{k,k0}
    # = sum_{j0} A[i,j0] * B[j0,k] * 1  (when i0=i, k0=k)
    # = sum_{j0} A[i,j0] * B[j0,k]
    # = (A@B)[i,k] ✓
    
    # Great! So the formula is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]
    
    # Let me verify this numerically.
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute reconstruction
    C_reconstructed = np.zeros((n, p))
    for r in range(rank):
        U_vec = U[:, r]  # (n*m,)
        V_vec = V[:, r]  # (m*p,)
        W_val = W[:, r]  # (p*n,)
        
        # sum_j U[i*m+j, r] * A[i,j] for each i
        U_A = np.zeros(n)
        for i in range(n):
            for j in range(m):
                U_A[i] += U[i * m + j, r] * A[i, j]
        
        # sum_j' V[j'*p+k, r] * B[j',k] for each k
        V_B = np.zeros(p)
        for k in range(p):
            for j in range(m):
                V_B[k] += V[j * p + k, r] * B[j, k]
        
        # W[k*n+i, r] for each (i,k)
        for i in range(n):
            for k in range(p):
                C_reconstructed[i, k] += U_A[i] * V_B[k] * W[k * n + i, r]
    
    return np.allclose(C_reconstructed, C_expected, atol=1e-10)

def _try_rank_reduction(n, m, p, target_rank, max_iter=1000, seed=42):
    """Try to find a rank-target_rank decomposition using iterative refinement."""
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    
    # Initialize with random factors
    U = rng.randn(n * m, target_rank)
    V = rng.randn(m * p, target_rank)
    W = rng.randn(p * n, target_rank)
    
    best_loss = float('inf')
    best_factors = (U.copy(), V.copy(), W.copy())
    
    for it in range(max_iter):
        # Compute loss using random test matrices
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        C_reconstructed = np.zeros((n, p))
        for r in range(target_rank):
            U_vec = U[:, r]
            V_vec = V[:, r]
            W_val = W[:, r]
            
            U_A = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    U_A[i] += U[i * m + j, r] * A[i, j]
            
            V_B = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    V_B[k] += V[j * p + k, r] * B[j, k]
            
            for i in range(n):
                for k in range(p):
                    C_reconstructed[i, k] += U_A[i] * V_B[k] * W[k * n + i, r]
        
        loss = np.sum((C_reconstructed - C_expected) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_factors = (U.copy(), V.copy(), W.copy())
        
        if best_loss < 1e-10:
            break
        
        # Gradient step (simplified)
        lr = 0.01 / (1 + it * 0.01)
        
        # Compute gradients numerically or use a simple update
        # For simplicity, use a random perturbation with acceptance
        U_new = U + lr * rng.randn(n * m, target_rank)
        V_new = V + lr * rng.randn(m * p, target_rank)
        W_new = W + lr * rng.randn(p * n, target_rank)
        
        # Evaluate new loss
        C_new = np.zeros((n, p))
        for r in range(target_rank):
            U_A = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    U_A[i] += U_new[i * m + j, r] * A[i, j]
            
            V_B = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    V_B[k] += V_new[j * p + k, r] * B[j, k]
            
            for i in range(n):
                for k in range(p):
                    C_new[i, k] += U_A[i] * V_B[k] * W_new[k * n + i, r]
        
        new_loss = np.sum((C_new - C_expected) ** 2)
        
        if new_loss < loss:
            U, V, W = U_new, V_new, W_new
    
    # Check if we found an exact solution
    if best_loss < 1e-10:
        U, V, W = best_factors
        # Try to round to integers/half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
            return U_rounded, V_rounded, W_rounded, True
    
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin
    
    # Start with schoolbook
    sb = _schoolbook_decomposition(n, m, p)
    best_result = {
        'U': sb['U'].tolist(),
        'V': sb['V'].tolist(),
        'W': sb['W'].tolist()
    }
    best_rank = sb['rank']
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, n*p) = max(6, 30, 20) = 30
    # But for rectangular matrices, the rank is at least max(n, m, p) * min(n, m, p) ... 
    # Actually, the rank of the multiplication tensor for (n x m) * (m x p) is at least 
    # max(n*m, m*p, n*p) / something... Let's just try various ranks.
    
    # Try ranks from lower bound up to schoolbook rank
    min_rank = max(n * m, m * p, n * p)  # Simple lower bound
    # Actually a better lower bound: the rank must be at least max(n, m, p) * min(n, m, p)
    # For 2x3x10: max(2,3,10)*min(2,3,10) = 10*2 = 20
    # But we know schoolbook is 60, so let's try from 20 to 59
    
    for target_rank in range(20, min(60, best_rank)):
        if time.time() > deadline - 10:
            break
        
        success = False
        for seed in range(5):  # Try multiple seeds
            if time.time() > deadline - 10:
                break
            
            U, V, W, exact = _try_rank_reduction(n, m, p, target_rank, max_iter=200, seed=seed)
            if exact:
                best_rank = target_rank
                best_result = {
                    'U': U.tolist(),
                    'V': V.tolist(),
                    'W': W.tolist()
                }
                success = True
                break
        
        if success:
            # Found a lower rank, continue trying even lower
            continue
        else:
            # If we couldn't find a decomposition at this rank, try higher ranks
            # But we already have schoolbook, so just return best
            break
    
    return best_result

# EVOLVE-BLOCK-END
