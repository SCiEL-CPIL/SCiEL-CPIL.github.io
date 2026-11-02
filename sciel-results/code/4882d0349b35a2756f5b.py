"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    # The matrix multiplication tensor T_{i,j,k} for (n x m) * (m x p) -> (n x p)
    # has indices i in [0,n), j in [0,m), k in [0,p).
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The convention is W[k*n + i, r]
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Schoolbook rank is n*m*p = 54. We want to find a lower rank.
    # Known results: for 3x3 * 3x6, the minimum rank is likely around 27-36.
    # Let's try to find a decomposition with rank < 54.
    
    # Strategy: Use a randomized approach to find a low-rank decomposition.
    # We'll try ranks from 27 upward and see if we can find one.
    
    def verify(U, V, W, tol=1e-6):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Build the tensor T from U, V, W
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i,j,k] = 1 if j == k (wait, no)
        # Actually, the matrix multiplication tensor is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = delta_{j,j} * delta_{k,k}? No.
        # The tensor T has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
        # Actually, the standard matrix multiplication tensor is:
        # T[i,j,k] = 1 for all i,j,k? No.
        # Let me think again. The matrix multiplication tensor M_{i,j,k} is defined as:
        # M_{i,j,k} = 1 if we use A[i,j] and B[j,k] to compute C[i,k].
        # So M[i,j,k] = 1 for all valid i,j,k. Wait, that's not right either.
        
        # The correct definition: The matrix multiplication tensor T is a 3-tensor where
        # T[i,j,k] = 1 if and only if the entry (i,k) of the product C = A*B depends on
        # A[i,j] and B[j,k]. Since C[i,k] = sum_j A[i,j]*B[j,k], each term A[i,j]*B[j,k]
        # contributes to C[i,k]. So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # Wait, that would mean the tensor is all ones, which can't be right for a rank problem.
        
        # Let me reconsider. The tensor for matrix multiplication of (n x m) and (m x p) matrices
        # is a 3-tensor T in R^{n x m x p} where T[i,j,k] = 1 if j is the "middle" index that
        # connects A[i,j] and B[j,k]. Actually, T[i,j,k] = 1 for all i,j,k because every
        # (i,j) entry of A and (j,k) entry of B contributes to the product.
        
        # Hmm, I think I'm confusing myself. Let me look at the schoolbook decomposition again.
        # In the schoolbook: U[i*m+j, column] = 1, V[j*p+k, column] = 1, W[k*n+i, column] = 1
        # for each (i,j,k). So the tensor is:
        # T[i,j,k] = sum_column U[i*m+j, column] * V[j*p+k, column] * W[k*n+i, column]
        # = sum_{(i',j',k')} delta_{i,i'} delta_{j,j'} delta_{k,k'} * delta_{j,j'} delta_{k,k'} delta_{i,i'}
        # = 1 for all (i,j,k).
        
        # So the tensor is indeed all ones! That means we're looking for the rank of the
        # all-ones tensor of size n x m x p. The rank of the all-ones tensor is 1 (just one
        # term with U=all ones, V=all ones, W=all ones).
        
        # Wait, that can't be right. Let me re-read the problem.
        
        # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 6 matrix."
        
        # I think the tensor is NOT all ones. The matrix multiplication tensor T_{i,j,k} is:
        # T_{i,j,k} = 1 if the (i,k)-th entry of the product is computed using A_{i,j} and B_{j,k}.
        # Since C_{i,k} = sum_j A_{i,j} B_{j,k}, the tensor should be:
        # T_{i,j,k} = 1 for all i,j,k.
        
        # But that gives rank 1, which is trivially achievable. So I must be misunderstanding.
        
        # Let me look at this differently. Perhaps the tensor encodes the full multiplication
        # operation, not just the structure. In that case, the tensor would be:
        # T_{i,j,k} = delta_{j,j} (which is 1) * ... 
        
        # Actually, I think the issue is that the tensor for matrix multiplication is:
        # T ∈ R^{(n*m) x (m*p) x (p*n)} where T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
        # i.e., T[(i,j), (j',k), (k',i')] = 1 if j=j', k=k', i=i', else 0.
        
        # In the schoolbook, U[(i,j), c] = 1 for the column c corresponding to (i,j,k),
        # V[(j,k), c] = 1, W[(k,i), c] = 1. So:
        # T[(i,j), (j',k), (k',i')] = sum_c U[(i,j),c] V[(j',k),c] W[(k',i'),c]
        # = 1 if there exists c such that U[(i,j),c]=V[(j',k),c]=W[(k',i'),c]=1
        # In schoolbook, c corresponds to (i,j,k), so U[(i,j),c]=1 only when the (i,j) part matches,
        # V[(j',k),c]=1 only when (j',k) matches (j,k), W[(k',i'),c]=1 only when (k',i') matches (k,i).
        # So T[(i,j),(j',k),(k',i')] = 1 if j=j' and k=k' and i=i', else 0.
        
        # So the tensor is: T[(i,j),(j',k),(k',i')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
        
        # This is the "identity-like" tensor. Its rank is n*m*p in the schoolbook, but might be lower.
        
        # OK so let's verify with this understanding.
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # Check: T[(i,j),(j',k),(k',i')] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
        # should equal delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
        
        # For efficiency, check a subset if needed
        errors = 0
        max_err = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for i2 in range(n):
                                expected = 1.0 if (i == i2 and j == jp and k == kp) else 0.0
                                val = 0.0
                                for r in range(rank):
                                    val += U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+i2, r]
                                err = abs(val - expected)
                                if err > tol:
                                    errors += 1
                                if err > max_err:
                                    max_err = err
        return errors == 0, max_err

    # Try to find a low-rank decomposition using random initialization and optimization
    # We'll use a simple approach: try various ranks and use random vectors
    
    best_rank = n * m * p  # schoolbook
    best_decomp = None
    
    # Schoolbook as fallback
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    best_decomp = (U.copy(), V.copy(), W.copy())
    
    # Try to find lower rank decompositions
    # The tensor T[(i,j),(j',k),(k',i')] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # This is essentially the "slice" tensor. 
    
    # One approach: use the fact that this tensor can be decomposed using
    # Kronecker products or other structured decompositions.
    
    # Actually, let's think about it differently. The tensor T is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k ⊗ e_i ⊗ e_j ⊗ e_k  (in some sense)
    
    # Or more precisely, T is a 3-tensor where T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k.
    
    # Let's try a simple structured decomposition.
    # For each i, we can try to use a rank-m*p decomposition of the "i-slice".
    
    # Alternative: use random vectors and check if they work.
    
    # Let's try a more systematic approach.
    # The tensor T can be written as:
    # T[(i,j),(j',k),(k',i')] = sum_{i'',j'',k''} delta_{i,i''} delta_{j,j''} delta_{k,k''} delta_{i,i'''} delta_{j,j'''} delta_{k,k'''}
    # where i''=i''', j''=j''', k''=k''
    
    # This is getting complex. Let's just try random decompositions with decreasing rank.
    
    rng = np.random.RandomState(42)
    
    for target_rank in range(27, best_rank + 1):
        if time.time() > deadline - 30:
            break
            
        # Try a few random initializations
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
                
            U = rng.randn(n*m, target_rank)
            V = rng.randn(m*p, target_rank)
            W = rng.randn(p*n, target_rank)
            
            # Scale to make entries roughly O(1)
            U = U / np.sqrt(target_rank)
            V = V / np.sqrt(target_rank)
            W = W / np.sqrt(target_rank)
            
            ok, max_err = verify(U, V, W, tol=1e-4)
            if ok:
                best_rank = target_rank
                best_decomp = (U.copy(), V.copy(), W.copy())
                break
        
        if best_rank == target_rank:
            break
    
    # Convert to list format
    U_list = best_decomp[0].tolist()
    V_list = best_decomp[1].tolist()
    W_list = best_decomp[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
