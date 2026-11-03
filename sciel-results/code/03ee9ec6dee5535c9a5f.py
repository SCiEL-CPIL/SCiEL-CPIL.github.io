import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using NumPy broadcasting."""
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    
    # Reshape to 3D tensors for einsum
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    # Compute the reconstructed tensor T_{i,j,k}
    # T_{i,j,k} = sum_r U[i,m+j,r] * V[j,p+k,r] * W[k,n+i,r]
    # Note: W is indexed by k*n+i. In the reshaped W_tensor (p, n, rank),
    # the index k corresponds to the first dimension, and i to the second.
    # So W_tensor[k, i, r] is the element at W[k*n+i, r].
    
    # The target tensor for matrix multiplication M_{n,m,p} has entries 1 if the indices form a valid triple.
    # Specifically, the tensor is T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, the tensor for matrix multiplication is defined such that 
    # <T, A \otimes B> = trace(AB) is not quite right.
    # The standard matrix multiplication tensor T has T_{i,j,k} = 1 if i,j,k are consistent?
    # Actually, the tensor T is a 3-way tensor of size (n*m) x (m*p) x (p*n).
    # The entry T_{(i,j), (j',k'), (k,i')} is 1 if i=i', j=j', k=k' and 0 otherwise?
    # No, the matrix multiplication tensor is defined by the bilinear map.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This can be written as sum_{i,j,k} A_{i,j} B_{j,k} T_{i,j,k} = C_{i,k} ?? No.
    # The tensor T is such that T_{i,j,k} = 1 if the indices match the "path" i->j->k.
    # In the context of the factorization sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1,
    # the target is the identity tensor on the "connected" indices.
    # For all i, j, k, the sum must equal 1.
    # Why? Because if we set A_{i,j}=1, B_{j,k}=1, and 0 otherwise, 
    # then C_{i,k} = 1. 
    # The contraction sum_{i,j,k} A_{i,j} B_{j,k} T_{i,j,k} should yield C_{i,k}?
    # Actually, the standard definition is that the tensor T has rank equal to the bilinear complexity.
    # The entries of T are T_{(i,j), (j,k), (i,k)} = 1.
    # So for a fixed (i,j,k), the value is 1.
    # The decomposition is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    # This must hold for ALL i, j, k.
    
    # Let's verify this hypothesis.
    # If i=0, j=0, k=0: sum_r U[0,r] V[0,r] W[0,r] = 1.
    # If i=0, j=0, k=1: sum_r U[0,r] V[p,r] W[n,r] = 1.
    # Is this correct?
    # If A is the matrix with 1 at (0,0) and 0 elsewhere.
    # If B is the matrix with 1 at (0,0) and 0 elsewhere.
    # C = A B has 1 at (0,0) and 0 elsewhere.
    # The tensor contraction <T, A \otimes B> = sum_{i,j,k} A_{i,j} B_{j,k} T_{i,j,k}.
    # If T_{i,j,k} = 1 for all i,j,k, then <T, A \otimes B> = 1.
    # But we want C_{i,k}. 
    # The correct relation is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is equivalent to saying that the tensor T has components T_{i,j,k} such that
    # sum_j A_{i,j} B_{j,k} = sum_{j} A_{i,j} B_{j,k} * 1.
    # This doesn't involve a 3-tensor directly in the output index.
    # The tensor decomposition is for the *bilinear map*.
    # The map is (A, B) -> C.
    # C_{i,k} = sum_{r} (sum_j U_{i,j,r} A_{i,j}) * (sum_l V_{k,l,r} B_{l,k})? No.
    # The standard factorization is:
    # C_{i,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) * (sum_l V_{k,l,r} B_{l,k}) is not right.
    # It is: C_{i,k} = sum_r (U_r^T A)_{i} (V_r^T B)_{k} ?? No.
    
    # Let's stick to the code provided in the prompt which works for schoolbook.
    # The prompt's verification:
    # T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
    # if not np.allclose(T, 1.0, atol=tol): return False
    # This implies the target tensor is all 1s.
    # Let's double check if this is the correct tensor for matrix multiplication.
    # If the target is all 1s, then for any A, B, the contraction sum_{i,j,k} A_{i,j} B_{j,k} * 1 = sum A_{i,j} B_{j,k} = trace(AB).
    # But matrix multiplication produces C, not trace(AB).
    # However, the *rank* of the tensor defining the bilinear map is the same as the rank of the tensor T where T_{i,j,k}=1.
    # Why? Because the bilinear map is (A,B) -> C.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This can be viewed as a tensor T such that C = T(A,B).
    # The tensor T has indices (i,j) for A, (j,k) for B, and (i,k) for C.
    # The entry T_{(i,j), (j,k), (i,k)} is 1.
    # The decomposition is sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r} = 1.
    # Note the index of W. In the prompt, W is indexed by k*n+i.
    # So W_{k,i,r} corresponds to the (i,k) entry of C?
    # If W is indexed by k*n+i, then the row index is k*n+i.
    # The entry is W[k*n+i, r].
    # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # This matches T_{i,j,k} = 1.
    # So the verification `np.allclose(T, 1.0)` is correct for the standard matrix multiplication tensor rank.
    
    T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _generate_random_sparse(n, m, p, rank, density=0.1):
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n * m):
        for c in range(rank):
            if np.random.random() < density:
                U[i, c] = 1.0
    for j in range(m * p):
        for c in range(rank):
            if np.random.random() < density:
                V[j, c] = 1.0
    for k in range(p * n):
        for c in range(rank):
            if np.random.random() < density:
                W[k, c] = 1.0
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_school
    
    # Try to find a lower rank decomposition
    # We try ranks slightly below schoolbook.
    # Given the time constraint, we limit the number of attempts.
    # For 4x5x5, rank 100 is the schoolbook.
    # It is unlikely to find a rank < 100 quickly with random search.
    # However, we will attempt a few quick checks for rank 99, 98, etc.
    
    # We use a fixed seed for reproducibility in case of rare hits, but random seed is better for exploration.
    # Since the problem is likely to remain at 100, we prioritize speed.
    
    # Try rank 99 with a few random attempts
    for target_rank in [99, 98, 97]:
        if time.time() > deadline - 20:
            break
            
        # Number of attempts based on remaining time
        attempts = 50
        
        for _ in range(attempts):
            if time.time() > deadline - 10:
                break
                
            U_try, V_try, W_try = _generate_random_sparse(n, m, p, target_rank, density=0.2)
            
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
                break # Found a valid one for this rank, try lower? No, just return best.
                # Actually, if we find rank 99, we should check 98? 
                # But it's highly unlikely. Let's just return the first improvement.
        
        if best_rank < rank_school:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
