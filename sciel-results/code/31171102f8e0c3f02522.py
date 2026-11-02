import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def get_schoolbook_solution(n, m, p):
    """Return the schoolbook (trivial) rank decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    return U, V, W

def residual_fn(U, V, W, n, m, p):
    """Compute the Frobenius norm of the residual tensor."""
    # Reconstruct the tensor from the decomposition
    # T[i, j, k, i2, j2, k2] = sum_r U[i,j,r] * V[j2,k2,r] * W[k,i2,r]
    # Wait, let me re-read the convention.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Indices:
    # U: (i, j) -> index i*m + j
    # V: (j2, k2) -> index j2*p + k2  (Wait, V is m*p, so indices are j2 in [0,m), k2 in [0,p))
    #    But the middle dimension is m. A is (n,m), B is (m,p).
    #    U corresponds to A indices (i, j).
    #    V corresponds to B indices (j2, k2). Here j2 is the shared index, k2 is output column.
    #    W corresponds to C indices (k2, i2)? No, W is p*n.
    #    Let's check the seed code: W[k * n + i].
    #    So W index is k * n + i.
    #    In the loop: U[i*m+j], V[j*p+k], W[k*n+i].
    #    This implies the tensor is T[i, j, j2, k2, k, i2] ??
    #    Let's look at the product: U[i,j] V[j2,k2] W[k,i]
    #    The summation is over r.
    #    The resulting tensor has indices (i, j, j2, k2, k, i).
    #    For this to be the matrix multiplication tensor, we need:
    #    C[i, k] = sum_j A[i, j] B[j, k]
    #    The tensor T should be 1 if (i, j, j2, k2, k, i) corresponds to a valid term in the sum?
    #    Actually, the tensor for matrix multiplication M_{n,m,p} is defined as:
    #    T[i, j, k, i', j', k'] ... no.
    #    Standard definition: T[i, j, k, i2, j2, k2] = 1 if i=i2, j=j2, k=k2? No.
    #    The tensor is T[i, j, k, i2, j2, k2] = 1 if i=i2 AND j=j2 AND k=k2?
    #    Let's check the seed code's comment: "T[i, j, j2, k, i2, k2] = 1 if i==i2, j==j2, k==k2"
    #    Wait, the indices in the comment are (i, j, j2, k, i2, k2).
    #    And the condition is i==i2, j==j2, k==k2.
    #    This is the identity tensor on the "diagonal" of the 6D space?
    #    No, matrix multiplication tensor is usually defined as:
    #    T[i, j, k, i2, j2, k2] = 1 if i=i2, j=j2, k=k2 is NOT correct for MM.
    #    MM tensor: T[i, j, k, i2, j2, k2] = 1 if i=i2 AND j=j2 AND k=k2?
    #    Let's verify with the schoolbook decomposition.
    #    U[i, j, r] = 1 if r corresponds to (i, j, k).
    #    V[j2, k2, r] = 1 if r corresponds to (i, j, k) AND j2==j AND k2==k.
    #    W[k, i2, r] = 1 if r corresponds to (i, j, k) AND k==k AND i2==i.
    #    Product: U[i,j,r] V[j2,k2,r] W[k,i2,r]
    #    Non-zero only if:
    #    1. r corresponds to some (i_r, j_r, k_r)
    #    2. U[i,j,r] != 0 => i == i_r, j == j_r
    #    3. V[j2,k2,r] != 0 => j2 == j_r, k2 == k_r
    #    4. W[k,i2,r] != 0 => k == k_r, i2 == i_r
    #    So non-zero if i=i_r, j=j_r, j2=j_r, k2=k_r, k=k_r, i2=i_r.
    #    This implies i=i2, j=j2, k=k2.
    #    So the tensor is indeed T[i, j, j2, k2, k, i2] = 1 if i==i2, j==j2, k==k2.
    #    Wait, the indices are ordered (i, j, j2, k2, k, i2) in the comment?
    #    Seed comment: T[i, j, j2, k, i2, k2].
    #    Let's stick to the seed code's structure.
    #    U is (n*m, rank). Index i*m+j.
    #    V is (m*p, rank). Index j*p+k. (Here j is the shared index, k is the output column of B).
    #    W is (p*n, rank). Index k*n+i. (Here k is the output column of C, i is the output row of C).
    #    Wait, if W index is k*n+i, then the first index of W is k (0..p-1) and second is i (0..n-1).
    #    So W[k, i].
    #    In the sum: sum_r U[i, j, r] V[j2, k2, r] W[k, i2, r].
    #    For this to equal the MM tensor, we need the result to be 1 if i=i2, j=j2, k=k2?
    #    No, MM tensor T[i, j, j2, k2, k, i2] should be 1 if i=i2 AND j=j2 AND k=k2?
    #    Actually, the standard MM tensor is T[i, j, k, i2, j2, k2] = 1 if i=i2, j=j2, k=k2 is WRONG.
    #    The standard MM tensor is T[i, j, k, i2, j2, k2] = 1 if i=i2 AND j=j2 AND k=k2?
    #    Let's check: C[i, k] = sum_j A[i, j] B[j, k].
    #    The tensor T is defined such that C[i, k] = sum_{j, j2, k2} T[i, j, k, i, j2, k2] A[i, j] B[j2, k2]? No.
    #    The tensor is T[i, j, k, i2, j2, k2] = 1 if i=i2, j=j2, k=k2?
    #    If T[i, j, k, i2, j2, k2] = 1 if i=i2, j=j2, k=k2, then
    #    sum_{j, j2, k2} T[i, j, k, i, j2, k2] A[i, j] B[j2, k2]
    #    = sum_{j, j2, k2} (1 if i=i, j=j2, k=k2) A[i, j] B[j2, k2]
    #    = sum_{j, k2} (1 if j=j2, k=k2) A[i, j] B[j2, k2]  (j2=j, k2=k)
    #    = sum_{j} A[i, j] B[j, k]
    #    This works!
    #    So the tensor is T[i, j, k, i2, j2, k2] = 1 if i=i2, j=j2, k=k2.
    #    But the seed code uses indices:
    #    U[i, j] -> (i, j)
    #    V[j2, k2] -> (j2, k2)  [Note: V is m*p, so j2 in [0,m), k2 in [0,p))
    #    W[k, i2] -> (k, i2)    [Note: W is p*n, so k in [0,p), i2 in [0,n))
    #    Wait, in the seed code, W index is k*n+i. So W[k, i].
    #    The product is U[i, j, r] V[j2, k2, r] W[k, i2, r].
    #    This sums over r.
    #    The resulting tensor has indices (i, j, j2, k2, k, i2).
    #    For this to be the MM tensor, we need this sum to be 1 if i=i2, j=j2, k=k2?
    #    Let's check the condition for non-zero in schoolbook:
    #    U[i, j, r] != 0 => i=i_r, j=j_r
    #    V[j2, k2, r] != 0 => j2=j_r, k2=k_r
    #    W[k, i2, r] != 0 => k=k_r, i2=i_r
    #    So non-zero if i=i_r, j=j_r, j2=j_r, k2=k_r, k=k_r, i2=i_r.
    #    This implies i=i2, j=j2, k=k2.
    #    So the tensor is indeed T[i, j, j2, k2, k, i2] = 1 if i=i2, j=j2, k=k2.
    #    This matches the MM tensor definition (with indices permuted appropriately).
    
    # To compute residual, we need to compute the full tensor T from U, V, W and compare to the target.
    # Target tensor T_target[i, j, j2, k2, k, i2] = 1 if i==i2, j==j2, k==k2 else 0.
    # T_decomp[i, j, j2, k2, k, i2] = sum_r U[i*m+j, r] V[j2*p+k2, r] W[k*n+i2, r]
    
    # This is a 6D tensor. n=3, m=4, p=7.
    # Size: 3*4*4*7*7*3 = 14112. Small enough to compute explicitly.
    
    T_target = np.zeros((n, m, m, p, p, n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, j, k, k, i] = 1
                
    # Compute T_decomp
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We can use einsum.
    # U[i, j, r] -> U[i*m+j, r]
    # V[j2, k2, r] -> V[j2*p+k2, r]
    # W[k, i2, r] -> W[k*n+i2, r]
    
    # Reshape U, V, W
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    
    # T_decomp[i, j, j2, k2, k, i2] = sum_r U_reshaped[i, j, r] * V_reshaped[j2, k2, r] * W_reshaped[k, i2, r]
    # This is a bit tricky with einsum directly because of the reshaping.
    # Let's use matmul or explicit loops if rank is small, but rank could be up to 84.
    # n=3, m=4, p=7.
    # U_reshaped: (3, 4, R)
    # V_reshaped: (4, 7, R)
    # W_reshaped: (7, 3, R)
    
    # T_decomp[i, j, j2, k2, k, i2]
    # = sum_r U[i, j, r] V[j2, k2, r] W[k, i2, r]
    
    # We can compute this by:
    # 1. Compute A[i, j, j2, k2, r] = U[i, j, r] * V[j2, k2, r]
    #    A = np.einsum('ijr,jkr->ijjk r', U_reshaped, V_reshaped) -> Shape (n, m, m, p, R)
    # 2. Compute T = sum_r A[i, j, j2, k2, r] * W[k, i2, r]
    #    T = np.einsum('ijjkr,ki2r->ijjkk i2', A, W_reshaped) -> Shape (n, m, m, p, p, n)
    
    # Let's verify indices:
    # U_reshaped: i, j, r
    # V_reshaped: j2, k2, r  (Note: V is m*p, so indices are j2, k2)
    # W_reshaped: k, i2, r   (Note: W is p*n, so indices are k, i2)
    
    # Step 1: A[i, j, j2, k2, r] = U[i, j, r] * V[j2, k2, r]
    # einsum('ijr,jkr->ijjkr', U_reshaped, V_reshaped)
    # Wait, V_reshaped indices are (j2, k2, r).
    # So 'ijr,jkr->ijjkr' is correct.
    
    # Step 2: T[i, j, j2, k2, k, i2] = sum_r A[i, j, j2, k2, r] * W[k, i2, r]
    # einsum('ijjkr,ki2r->ijjkki2', A, W_reshaped)
    
    try:
        A = np.einsum('ijr,jkr->ijjkr', U_reshaped, V_reshaped, optimize=True)
        T_decomp = np.einsum('ijjkr,ki2r->ijjkki2', A, W_reshaped, optimize=True)
    except Exception:
        return np.inf
    
    residual = np.linalg.norm(T_decomp - T_target)
    return residual

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook solution
    U_school, V_school, W_school = get_schoolbook_solution(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find a lower rank decomposition
    # We can try random projections or SVD-based approaches
    
    # Strategy: Randomized rank-revealing decomposition
    # 1. Generate random matrices U_rand, V_rand, W_rand of size (n*m, R), (m*p, R), (p*n, R)
    # 2. Compute the tensor T_rand
    # 3. Check if it approximates the target tensor
    # 4. If not, try to refine
    
    # However, the task requires EXACT decomposition with integer/half-integer entries.
    # Random floating point entries won't work directly.
    # We need to find a rank R such that there exist integer/half-integer U, V, W
    # that exactly decompose the tensor.
    
    # The schoolbook solution has rank n*m*p = 84.
    # Is there a lower rank?
    # For 3x4 by 4x7, the lower bound is max(n*m, m*p, p*n) = max(12, 28, 21) = 28.
    # The upper bound is n*m*p = 84.
    
    # Let's try to find a decomposition with rank < 84.
    # We can use the fact that the tensor is defined over the reals (or rationals).
    # If a decomposition exists with real entries, it can be scaled to have rational entries.
    # But we need integer/half-integer entries.
    
    # Approach:
    # 1. Try ranks from 28 to 83.
    # 2. For each rank R, try to find U, V, W such that the decomposition is exact.
    # 3. Use random initialization and gradient descent to minimize the residual.
    # 4. If the residual is close to 0, snap to nearest integer/half-integer and check exactness.
    
    # This is a hard problem. Let's see if we can find any improvement.
    
    # For now, let's just return the schoolbook solution if we don't find a better one.
    # But we can try a few random restarts.
    
    np.random.seed(42)
    
    for rank in range(28, best_rank):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Normalize to avoid numerical issues
            U = U / np.linalg.norm(U)
            V = V / np.linalg.norm(V)
            W = W / np.linalg.norm(W)
            
            # Check residual
            res = residual_fn(U, V, W, n, m, p)
            
            if res < 1e-6:
                # Found a solution! Snap to integers/half-integers
                # This is tricky. For now, just accept if residual is small.
                # But we need exact integer/half-integer entries.
                # Let's try to snap to nearest 0.5
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                res_snapped = residual_fn(U_snapped, V_snapped, W_snapped, n, m, p)
                
                if res_snapped < 1e-6:
                    best_rank = rank
                    best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
                    break
        if time.time() > deadline:
            break
            
    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
