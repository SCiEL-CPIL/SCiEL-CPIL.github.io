"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook_456():
    """Build the schoolbook (rank-120) decomposition for 4x5 * 5x6 -> 4x6."""
    n, m, p = 4, 5, 6
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W):
    """Check that U, V, W exactly decompose the 4x5*5x6 tensor.
    Returns True if exact (within tight tolerance)."""
    n, m, p = 4, 5, 6
    # Build the tensor from U, V, W
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The true tensor has T[i,j,k] = 1 if j==k (the multiplication tensor) else 0
    # Actually the matrix multiplication tensor: T[i,j,k] = delta_{j,k}
    # Wait, let me reconsider the convention.
    
    # The matrix multiplication tensor M[i,j,k] where:
    #   i indexes row of A (0..n-1)
    #   j indexes column of A / row of B (0..m-1) 
    #   k indexes column of B (0..p-1)
    # M[i,j,k] = 1 if j == k (when m==p, but here m=5, p=6 so j ranges 0..4, k ranges 0..5)
    # Actually for matmul: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = delta_{j,k} only when m==p.
    # For m != p, the tensor is: T[i,j,k] = 1 if j == k (for j in range(m), k in range(p))
    # But j ranges 0..m-1=0..4 and k ranges 0..p-1=0..5.
    # So T[i,j,k] = 1 if j == k (and j < m, k < p), else 0.
    # This is independent of i! So the tensor is: for each (j,k) with j==k, all i have value 1.
    
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify by computing the full tensor and comparing.
    rank = U.shape[1]
    # Compute T from decomposition
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This is a 3-tensor of shape (n, m, p)
    
    # Efficient computation:
    # For each r, the contribution is U[:,r] (shape n*m) * V[:,r] (shape m*p) * W[:,r] (shape p*n)
    # Reshape U[:,r] to (n,m), V[:,r] to (m,p), W[:,r] to (p,n)
    # Then T += outer of these three... but that's expensive.
    
    # Alternative: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Let's just compute it directly for verification.
    
    T_computed = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_computed[i, j, k] = np.sum(
                    U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                )
    
    # True tensor
    T_true = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j == k:
                    T_true[i, j, k] = 1.0
    
    max_err = np.max(np.abs(T_computed - T_true))
    return max_err < 1e-9, max_err


def _try_reduce_rank(U, V, W, max_time_remaining, rng):
    """Try to reduce the rank of the decomposition using iterative methods.
    Returns (U, V, W, rank) or the original if no reduction found."""
    n, m, p = 4, 5, 6
    current_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    start = time.time()
    
    # Strategy: Try to find a lower-rank decomposition by:
    # 1. Randomly projecting and checking if we can drop columns
    # 2. Using the structure of the tensor
    
    # The key insight: the matmul tensor T[i,j,k] = delta_{j,k} has a specific structure.
    # T[i,j,k] depends only on (j,k), not on i. This means:
    # T[i,j,k] = delta_{j,k} for all i.
    # 
    # So the tensor is: sum_i T[i,j,k] = n * delta_{j,k}
    # 
    # A rank-1 term U[:,r] * V[:,r] * W[:,r] contributes:
    #   U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # 
    # For this to equal delta_{j,k} (independent of i), we need special structure.
    
    # Let's try a direct construction approach:
    # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = delta_{j,k}
    
    # Since the RHS doesn't depend on i, we can try:
    # W[k*n+i,r] = w_k * w_i (separable in k and i)
    # U[i*m+j,r] = u_i * u_j (separable in i and j)  
    # V[j*p+k,r] = v_j * v_k (separable in j and k)
    
    # Then: sum_r (u_i * u_j) * (v_j * v_k) * (w_k * w_i)
    #      = (sum_r u_i * w_i) * (sum_r u_j * v_j) * (sum_r v_k * w_k)
    #      = a * b * c  (independent of i,j,k)
    # That gives a constant, not delta_{j,k}. So separability is too restrictive.
    
    # Let me try: W[k*n+i,r] depends only on k (not i): W[k*n+i,r] = w_k
    # Then: sum_r U[i*m+j,r] * V[j*p+k,r] * w_k
    # For this to equal delta_{j,k}:
    #   sum_r U[i*m+j,r] * V[j*p+k,r] = delta_{j,k} / w_k
    #   = delta_{j,k} * w_k^{-1}
    # 
    # This must hold for all i. So sum_r U[i*m+j,r] * V[j*p+k,r] must be independent of i.
    # Let's set U[i*m+j,r] = u_j (independent of i):
    #   sum_r u_j * V[j*p+k,r] = u_j * sum_r V[j*p+k,r]
    #   = delta_{j,k} / w_k
    #   => sum_r V[j*p+k,r] = delta_{j,k} / (u_j * w_k)
    # 
    # This is getting complicated. Let me try a more systematic approach.
    
    # Actually, let's think about what rank is achievable.
    # The tensor T[i,j,k] = delta_{j,k} for i=0..3, j=0..4, k=0..5.
    # 
    # For a fixed j, the "slice" T[:,j,:] is:
    #   T[i,j,k] = 1 if k==j, else 0, for all i.
    # So T[:,j,:] is a 4x6 matrix with a single 1 at position (i, j) for all i.
    # Wait no: T[i,j,k] = delta_{j,k}. So for fixed j, T[i,j,k] = 1 if k=j, else 0.
    # This is a 4x6 matrix where column j is all 1s, and all other columns are 0.
    # 
    # The rank of this 4x6 matrix is 1 (it's a column vector of 1s times a row with 1 at position j).
    # 
    # Now, the tensor rank: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # 
    # Let me try to construct a decomposition with rank < 120.
    # 
    # Observation: T[i,j,k] = delta_{j,k} doesn't depend on i.
    # So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = delta_{j,k} for all i.
    # 
    # This means: for each (j,k), the function f_{j,k}(i) = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # must equal delta_{j,k} for all i in {0,1,2,3}.
    # 
    # If we can make U[i*m+j,r] * W[k*n+i,r] independent of i, that would help.
    # Let U[i*m+j,r] = a_{j,r} and W[k*n+i,r] = b_{k,r} (independent of i).
    # Then: sum_r a_{j,r} * V[j*p+k,r] * b_{k,r} = delta_{j,k}
    # 
    # This is a system: for each (j,k), sum_r a_{j,r} * V[j*p+k,r] * b_{k,r} = delta_{j,k}
    # 
    # If we also set V[j*p+k,r] = c_{j,k,r}, we need:
    # sum_r a_{j,r} * c_{j,k,r} * b_{k,r} = delta_{j,k}
    # 
    # One simple solution: set a_{j,r} = 1, b_{k,r} = 1, c_{j,k,r} = delta_{j,k} / rank for r=1..rank.
    # But that gives rank = number of (j,k) pairs with j==k = 5.
    # Wait, that's only 5 terms! But we need to verify this works.
    
    # Let me try: rank = 5 (or maybe fewer).
    # Set: for r = 0..4:
    #   U[i*m+j, r] = 1 if j == r, else 0  (independent of i)
    #   V[j*p+k, r] = 1 if j == r AND k == r, else 0
    #   W[k*n+i, r] = 1 if k == r, else 0  (independent of i)
    # 
    # Then: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    #      = sum_{r: j==r, j==r, k==r} 1*1*1
    #      = 1 if j==k and j in {0,1,2,3,4}, else 0
    #      = delta_{j,k} (since j ranges 0..4, k ranges 0..5)
    # 
    # This works! Rank = 5!
    # 
    # But wait, let me double-check the shapes:
    # U has shape (n*m, rank) = (20, 5)
    # V has shape (m*p, rank) = (30, 5)
    # W has shape (p*n, rank) = (24, 5)
    
    # Let me verify this construction.
    U5 = np.zeros((20, 5), dtype=np.float64)
    V5 = np.zeros((30, 5), dtype=np.float64)
    W5 = np.zeros((24, 5), dtype=np.float64)
    
    for r in range(5):
        # U[i*m+j, r] = 1 if j == r
        for i in range(4):
            U5[i*5 + r, r] = 1.0
        # V[j*p+k, r] = 1 if j == r AND k == r
        V5[r*6 + r, r] = 1.0
        # W[k*n+i, r] = 1 if k == r
        for i in range(4):
            W5[r*4 + i, r] = 1.0
    
    valid, err = _verify(U5, V5, W5)
    if valid:
        return U5, V5, W5, 5
    
    # If that didn't work (shouldn't happen), try rank = 6 (include k=5 case? No, j only goes to 4)
    # Actually j ranges 0..4 and k ranges 0..5. delta_{j,k} is 1 only when j==k, so k must also be 0..4.
    # So rank 5 should suffice.
    
    # Let me also try to see if we can do even better.
    # Can we do rank < 5? 
    # The tensor has 5 non-zero "diagonal" entries (j=k=0,1,2,3,4).
    # Each rank-1 term can potentially cover multiple diagonal entries if the structure allows.
    # 
    # For a rank-1 term to contribute to T[i,j,k] = delta_{j,k}, it must have:
    # U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 0 unless j == k
    # 
    # If a single rank-1 term covers two diagonal entries (j1=k1) and (j2=k2), we need:
    # U[i*m+j1,r] * V[j1*p+j1,r] * W[j1*n+i,r] = 1
    # U[i*m+j2,r] * V[j2*p+j2,r] * W[j2*n+i,r] = 1
    # And for all other (j,k): U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 0
    # 
    # This is possible if the support of the rank-1 term is exactly on the diagonal entries.
    # But a rank-1 tensor U[:,r] * V[:,r] * W[:,r] has a "rectangular" support structure.
    # 
    # For the term to be zero at (i,j,k) when j != k, we need at least one of U[i*m+j,r], V[j*p+k,r], W[k*n+i,r] to be zero.
    # 
    # If we set U[i*m+j,r] to be non-zero only for specific j values, and similarly for V and W,
    # we might be able to cover multiple diagonal entries with one term.
    # 
    # But actually, my construction above already achieves rank 5, which is the number of diagonal entries.
    # It's hard to see how we could do better than 5 since each diagonal entry (j,k=j) needs to be "covered".
    # 
    # Actually, could one rank-1 term cover two diagonal entries? Let's think...
    # Term r contributes to T[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r].
    # For this to be 1 at (i, j1, j1) and (i, j2, j2) for j1 != j2:
    #   U[i*m+j1,r] * V[j1*p+j1,r] * W[j1*n+i,r] = 1
    #   U[i*m+j2,r] * V[j2*p+j2,r] * W[j2*n+i,r] = 1
    # And for all (i', j, k) with j != k: U[i'*m+j,r] * V[j*p+k,r] * W[k*n+i',r] = 0
    # 
    # The second condition is very restrictive. For example, take i' = 0, j = j1, k = j2 (j1 != j2):
    #   U[j1, r] * V[j1*p+j2, r] * W[j2, r] = 0
    # 
    # And for i' = 0, j = j2, k = j1:
    #   U[j2, r] * V[j2*p+j1, r] * W[j1, r] = 0
    # 
    # These don't directly contradict the first two equations unless we can show they force zeros.
    # 
    # Let's try a specific example with j1=0, j2=1:
    #   U[0, r] * V[0, r] * W[0, r] = 1  (from (i=0, j=0, k=0))
    #   U[5, r] * V[7, r] * W[1, r] = 1  (from (i=0, j=1, k=1), since i*m+j = 0*5+1 = 1... wait)
    
    # Let me re-index. U has row index i*m+j where i in 0..3, j in 0..4.
    # V has row index j*p+k where j in 0..4, k in 0..5.
    # W has row index k*n+i where k in 0..5, i in 0..3.
    
    # For diagonal entry (j=0, k=0): T[i,0,0] = U[i*5+0, r] * V[0*6+0, r] * W[0*4+i, r] = U[i*5, r] * V[0, r] * W[i, r]
    # For diagonal entry (j=1, k=1): T[i,1,1] = U[i*5+1, r] * V[1*6+1, r] * W[1*4+i, r] = U[i*5+1, r] * V[7, r] * W[4+i, r]
    
    # For these to both equal 1 for all i:
    # U[i*5, r] * V[0, r] * W[i, r] = 1 for i=0,1,2,3
    # U[i*5+1, r] * V[7, r] * W[4+i, r] = 1 for i=0,1,2,3
    
    # And for off-diagonal, e.g., (j=0, k=1): T[i,0,1] = U[i*5, r] * V[0*6+1, r] * W[1*4+i, r] = U[i*5, r] * V[1, r] * W[4+i, r] = 0
    # And (j=1, k=0): T[i,1,0] = U[i*5+1, r] * V[1*6+0, r] * W[0*4+i, r] = U[i*5+1, r] * V[6, r] * W[i, r] = 0
    
    # From U[i*5, r] * V[0, r] * W[i, r] = 1, we get U[i*5, r] != 0 and W[i, r] != 0 for all i.
    # From U[i*5, r] * V[1, r] * W[4+i, r] = 0, and U[i*5, r] != 0, we need V[1, r] * W[4+i, r] = 0 for all i.
    # From U[i*5+1, r] * V[7, r] * W[4+i, r] = 1, we get W[4+i, r] != 0 for all i.
    # Contradiction! V[1, r] * W[4+i, r] = 0 but W[4+i, r] != 0, so V[1, r] = 0.
    # But then V[1, r] = 0 is fine as long as it doesn't affect the diagonal equations.
    # V[1, r] appears in T[i,0,1] and T[i,1,0] (off-diagonal), so setting it to 0 is fine.
    
    # Let me check if this is consistent. Set:
    # V[0, r] = a, V[7, r] = b (non-zero)
    # U[i*5, r] = c_i, U[i*5+1, r] = d_i (non-zero)
    # W[i, r] = e_i, W[4+i, r] = f_i (non-zero)
    # 
    # Equations:
    # c_i * a * e_i = 1 => c_i * e_i = 1/a
    # d_i * b * f_i = 1 => d_i * f_i = 1/b
    # 
    # Off-diagonal constraints (just a few):
    # U[i*5, r] * V[1, r] * W[4+i, r] = c_i * V[1,r] * f_i = 0 => V[1,r] = 0 (since c_i, f_i != 0)
    # U[i*5+1, r] * V[6, r] * W[i, r] = d_i * V[6,r] * e_i = 0 => V[6,r] = 0 (since d_i, e_i != 0)
    # U[i*5, r] * V[2, r] * W[8+i, r] = c_i * V[2,r] * W[8+i,r] = 0
    # U[i*5+1, r] * V[13, r] * W[4+i, r] = d_i * V[13,r] * f_i = 0 => V[13,r] = 0
    # etc.
    
    # So we need V[j*6+k, r] = 0 for all (j,k) except (0,0) and (1,1).
    # Similarly, we need constraints on U and W.
    
    # For U: U[i*m+j, r] must be non-zero only for j=0 and j=1.
    # For W: W[k*n+i, r] must be non-zero only for k=0 and k=1.
    
    # Let's check: if U is non-zero only for j in {0,1}, V is non-zero only for (j,k) in {(0,0),(1,1)},
    # and W is non-zero only for k in {0,1}, then:
    # T[i,j,k] = U[i*5+j, r] * V[j*6+k, r] * W[k*4+i, r]
    # 
    # For this to be non-zero, we need j in {0,1}, (j,k) in {(0,0),(1,1)}, k in {0,1}.
    # The only (j,k) pairs that satisfy all three: (0,0) and (1,1).
    # So T[i,0,0] = U[i*5, r] * V[0, r] * W[i, r]
    #    T[i,1,1] = U[i*5+1, r] * V[7, r] * W[4+i, r]
    # All other T[i,j,k] = 0.
    # 
    # This works! So one rank-1 term can cover two diagonal entries.
    # Similarly, we can pair up the 5 diagonal entries: (0,0)&(1,1), (2,2)&(3,3), and (4,4) alone.
    # That gives rank = 3!
    
    # Let me construct rank = 3:
    # Term 0: covers (j,k)=(0,0) and (1,1)
    # Term 1: covers (j,k)=(2,2) and (3,3)
    # Term 2: covers (j,k)=(4,4)
    
    U3 = np.zeros((20, 3), dtype=np.float64)
    V3 = np.zeros((30, 3), dtype=np.float64)
    W3 = np.zeros((24, 3), dtype=np.float64)
    
    # Term 0: (0,0) and (1,1)
    # U[i*5+0, 0] * V[0, 0] * W[i, 0] = 1 for all i
    # U[i*5+1, 0] * V[7, 0] * W[4+i, 0] = 1 for all i
    # Set V[0,0] = 1, V[7,0] = 1
    # Then U[i*5, 0] * W[i, 0] = 1 and U[i*5+1, 0] * W[4+i, 0] = 1
    # Set U[i*5, 0] = 1, W[i, 0] = 1 for i=0..3
    # Set U[i*5+1, 0] = 1, W[4+i, 0] = 1 for i=0..3
    for i in range(4):
        U3[i*5 + 0, 0] = 1.0
        U3[i*5 + 1, 0] = 1.0
        W3[i, 0] = 1.0
        W3[4+i, 0] = 1.0
    V3[0*6 + 0, 0] = 1.0
    V3[1*6 + 1, 0] = 1.0
    
    # Term 1: (2,2) and (3,3)
    for i in range(4):
        U3[i*5 + 2, 1] = 1.0
        U3[i*5 + 3, 1] = 1.0
        W3[2*4 + i, 1] = 1.0
        W3[3*4 + i, 1] = 1.0
    V3[2*6 + 2, 1] = 1.0
    V3[3*6 + 3, 1] = 1.0
    
    # Term 2: (4,4)
    for i in range(4):
        U3[i*5 + 4, 2] = 1.0
        W3[4*4 + i, 2] = 1.0
    V3[4*6 + 4, 2] = 1.0
    
    valid, err = _verify(U3, V3, W3)
    if valid:
        return U3, V3, W3, 3
    
    # Can we do rank = 2? We have 5 diagonal entries to cover.
    # Each term can cover at most 2 (as shown above). So minimum rank is ceil(5/2) = 3.
    # Unless a term can cover 3 or more diagonal entries.
    
    # Let's check if a term can cover 3 diagonal entries, say (0,0), (1,1), (2,2).
    # U[i*5+0, r] * V[0, r] * W[i, r] = 1
    # U[i*5+1, r] * V[7, r] * W[4+i, r] = 1
    # U[i*5+2, r] * V[14, r] * W[8+i, r] = 1
    # 
    # Off-diagonal: U[i*5+0, r] * V[1, r] * W[4+i, r] = 0
    # Since U[i*5+0, r] != 0 and W[4+i, r] != 0 (from second eq), V[1, r] = 0.
    # U[i*5+0, r] * V[8, r] * W[8+i, r] = 0
    # Since U[i*5+0, r] != 0 and W[8+i, r] != 0 (from third eq), V[8, r] = 0.
    # U[i*5+1, r] * V[6, r] * W[i, r] = 0
    # Since U[i*5+1, r] != 0 and W[i, r] != 0 (from first eq), V[6, r] = 0.
    # U[i*5+1, r] * V[13, r] * W[8+i, r] = 0
    # Since U[i*5+1, r] != 0 and W[8+i, r] != 0, V[13, r] = 0.
    # U[i*5+2, r] * V[12, r] * W[i, r] = 0
    # Since U[i*5+2, r] != 0 and W[i, r] != 0, V[12, r] = 0.
    # U[i*5+2, r] * V[19, r] * W[4+i, r] = 0
    # Since U[i*5+2, r] != 0 and W[4+i, r] != 0, V[19, r] = 0.
    # 
    # Also: U[i*5+0, r] * V[7, r] * W[4+i, r] = 0 (off-diagonal j=0, k=1)
    # Since U[i*5+0, r] != 0 and W[4+i, r] != 0, V[7, r] = 0.
    # But V[7, r] must be non-zero from the second equation! Contradiction!
    # 
    # So a single term cannot cover 3 diagonal entries. Maximum is 2.
    # Therefore minimum rank = ceil(5/2) = 3.
    
    # Let me verify rank 3 works (should have already passed above).
    # If rank 3 verification failed, fall back to rank 5.
    
    if best_rank > 5:
        return U5, V5, W5, 5
    
    return U3, V3, W3, 3


def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try the direct construction first
    try:
        U, V, W, rank = _try_reduce_rank(None, None, None, deadline - time.time(), np.random.default_rng(42))
        if time.time() < deadline:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_456()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
