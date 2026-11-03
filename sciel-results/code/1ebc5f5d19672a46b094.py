"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time
import random
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 by 5x5 matrix multiplication.
    Strategy: Use the schoolbook decomposition as a baseline (rank 75), then attempt
    to reduce rank via random perturbation and iterative compression (rank reduction).
    Since finding a provably minimal rank is hard, we use a heuristic search that
    tries to find a decomposition with rank < 75 by exploiting linear dependencies.
    
    For small matrices, we can try:
    1. Start with schoolbook (rank n*m*p = 75)
    2. Try to find linear combinations that reduce rank
    3. Use Strassen-like ideas for sub-blocks
    
    Given the dimensions (3,5,5), let's think about block structure:
    - The 3x5 matrix A and 5x5 matrix B produce 3x5 matrix C
    - C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can partition rows of A into groups and columns of B into groups.
    
    For now, let's implement a search that tries to find rank < 75 by:
    1. Using the tensor formulation
    2. Trying random rank-r decompositions and checking if they're valid
    
    The tensor T has indices (i,j,k) -> contribution to (i,k) from (i,j) and (j,k)
    T[i,j,k] = e_{i,k} in the output space
    
    We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k} * (product structure)
    
    Actually, the correct tensor for matrix multiplication is:
    T[i,j,k] = e_{i,k} where the output index is (i,k) and input indices are (i,j) and (j,k)
    
    More precisely, the multiplication tensor has entries:
    T_{(i,j),(j,k),(i,k)} = 1 for all valid i,j,k
    
    We seek rank-r decomposition:
    T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    
    And the constraint is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) corresponds to valid mult
    Actually, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    and the output is at position (i,k) with weight... 
    
    Let me re-derive. The matrix multiplication tensor for C = A * B where A is n×m, B is m×p, C is n×p:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor T has components T[i,j,k] such that:
    <T, A ⊗ B> = C in the sense that T[i,j,k] extracts the coefficient.
    
    Specifically, T[i,j,k] = δ_{i,k}... no.
    
    The correct formulation: T is a 3-tensor with indices (i,j,k) where:
    - i ranges over rows of A (n values)
    - j ranges over columns of A / rows of B (m values)  
    - k ranges over columns of B (p values)
    
    And T[i,j,k] = 1, and the output C[i,k] = sum_{i',j',k'} T[i',j',k'] A[i',j'] B[j',k'] δ_{i,i'} δ_{k,k'}
    
    Wait, that's not right either. Let me think more carefully.
    
    The multiplication tensor M for n×m by m×p has the property that:
    M(A, B) = A * B
    
    In tensor form, M ∈ (R^{n×m}*)* ⊗ (R^{m×p}*)* ⊗ R^{n×p}
    
    The components are: M[i,j,k] where (i,j) indexes A, (j,k) indexes B, and the output is at (i,k).
    
    So M[i,j,k] = e_{i,k} in the output space, meaning:
    sum_{i,j,k} M[i,j,k] A[i,j] B[j,k] = C[i,k] = sum_j A[i,j] B[j,k]
    
    This works if M[i,j,k] = 1 for all i,j,k and we sum over j for fixed i,k.
    
    For the rank decomposition, we write:
    M = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    
    where u_c ∈ R^{n×m}, v_c ∈ R^{m×p}, w_c ∈ R^{n×p}
    
    And: sum_c u_c[i,j] v_c[j,k] w_c[i,k] = M[i,j,k] = 1 for all i,j,k
    
    Wait, that's not quite right because the indices don't match up cleanly.
    
    Let me use the standard formulation. The matrix multiplication tensor for multiplying an n×m matrix by an m×p matrix has rank at most n*m*p (schoolbook).
    
    The tensor T ∈ R^{(n*m) × (m*p) × (n*p)} has entries:
    T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k', else 0
    
    No wait. Let's be very precise.
    
    Input 1: A ∈ R^{n×m}, flattened to vector of length n*m with index a = i*m + j
    Input 2: B ∈ R^{m×p}, flattened to vector of length m*p with index b = j*p + k  
    Output: C ∈ R^{n×p}, flattened to vector of length n*p with index c = i*p + k (or k*n+i per the problem)
    
    The multiplication is: C[i,k] = sum_j A[i,j] B[j,k]
    
    So the tensor T has entries:
    T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = i'*p+k' with i=i', j=j', k=k'
    Otherwise 0.
    
    But this is just the identity-like tensor. Actually:
    T[(i,j), (j',k), (i',k')] = δ_{i,i'} δ_{j,j'} δ_{k,k'}
    
    This is a diagonal tensor of rank n*m*p in the most naive sense, but it can have lower rank.
    
    For the schoolbook: for each (i,j,k), we have a rank-1 term:
    u_{ijk} has 1 at position (i*m+j), 0 elsewhere
    v_{ijk} has 1 at position (j*p+k), 0 elsewhere  
    w_{ijk} has 1 at position (i*p+k) or (k*n+i) per problem convention, 0 elsewhere
    
    And sum over all (i,j,k) gives the correct multiplication.
    
    The problem states W[k*n+i, r], so the output index is c = k*n + i.
    
    So T[(i*m+j), (j*p+k), (k*n+i)] = 1, and we want to decompose this tensor.
    
    Rank minimization is the goal. For 3×5×5, the schoolbook rank is 75.
    
    Known results: For 2×2×2, Strassen gives rank 7 < 8.
    For 3×3×3, best known is 23 < 27.
    
    For 3×5×5, I don't know the exact minimum, but it's likely less than 75.
    
    Let's try a block approach:
    - Split the 3 rows into 2+1
    - The 2×5×5 part might benefit from Strassen-like ideas if we further split 5 into 2+3
    - The 1×5×5 part is just 5 rank-1 terms (each row of A times corresponding row of B)
    
    Actually, for 1×m by m×p, the rank is exactly m*p... no wait.
    For 1×m by m×p: C[0,k] = sum_j A[0,j] B[j,k]
    This is a bilinear form with m*p terms, and the tensor has rank m*p (it's a sum of m*p rank-1 terms and they're independent).
    
    Hmm, actually for 1×m by m×p, the tensor T has entries T[j, j', k] = δ_{j,j'} for the single row.
    This is just the identity tensor of size m×m×p... 
    
    Let me just implement a numerical search for low-rank decompositions.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    def time_left():
        return deadline - time.time()
    
    # Build the schoolbook decomposition as baseline
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Verify schoolbook
    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition is valid."""
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        # And for all other index combinations, the sum should be 0
        # Actually, we need to check that the reconstructed multiplication is correct.
        # The tensor entry T[a,b,c] should equal sum_d U[a,d]*V[b,d]*W[c,d]
        # where a=i*m+j, b=j'*p+k, c=k'*n+i'
        # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i (same i,j,k), else 0
        
        # More efficiently: check that for random A,B, the reconstructed C matches A*B
        rng = np.random.RandomState(42)
        for trial in range(10):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_true = A @ B
            # Reconstruct C from U,V,W
            C_rec = np.zeros((n, p))
            for c in range(U.shape[1]):
                A_u = U[:, c].reshape(n, m)
                B_v = V[:, c].reshape(m, p)
                C_w = W[:, c].reshape(p, n)  # W[k*n+i, c], so reshape to (p, n) then transpose
                # C_w[i,k] = W[k*n+i, c], so C_w = W[:,c].reshape(p,n).T
                C_w = W[:, c].reshape(p, n).T
                C_rec += (A_u @ B_v) * C_w
            if not np.allclose(C_rec, C_true, atol=tol, rtol=tol):
                return False
        return True
    
    # Try to reduce rank using iterative methods
    # Strategy: Start with schoolbook, then try to find linear dependencies
    
    # Method: Try rank r < schoolbook_rank by solving the system
    # For each rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all a,b,c: sum_d U[a,d]*V[b,d]*W[c,d] = T[a,b,c]
    
    # This is a hard non-convex problem. Let's use a heuristic approach.
    
    # Heuristic: Try to merge schoolbook terms
    # If two schoolbook terms can be combined, we reduce rank by 1
    
    # For terms (i,j,k) and (i,j,k'), they share the same U column pattern
    # If W[k*n+i, :] and W[k'*n+i, :] are linearly dependent in a useful way...
    
    # Actually, let's try a different approach: use the fact that for fixed i,
    # the terms (i,j,k) for all j,k form a 5x5 block. The rank of this block
    # in the tensor sense might be reducible.
    
    # For fixed i, we have a 5×5 multiplication: A_i (1×5) * B (5×5) = C_i (1×5)
    # The tensor for this has rank 25 (schoolbook) but might be reducible.
    
    # For 1×5 by 5×5, the tensor is T[j, j', k] = δ_{j,j'}
    # This is the identity tensor I_5 ⊗ I_5... 
    # The rank of the identity tensor I_m ⊗ I_p is m*p (it's a diagonal tensor)
    # So 1×5 by 5×5 has rank exactly 25.
    
    # Similarly, each row block has rank 25, so 3 rows give rank 75 total.
    # This suggests the schoolbook might be optimal for this case!
    
    # But wait, there could be cross-row interactions that allow rank reduction.
    
    # Let's try a numerical optimization approach for small rank reductions.
    
    # Try rank 74, 73, ... by perturbing the schoolbook and projecting
    
    if time_left() < 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to find rank reduction by looking for linear dependencies
    # between the W vectors for different i values
    
    # For the schoolbook, W[k*n+i, c] = 1 only when c corresponds to (i,j,k)
    # So W vectors for different i have disjoint supports -> no linear dependencies
    
    # This means simple linear combination won't help. We need a more sophisticated approach.
    
    # Let's try a random search for low-rank decompositions using gradient descent
    # on the tensor reconstruction error
    
    def try_rank_search(max_rank, num_trials=100):
        """Try to find a valid decomposition with rank <= max_rank."""
        best = None
        rng = np.random.RandomState(123)
        
        for trial in range(num_trials):
            if time_left() < 5:
                break
            
            r = max_rank
            # Initialize with random Gaussian
            U = rng.randn(n * m, r) * 0.1
            V = rng.randn(m * p, r) * 0.1
            W = rng.randn(p * n, r) * 0.1
            
            # Simple gradient descent on the loss
            # Loss = sum_{a,b,c} (sum_d U[a,d]V[b,d]W[c,d] - T[a,b,c])^2
            # This is expensive, so use random sampling of (a,b,c) triples
            
            lr = 0.01
            for step in range(200):
                if time_left() < 5:
                    break
                
                # Sample random triples
                samples = 50
                A_idx = rng.randint(0, n * m, samples)
                B_idx = rng.randint(0, m * p, samples)
                C_idx = rng.randint(0, p * n, samples)
                
                # Compute current values
                vals = np.zeros(samples)
                for s in range(samples):
                    a, b, c = A_idx[s], B_idx[s], C_idx[s]
                    vals[s] = np.dot(U[a, :], np.dot(V[b, :], W[c, :]))
                
                # Target values
                targets = np.zeros(samples)
                for s in range(samples):
                    a, b, c = A_idx[s], B_idx[s], C_idx[s]
                    i, j = divmod(a, m)
                    j2, k = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    if i == i2 and j == j2 and k == k2:
                        targets[s] = 1.0
                
                errors = vals - targets
                loss = np.mean(errors ** 2)
                
                # Check if we've found a valid solution
                if loss < 1e-8:
                    # Verify fully
                    if verify(U, V, W):
                        return U, V, W, r
                
                # Gradient step (simplified)
                for s in range(samples):
                    a, b, c = A_idx[s], B_idx[s], C_idx[s]
                    err = errors[s]
                    # dU[a,:] = err * V[b,:] * W[c,:]
                    # dV[b,:] = err * U[a,:] * W[c,:]
                    # dW[c,:] = err * U[a,:] * V[b,:]
                    U[a, :] -= lr * err * (V[b, :] * W[c, :])
                    V[b, :] -= lr * err * (U[a, :] * W[c, :])
                    W[c, :] -= lr * err * (U[a, :] * V[b, :])
        
        return best
    
    # Try a few rank values below schoolbook
    for target_rank in [74, 73, 72, 70, 65, 60]:
        if time_left() < 15:
            break
        result = try_rank_search(target_rank, num_trials=20)
        if result is not None:
            U, V, W, r = result
            if r < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, r
            break
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
