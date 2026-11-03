"""Rewritten program for matrix multiplication tensor decomposition (3x5 by 5x7)."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Schoolbook decomposition (rank = n*m*p = 105)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # The tensor T is defined by:
    # T[(i,j), (j',k), (k',i')] = 1 if j==j' and k==k' and i==i' else 0
    # We need: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = T[(i,j),(j',k),(k',i')]
    
    # Key insight: For each fixed j, the sub-tensor over (i, k, i') with j'=j is:
    # T[(i,j), (j,k), (k',i')] = 1 if k==k' and i==i'
    # This is essentially the identity tensor of size n*p*p... wait no.
    
    # Let me think differently. The multiplication tensor for (n x m) * (m x p) -> (n x p)
    # has entries: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i,j), (j',k), (i',k') where:
    # T[(i,j),(j',k),(i',k')] = 1 if j==j' and i==i' and k==k'
    
    # So for each j, we have a "slice" that looks like the identity map from (n*p) to (n*p):
    # When j==j', T[(i,j),(j,k),(i',k')] = delta_{i,i'} * delta_{k,k'}
    
    # This means the tensor is a direct sum over j of (n x p) identity tensors!
    # Each slice for fixed j has rank n*p (since it's the identity tensor of dimension n*p)
    # So the total rank is at most m * n * p = 5 * 3 * 7 = 105 (schoolbook)
    
    # But can we do better? The rank of the identity tensor of dimension d is d (over any field)
    # because the identity matrix has rank d, and the tensor rank of the identity is the matrix rank.
    # Wait, the tensor rank of the d-dimensional identity tensor (which is just the identity matrix viewed as a 3-tensor with one index being trivial) is exactly d.
    
    # Actually, let me reconsider. The tensor for multiplying (n x m) by (m x p) is:
    # T_{(i,j),(j',k),(i',k')} = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    # This is a block diagonal tensor where each block (for fixed j) is the (n*p)-dimensional identity tensor.
    # The rank of such a block diagonal tensor is the sum of the ranks of the blocks IF the blocks are on disjoint supports.
    # Here, the blocks ARE on disjoint supports (different j values), so rank = sum over j of rank(identity_{n*p}) = m * n * p.
    
    # Hmm, but that's just the schoolbook. Can we exploit structure across different j values?
    
    # Actually, I think the key observation is that the tensor rank of matrix multiplication M_{n,m,p} 
    # is at most n*m*p, but it could be lower. For small dimensions, we can try to find explicit decompositions.
    
    # Let's try a different approach: use the fact that we can share factors across different j values.
    
    # One idea: if we can find matrices U_r, V_r, W_r such that the sum works out.
    
    # Let's try a systematic search for low-rank decompositions using optimization.
    
    # The constraints are:
    # For all i, i', k, k', j:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'} * delta_{k,k'}
    # For all i, i', k, k', j, j' with j != j':
    #   sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = 0
    
    # This is a system of polynomial equations. Let's try to solve it numerically for small ranks.
    
    # Lower bound on rank: The tensor has "border rank" considerations, but a simple lower bound is:
    # rank >= max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    
    # Let's try to find decompositions of rank R for R from 35 up to 104.
    
    # For each R, we'll try random initialization and gradient descent on the non-convex objective.
    
    # Objective: sum over all (i,j,j',k,k',i') of (sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i,r] - T[i,j,j',k,k',i'])^2
    
    # This is expensive to compute naively, but for small dimensions it's manageable.
    
    # Precompute the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def compute_residual(U, V, W):
        """Compute the residual tensor R = U @ V @ W - T"""
        # R[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use einsum for efficiency
        R = np.einsum('ar,br,cr->abc', U, V, W)
        return R - T
    
    def compute_loss(U, V, W):
        """Compute the squared loss"""
        R = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((R - T)**2)
    
    # Try various ranks
    for R in range(35, min(105, 60)):  # Try up to rank 59
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 25:
                break
            
            # Initialize with small random values
            np.random.seed(42 + R * 100 + attempt)
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            # Use simple gradient descent
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - 20:
                    break
                
                # Compute gradients
                R_tensor = np.einsum('ar,br,cr->abc', U, V, W)
                residual = R_tensor - T
                
                # Gradient of loss w.r.t. U[a,r] = sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
                grad_U = np.einsum('abc,br,cr->ar', residual, V, W)
                grad_V = np.einsum('abc,ar,cr->br', residual, U, W)
                grad_W = np.einsum('abc,ar,br->cr', residual, U, V)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Periodically check loss
                if step % 50 == 0:
                    loss = np.sum(residual**2)
                    if loss < 1e-10:
                        break
            
            # Check final loss
            final_loss = compute_loss(U, V, W)
            if final_loss < 1e-8:
                # Found a valid decomposition!
                # Check if it's actually exact (within floating point)
                R_tensor = np.einsum('ar,br,cr->abc', U, V, W)
                max_err = np.max(np.abs(R_tensor - T))
                if max_err < 1e-6:
                    # Round to nearest rational/half-integer if possible
                    # For now, just accept the floating point solution
                    if R < best_rank:
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = R
                    break  # No need to try more attempts for this rank
        
        # If we found a decomposition at this rank, try lower ranks
        if best_rank == R:
            continue
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(), 
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
