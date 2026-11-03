"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import numpy as np
from itertools import product
import time

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x5x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor T[i,j,k] = delta_{i,k} (for C_ik = sum_j A_ij B_jk)
    # Actually, the tensor is: T[i,j,k] = 1 if i==k else 0... no.
    # The matrix multiplication tensor: T_{i,j,k} where C_ik = sum_j A_ij B_jk
    # So the tensor has entries T[i,k,j] = delta... 
    # In the standard formulation: the tensor T has indices (i, j, k) with T[i,j,k] = 1 if the 
    # term A[i,j]*B[j,k] contributes to C[i,k]. So T[i,j,k] = 1 for all valid i,j,k.
    # Wait, that's the structure tensor. Let me think more carefully.
    
    # The matrix multiplication tensor T ∈ R^(n×m) ⊗ R^(m×p) ⊗ R^(p×n) 
    # Actually for the decomposition we need:
    # C[i,k] = sum_r U[i*m + something, r] * V[...r] * W[k*n+i, r]
    # The convention is: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # And the decomposition satisfies: sum_r U[i,j,r] * V[j,k,r] * W[k,n_i,r] = delta (structure)
    
    # Let me re-read: W[k*n+i, r] - so W index is k*n + i
    # The full tensor T[i,j,k] should satisfy: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 if this is a 
    # "valid" term... 
    
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor decomposition gives: C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k]) W[k,i,r]... 
    # No, that's not right either.
    
    # Standard: we want sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] to equal the structure tensor
    # where the structure tensor T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] product 
    # contributes to C[i,k]).
    
    # Wait, I think the convention is:
    # The multiplication tensor has T[i,j,k] = 1 if the product A[i,j]*B[j,k] is part of C[i,k]
    # In standard matrix mult, every (i,j,k) triple with valid indices contributes, so T[i,j,k]=1.
    
    # Let me verify with the seed: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
    # So for column c corresponding to (i,j,k): U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    # The product for a given output C[i',k']: sum over all columns where i'=i and k'=k
    # Each column contributes U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] to... 
    
    # I think the reconstruction is: C[i,k] = sum_j (sum_c U[i*m+j,c] * A[i,j]) * (sum_c V[j*p+k,c] * B[j,k])... 
    # No. Let me think of it as: 
    # Define intermediate: M1[i,j,k,c] = U[i*m+j,c] * A[i,j]  (shape n x m x p x rank)
    # M2[j,k,c] = V[j*p+k,c] * B[j,k]  (shape m x p x rank)  
    # Then C[i,k] = sum_j sum_c M1[i,j,k,c] * M2[j,k,c] * W[k*n+i,c]
    
    # For this to equal sum_j A[i,j]*B[j,k], we need:
    # sum_j sum_c U[i*m+j,c] * A[i,j] * V[j*p+k,c] * B[j,k] * W[k*n+i,c] = sum_j A[i,j]*B[j,k]
    
    # This must hold for all A,B, so:
    # sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] * A[i,j] * B[j,k] = sum_j A[i,j]*B[j,k]
    
    # For this to be an identity in A and B:
    # The coefficient of A[i',j']*B[j',k'] on the LHS is: sum_c U[i'*m+j',c] * V[j'*p+k',c] * W[k'*n+i',c]
    # This should equal 1 if i'=i, k'=k, j'=j (the diagonal of the "full" tensor) and 0 otherwise.
    # But wait, in standard matrix mult, C[i,k] = sum_j A[i,j]*B[j,k], so the coefficient of 
    # A[i',j']*B[j',k'] in C[i,k] is 1 if i'=i, j'=j, k'=k, else 0.
    
    # So the constraint is: for all i,k: sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] 
    # must equal... no, that's not right.
    
    # Let me re-derive. We want:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (some expression involving U,V,W)
    
    # The standard bilinear form approach: C[i,k] = sum_r U_r[i,*] · A * V_r[*,k] * W_r[i,k]
    # where U_r is a row vector of length m (indexed by j), V_r is a column vector of length m (indexed by j), 
    # and W_r[i,k] is a scalar.
    
    # More precisely: C[i,k] = sum_r W_r[i,k] * (sum_j U_r[i,j] * A[i,j]) * (sum_j V_r[j,k] * B[j,k])
    # But that doesn't factor right.
    
    # Actually, the standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_r sum_j U_r[i,j] * A[i,j] * V_r[j,k] * W_r[i,k]
    # No, that's overcomplicated.
    
    # Let me just use the seed's convention directly and verify it works:
    # For each (i,j,k) triple, column c(i,j,k) has:
    #   U[i*m+j, c] = 1
    #   V[j*p+k, c] = 1  
    #   W[k*n+i, c] = 1
    # All other entries in that column are 0.
    
    # Then for output C[i',k']:
    # sum over all columns: U[i'*m+j, c] * V[j*p+k, c] * W[k*n+i', c] * A[i',j] * B[j,k]... 
    # Hmm, I need to understand how the factors combine.
    
    # I think the evaluation is: 
    # Compute X[i,j,c] = U[i*m+j, c] * A[i,j]
    # Compute Y[j,k,c] = V[j*p+k, c] * B[j,k]
    # Then C[i,k] = sum_j sum_c X[i,j,c] * Y[j,k,c] * W[k*n+i, c]
    
    # With the seed: X[i,j,c(i,j,k)] = 1 * A[i,j] = A[i,j] (only nonzero when c corresponds to (i,j,k))
    # Y[j,k,c(i,j,k)] = 1 * B[j,k] = B[j,k] (only nonzero when c corresponds to (i,j,k))
    # W[k*n+i, c(i,j,k)] = 1
    # So C[i,k] = sum_j sum_k A[i,j] * B[j,k] * 1 = sum_j A[i,j] B[j,k] ✓
    
    # Great, so the constraint is: for all i,k:
    # sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] * A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k]
    # This means: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # (because this must hold for all A and B independently)
    
    # So the constraint is: for all (i,j,k) in [0,n)×[0,m)×[0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 60 equations in 3*rank*? unknowns.
    # We want to minimize rank.
    
    # Let me set up the problem properly.
    
    start_time = time.time()
    DEADLINE = 140  # Leave 10 seconds margin
    
    # Start with schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb), dtype=object)
    V_sb = np.zeros((m*p, rank_sb), dtype=object)
    W_sb = np.zeros((p*n, rank_sb), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank by searching for linear dependencies
    # Strategy: try ranks from 60 downward, and use a randomized/heuristic approach
    
    # First, let's try a structured approach based on known results for small matrix multiplication.
    # For 2x2x2: rank 7 (Strassen)
    # For 2x2x1: rank 3
    # For 2x1x1: rank 1 (trivial)
    # For 2xm x m x 1: rank 2m-1 (known)
    # For 2xm x m x p: various known bounds
    
    # Let me try a direct optimization approach using the constraint equations.
    # The constraint is: for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a bilinear/trilinear system. Let's try to solve it for small ranks.
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition satisfies the constraint."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def verify_decomposition_np(U, V, W, rank):
        """Vectorized verification using numpy."""
        # For each (i,j,k), compute sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # and check it equals 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :rank] * V[j*p+k, :rank] * W[k*n+i, :rank])
                    if abs(s - 1) > 1e-8:
                        return False
        return True
    
    # Try to find a decomposition with rank < 60 using a heuristic approach.
    # Idea: start with schoolbook and try to merge/eliminate terms.
    
    # Alternative: use the fact that for 2-row matrices, there are known lower bounds.
    # The rank of 2xm x m x p matrix multiplication is at least max(2m, 2p, m+p-1, ...).
    # Known: rank >= 2m + 2p - 2 (for m=p=2, gives 6, but Strassen gives 7 for 2x2x2... wait)
    # Actually for 2x2x2, the rank is 7, and 2*2+2*2-2=6, so the bound is not tight.
    
    # Let me try a randomized search with local improvement.
    
    import random
    random.seed(42)
    
    def random_decomposition(rank, scale=1.0):
        """Generate a random decomposition and try to verify."""
        U = np.random.randn(n*m, rank) * scale
        V = np.random.randn(m*p, rank) * scale
        W = np.random.randn(p*n, rank) * scale
        return U, V, W
    
    # Try structured approach: use Strassen-like ideas.
    # For 2x5x6, think of it as 2x5 x 5x6.
    # One approach: compute in blocks.
    
    # Actually, let me try a different strategy: use the constraint equations directly.
    # For each (i,j,k), we need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is equivalent to: the elementwise product of the three vectors (for fixed i,j,k) 
    # sums to 1.
    
    # Let me try to find a solution with rank R by treating it as a constraint satisfaction problem.
    
    # For now, let me try a simple improvement: use half-integer values to get a lower rank.
    
    # Actually, let me try the following structured approach:
    # For 2xm x m x p, one known decomposition uses rank = 2m*p - m - p + 1 (for small cases).
    # For m=5, p=6: 2*5*6 - 5 - 6 + 1 = 60 - 11 + 1 = 50? That seems too optimistic.
    
    # Let me just try to verify the schoolbook and return it if I can't find better quickly.
    
    # Try a few randomized attempts with different rank targets
    for target_rank in [59, 58, 57, 56, 55, 54, 53, 52, 51, 50]:
        if time.time() - start_time > DEADLINE - 20:
            break
            
        # Try to find a solution with this rank using a simple iterative approach
        # Initialize with schoolbook truncated and add randomness
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # First, try a structured initialization
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < target_rank:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
        
        # If we didn't fill all columns, add some randomness
        if col < target_rank:
            U[:, col:] = np.random.randn(n*m, target_rank-col) * 0.5
            V[:, col:] = np.random.randn(m*p, target_rank-col) * 0.5
            W[:, col:] = np.random.randn(p*n, target_rank-col) * 0.5
        
        # Try to verify
        if verify_decomposition_np(U, V, W, target_rank):
            best_rank = target_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < rank_sb:
        U_final = best_U
        V_final = best_V
        W_final = best_W
        rank_final = best_rank
    else:
        U_final = U_sb
        V_final = V_sb
        W_final = W_sb
        rank_final = rank_sb
    
    # Convert to lists for return
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
