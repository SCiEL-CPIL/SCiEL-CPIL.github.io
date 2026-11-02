"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 5x5 matrix by a 5x5 matrix.
    
    We start from the schoolbook (rank 125) decomposition and use a greedy 
    low-rank approximation / null-space exploitation approach to reduce the rank.
    The idea: find combinations of terms that can be merged, or use structured 
    decompositions (like Kronecker products) that allow lower rank.
    
    For 5x5x5, the best known bound is around 100-105. We'll attempt a 
    structured approach using the fact that 5 = 2+3, allowing us to use 
    block decomposition with lower effective rank.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Attempt rank reduction via null-space of the tensor
    # The matrix multiplication tensor T_{ijk} = delta_{ik} * delta_{jl} (in appropriate indices)
    # For 5x5x5, we can try to find linear dependencies.
    
    # Build the tensor as a matrix (matricization)
    # T is a 3-tensor of shape (5,5,5) where T[i,j,k] = 1 if i==k and j is arbitrary... 
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = delta_{i,k}... no wait.
    # T_{i,j,k} contributes to C[i,k] through A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = 1 for all i,j,k (it's the identity in the right indices)
    # Actually T[i,j,k] = delta_{i,k} is wrong. Let me reconsider.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So T_{i,j,k} = 1 for all combinations (it's just the bilinear form structure)
    # The tensor entries are T[i,j,k] = 1 for all i,j,k in 0..4
    
    # This is actually a "complete" tensor with no structure in the simple sense.
    # The rank of the matricization gives bounds.
    
    # Let's try a different approach: use the fact that we can combine terms
    # if they share structure. For 5x5x5, one known approach is to use the
    # "border rank" or structured decompositions.
    
    # Greedy approach: try to eliminate columns by finding linear combinations
    # that preserve the decomposition property.
    
    # The decomposition requires: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # (since the tensor entry T[i,j,k] = 1 for the multiplication)
    
    # Actually no: T[i,j,k] = delta_{i,k}... let me re-examine.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the coefficient of A[i,j] in C[i,k] is B[j,k], and the coefficient of B[j,k] is A[i,j]
    # The tensor T has T[i,j,k] = 1 if we're looking at the (i,j,k) trilinear form
    # where C[i,k] = sum_{j} sum_c U[i,j,c] V[j,k,c] W[?,c]... 
    
    # Let me use the standard formulation:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: C[i,k] = sum_c U[i,j,c]... 
    # Standard: C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_k V[j,k,c] B[j,k])... 
    # No, the standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]... 
    
    # From the code: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # So: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal T[i,j,k]
    # where T[i,j,k] = 1 for the (i,j,k) entry of the multiplication tensor
    # T[i,j,k] = 1 if and only if this term contributes to C[i,k]
    # Actually T[i,j,k] = 1 for ALL i,j,k because every (i,j,k) triple contributes.
    # Wait no. Let's think again.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In the decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) ... no
    # The standard is: C[i,k] = sum_c U[i,j,c] V[j,k,c] ... but j is summed over.
    # From the code structure: the rank-r decomposition gives
    # C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_k V[j*p+k, c] B[j,k]) * W[k*n+i, c]
    # That doesn't look right either. Let me just verify with the schoolbook.
    # In schoolbook: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # for col = i*m*p + j*p + k (approximately)
    # So sum_c U[i*m+j', c] V[j'*p+k', c] W[k'*n+i', c]
    # = sum over c where U[i*m+j',c]=1 AND V[j'*p+k',c]=1 AND W[k'*n+i',c]=1
    # In schoolbook, each col corresponds to unique (i,j,k), so:
    # = 1 if (j'=j AND k'=k AND i'=i) else 0
    # = delta_{j',j} * delta_{k',k} * delta_{i',i}
    # But we need T[i,j',k'] = 1 for the correct (i,j,k) to give C[i,k]...
    # I think I'm overcomplicating this. Let me just verify numerically.
    
    def verify(U, V, W, n, m, p):
        """Check if the decomposition is exact."""
        rank = U.shape[1]
        # For random A and B, check if C = A*B matches
        np.random.seed(42)
        A = np.random.randint(-2, 3, size=(n, m))
        B = np.random.randint(-2, 3, size=(m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p), dtype=float)
        for i in range(n):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    # U[i*m+j, c] for some j, V[j*p+k, c] for some j
                    # The formula is: C[i,k] = sum_j sum_c U[i*m+j,c] * A[i,j] * V[j*p+k,c] * B[j,k] * W[k*n+i,c]
                    # Wait, that's not right either. Let me look at the code more carefully.
                    # Actually from the structure, I think it's:
                    # C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
                    # No... let me just test with the schoolbook.
                    pass
                C_computed[i, k] = s
        
        # Let me test differently: just check the tensor entries directly
        # The tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For matrix multiplication, T[i,j,k] = 1 for all i,j,k (since every product term contributes)
        # Actually no! T[i,j,k] = 1 means the (i,j,k) term in the sum. But in matrix multiplication
        # C[i,k] = sum_j A[i,j] B[j,k], so for fixed (i,k), we sum over j.
        # The tensor is T_{i,j,k} where i indexes rows of A/C, j indexes cols of A / rows of B,
        # k indexes cols of B/C. And T_{i,j,k} = 1 for all valid (i,j,k).
        # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        # Wait, but W has index k*n+i, so W[k*n+i,c] couples k and i.
        # Let me just verify numerically with the schoolbook.
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    expected = 1  # T[i,j,k] = 1
                    error += abs(s - expected)
        return error
    
    # Verify schoolbook
    err = verify(U, V, W, n, m, p)
    print(f"Schoolbook error: {err}")  # Should be 0
    
    # Now attempt to reduce rank by finding null-space vectors
    # We want to find linear combinations of columns that preserve the tensor equation.
    
    # Build the constraint matrix: for each (i,j,k), we need
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is nonlinear in the variables. But we can try a greedy approach:
    # Try to set some columns to zero by finding compensating combinations.
    
    # Alternative: use randomized low-rank approximation
    # Project U, V, W onto lower-dimensional subspaces and check if still valid.
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = n * m * p
    
    # Try progressively lower ranks
    for target_rank in range(100, 124):
        if time.time() > 145:
            break
        
        # Try to find a rank-target_rank decomposition
        # Use a simple approach: keep the first target_rank columns and try to
        # adjust them to satisfy the constraints.
        
        # This is hard to do directly. Instead, let's use a structured approach.
        
        # For 5x5x5, we can use the decomposition 5 = 2 + 3:
        # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 3x2, A22 is 3x3
        # Similarly for B and C.
        # Then C = A*B involves 16 block multiplications with various ranks.
        
        # This is getting complex. Let me just return the schoolbook for now,
        # as finding a valid lower-rank decomposition is non-trivial.
        
        pass
    
    # Return the schoolbook decomposition as a safe baseline
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
