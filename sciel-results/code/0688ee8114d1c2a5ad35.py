import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Trivial decomposition rank
    rank_trivial = n * m * p  # 54
    
    # Build trivial decomposition
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return [[int(x) for x in row] for row in mat]
    
    def verify(U, V, W):
        """Verify that the decomposition is exact"""
        # Check that U @ diag(1) @ V^T @ W^T reconstructs the tensor
        # Actually, the decomposition is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Let's verify by checking a few random entries
        
        # Build the multiplication tensor T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k}
        # Actually, the matrix multiplication tensor is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor is T[i,j,k] = 1 if this is the (i,j,k) term
        
        # The decomposition satisfies:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's verify for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] should be 1 (it's always 1 for the multiplication tensor)
                    # Wait, no. The tensor T[i,j,k] = 1 for all i,j,k in the standard basis
                    # Actually, the multiplication tensor has T[i,j,k] = 1 for all valid indices
                    # Let me reconsider.
                    
                    # The matrix multiplication tensor T is defined as:
                    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] * B[j,k]
                    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
                    
                    # The decomposition is:
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
                    
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a lower rank decomposition using the DFT-based approach
    # The idea: exploit the structure where m=3, p=9=3*3
    
    # Let's try a structured decomposition based on grouping
    # Group the 9 columns of B into 3 groups of 3
    
    # For now, just return the trivial decomposition if no improvement found
    # This ensures we always have a valid answer
    
    # Check time
    if time.time() > deadline - 5:
        return {"U": to_list(U_trivial), "V": to_list(V_trivial), "W": to_list(W_trivial)}
    
    # Try to find a rank 27 decomposition by exploiting the 3x3 structure
    # Idea: if we can factor the p=9 dimension into two 3-dimensions
    # and use some kind of tensor product structure
    
    # Actually, let's try a simple rank reduction:
    # Can we merge some of the trivial terms?
    
    # The trivial decomposition has 54 terms.
    # Let's see if we can find a rank 45, 36, or 27 decomposition.
    
    # For a 2x3x9 multiplication, the border rank might be lower than 54.
    # But we need an exact decomposition.
    
    # Let's try to use the fact that n=2, m=3, p=9
    # The tensor is T[i,j,k] where i in {0,1}, j in {0,1,2}, k in {0,...,8}
    
    # One approach: group k into 3 groups of 3
    # k = 3*g + h where g in {0,1,2}, h in {0,1,2}
    
    # Then T[i,j,3*g+h] = 1 for all i,j,g,h
    
    # Can we write this as a sum of fewer rank-1 terms?
    
    # Let's try: sum_{g,h} U[i,j,g,h] * V[j,3*g+h,g,h] * W[3*g+h,i,g,h]
    # This is still 54 terms.
    
    # To reduce rank, we need to find linear dependencies.
    
    # For now, return the trivial decomposition
    return {"U": to_list(U_trivial), "V": to_list(V_trivial), "W": to_list(W_trivial)}
