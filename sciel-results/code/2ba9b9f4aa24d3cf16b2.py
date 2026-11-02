import numpy as np

def construct():
    n, m, p = 2, 3, 7
    
    # Try to find a decomposition with rank < 42
    # Let's try rank 21 first
    rank = 21
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We need: for each i in [0,n), k in [0,p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 for the multiplication tensor
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Trivial: one term per (i,j,k) triple, rank = n*m*p = 42
    # Let's try to be smarter
    
    # Strategy: For each i and k, we compute sum_j A[i,j]*B[j,k]
    # That's 14 inner products of length 3
    
    # Let's try: for each i (2) and for each "combination" of j values
    # Actually, let me just use the trivial decomposition but verify it's correct
    # then try to improve
    
    # For now, use trivial with rank 42 to ensure correctness,
    # but let me try rank 28 or 21
    
    # Let me try a structured approach with rank = n*p*m but with sharing
    # Actually, let me just try rank 21 and construct it
    
    col = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                if col < rank:
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    # Verify the decomposition
    # T[i,j,k] should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # For trivial: T[i,j,k] = 1 if there's a term with that (i,j,k), else 0
    # But we need T[i,j,k] = 1 for ALL (i,j,k) to correctly compute the product
    
    # Wait, I think I'm confusing myself. Let me re-read the problem.
    
    # The matrix multiplication tensor: we want to compute C = A*B where
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k (it's the identity in the bilinear sense)
    # No wait, that's not right either.
    
    # Actually, the tensor decomposition for matrix multiplication:
    # We have a trilinear form or tensor T where T[i,j,k] represents the coefficient
    # such that C[i,k] = sum_{j,r} U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * A[i,j] * B[j,k]
    # No, that's not the standard formulation.
    
    # Let me think again. The standard way:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a bilinear map from (A,B) to C.
    # A tensor decomposition of rank r means we can write:
    # C[i,k] = sum_{r=0}^{rank-1} sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # Wait, that doesn't look right dimensionally.
    
    # Actually, I think the formulation is:
    # The multiplication tensor M has entries M[i,j,k] = 1 for the "diagonal" in some sense
    # No, let me look at this differently.
    
    # I believe the correct formulation is that we want:
    # For all i, j, k: the product A[i,j]*B[j,k] contributes to C[i,k]
    # So the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And we decompose T as a sum of rank-1 tensors:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # So we need: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # The trivial decomposition: one term per (i,j,k), with U[i*m+j,col]=1, V[j*p+k,col]=1, W[k*n+i,col]=1
    # This gives rank = n*m*p = 42
    
    # Can we do better? We need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    # Let me try to find a lower rank solution.
    
    # For now, let me just output the trivial one and verify it works.
    # The score is 1.0 for rank 42, so any improvement would be better.
    
    # Let me try rank 28 or 21.
    
    # Actually, I'll just use the trivial decomposition for now to ensure correctness.
    rank = 42
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
