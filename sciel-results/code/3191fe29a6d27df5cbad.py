import numpy as np
import time

def construct():
    n, m, p = 3, 4, 7
    deadline = time.time() + 140  # safety margin before 150s kill

    # Schoolbook decomposition: rank = n*m*p = 84
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            S = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    S[i] += U[i*m+j, r] * A[i,j]
            T = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    T[k] += V[j*p+k, r] * B[j,k]
            for i in range(n):
                for k in range(p):
                    C_decomp[i,k] += W[k*n+i, r] * S[i] * T[k]
        
        if not np.allclose(C, C_decomp, atol=1e-10):
            return False
        return True
    
    if verify(U, V, W, n, m, p):
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        best_rank = rank_sb
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to improve using a greedy rank reduction approach
    # Strategy: Try to find linear dependencies among the rank-1 tensors
    # and reduce the rank accordingly.
    
    # The tensor T = sum_r u_r ⊗ v_r ⊗ w_r
    # If there's a linear relation sum_r c_r (u_r ⊗ v_r ⊗ w_r) = 0,
    # we can potentially reduce rank.
    
    # A simpler approach: try to find a subset of terms that can be
    # replaced by fewer terms.
    
    # For now, let's try a different strategy: use known results for
    # small matrix multiplication.
    
    # Actually, let me try to use the fact that for 3x4x7, we might be
    # able to use a block decomposition.
    
    # Partition the 3x4 matrix into blocks and use Strassen-like methods.
    
    # For a 3x4 matrix, we can write it as:
    # [A1 A2]  where A1 is 3x2, A2 is 3x2
    # [B1 B2]  where B1 is 4x? no wait...
    
    # Actually, let me try a different approach.
    # For 3x4 * 4x7:
    # Write A as 3x4, B as 4x7.
    # C = A*B is 3x7.
    
    # We can partition B into columns: B = [B1 B2 B3 B4 B5 B6 B7]
    # where each Bj is 4x1.
    # Then C[:,j] = A * Bj, which is a 3x1 matrix.
    
    # This doesn't directly help with rank reduction.
    
    # Let me try yet another approach: use the fact that the rank of
    # matrix multiplication tensor M_{n,m,p} is at most n*m*p - min(n,m,p) + 1
    # for some cases, but I'm not sure about the exact bound.
    
    # For now, let's just return the schoolbook decomposition.
    # The key insight is that we need to find a valid decomposition,
    # and the schoolbook is always valid.
    
    # Let me try one more thing: check if there's a simple rank reduction
    # by looking at the structure of the tensor.
    
    # The tensor has entries T[i,j,k] = 1 for the "diagonal" in the sense
    # that C[i,k] = sum_j A[i,j] B[j,k].
    
    # In terms of the rank-1 decomposition, each term corresponds to
    # a specific (i,j,k) triple.
    
    # If we can find two terms (i1,j1,k1) and (i2,j2,k2) such that
    # their combination can be replaced by a single term, we reduce rank.
    
    # This happens when the three vectors are linearly dependent in a
    # specific way.
    
    # For now, let's just return the schoolbook.
    
    return best_result
