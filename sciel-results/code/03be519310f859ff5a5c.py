"""Matrix multiplication 4,5,5 - tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125  # seconds, leave safety margin before 150s kill

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the matmul tensor."""
        # The matmul tensor T[i,j,k] = delta_{i,j,k} in the sense that
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # Rank-r decomposition: for each column c, U[:,c], V[:,c], W[:,c]
        # The product sum_c outer(U[:,c], V[:,c]) should match the bilinear form
        # More precisely: for all A,B, C = A*B iff
        # sum_c U_c * (V_c^T B A^T) matches... 
        # Actually the standard check: the tensor T with T[i,j,k]=1 if (i,j,k) is valid
        # For matmul: C[i,k] = sum_j A[i,j] B[j,k]
        # Rank-1 term c contributes: U_c[i] * V_c[j] * W_c[k] to C[i,k] where
        # U_c has indices over (n*m) flattened (i,j), V_c over (m*p) flattened (j,k),
        # W_c over (p*n) flattened (k,i).
        # 
        # The correct check: build the full matmul tensor and verify.
        # T[i,j,k] = 1 for the identity mapping.
        # sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] should equal delta for matmul...
        # Actually no. Let me think more carefully.
        #
        # In the standard formulation: C = A * B where C[i,k] = sum_j A[i,j] * B[j,k]
        # A rank-1 tensor decomposition gives:
        # C[i,k] = sum_c (sum_j U_c[i,j] * A[i,j]) * (sum_k V_c[j,k] * B[j,k]) ... no
        #
        # The proper way: The matrix multiplication tensor T is a 3-tensor where
        # T[i,j,k] = 1 if the triple (i,j,k) corresponds to a "valid" product term.
        # For C = A*B: C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor has T[i,j,k] = 1 (it's just the identity in the middle index pairing)
        #
        # Rank-r decomposition: T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
        # Wait, that's not quite right either. Let me use the standard convention.
        #
        # Standard: We want U of shape (n*m, r), V of shape (m*p, r), W of shape (p*n, r)
        # such that for the matmul tensor:
        # For each (i,j,k) with 0<=i<n, 0<=j<m, 0<=k<p:
        #   T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # where T[i,j,k] = 1 (the matmul tensor is all ones in the valid entries,
        # which is every (i,j,k) triple, and the "product" is just the identity pairing).
        #
        # Actually, the matmul tensor IS all ones. T[i,j,k] = 1 for all valid (i,j,k).
        # And we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
        #
        # But wait, that's not right. The matmul tensor encodes the bilinear map.
        # T[i,j,k] = 1 means that C[i,k] gets contribution A[i,j]*B[j,k].
        # So yes, T[i,j,k] = 1 for all (i,j,k) in [0,n) x [0,m) x [0,p).
        #
        # And the rank-r decomposition satisfies:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k] = 1
        # for all i in [0,n), j in [0,m), k in [0,p).
        
        # Efficient check using tensor contraction
        # Reshape: U is (n*m, r), V is (m*p, r), W is (p*n, r)
        # For each c, we get a rank-1 tensor of size n x m x p:
        #   tensor_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We need sum_c tensor_c[i,j,k] = 1 for all (i,j,k)
        
        # Build the full tensor
        # For efficiency, check via random sampling or full enumeration
        # n*m*p = 100 entries, so full check is fine
        total = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            # tensor_c[i,j,k] = Uc[i,j] * Vc[j,k] * Wc[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
        
        # Check if total equals all ones
        expected = np.ones((n, m, p))
        if not np.allclose(total, expected, atol=tol):
            return False
        return True

    # Try to find a lower-rank decomposition
    # Strategy: random search with verification
    
    best_rank = n * m * p
    best_decomp = None
    
    # First, get the schoolbook solution
    U0, V0, W0 = schoolbook()
    best_rank = U0.shape[1]
    best_decomp = (U0, V0, W0)
    
    # Try random searches for lower rank
    # For small ranks, random initialization and iterative refinement
    # But verification is expensive, so let's be strategic.
    
    # Known results: 
    # 2x2x2 has rank 7 (Strassen)
    # For 4x5x5, let's try various approaches
    
    # Approach 1: Use Strassen-like block decomposition
    # Split 4x5 into blocks and use Strassen on 2x2 blocks
    
    # Actually, let me try a direct random search with a time limit
    # Try ranks from some lower bound up to schoolbook - 1
    
    # Lower bound for rank: max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # But that's very loose. A better lower bound is n*p = 20 (since C has 20 entries)
    # Actually the rank must be at least max(n*m, m*p, p*n) for the tensor to have
    # enough "freedom", but that's not a proven lower bound.
    
    # Let's try random search for a few ranks
    random.seed(42)
    np.random.seed(42)
    
    # Try a small number of random attempts at various ranks
    ranks_to_try = []
    # Start from reasonable lower ranks
    for r in range(30, best_rank):
        ranks_to_try.append(r)
    
    # Limit the number of attempts to stay within time budget
    max_attempts_per_rank = 5
    attempts_done = 0
    max_total_attempts = 50
    
    for r in reversed(ranks_to_try):  # Try higher ranks first (more likely to succeed)
        if time.time() - start_time > deadline:
            break
        if attempts_done >= max_total_attempts:
            break
        
        for attempt in range(max_attempts_per_rank):
            if time.time() - start_time > deadline:
                break
            if attempts_done >= max_total_attempts:
                break
            attempts_done += 1
            
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Simple gradient-free approach: just check if it works
            # (Unlikely for random init, but let's try)
            if verify(U, V, W):
                best_rank = r
                best_decomp = (U, V, W)
                break
        
        if best_rank < best_rank:  # If we found something better
            break
    
    # If we found a better decomposition, return it
    if best_decomp is not None and best_rank < n * m * p:
        U, V, W = best_decomp
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
