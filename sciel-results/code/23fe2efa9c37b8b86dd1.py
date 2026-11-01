"""Matrix multiplication 2,5,5 - rank minimization via structured search."""
import time
import numpy as np
import itertools

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition of rank n*m*p
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    U_sb_np = np.array(U_sb, dtype=float)
    V_sb_np = np.array(V_sb, dtype=float)
    W_sb_np = np.array(W_sb, dtype=float)
    
    # The matrix multiplication tensor T[a, b, c] where:
    # a = i*m + j (row of A, col of A / row of B)
    # b = j*p + k (row of B, col of B)
    # c = k*n + i (row of C = col of B, col of C = row of A)
    # T[a, b, c] = 1 if a, b, c are consistent (same j, k, i), else 0
    # Specifically: if a = i*m+j, b = j*p+k, c = k*n+i then T=1
    
    # Let's build the full tensor for verification
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify_decomposition(U, V, W, rank):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Efficient check: for each (a,b), compute the vector over c
        # Actually, let's just check all entries
        # T_reconstructed[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a bilinear form in a,b,c
        
        # We can compute this as: for each r, outer product
        # But that's O(rank * n*m * m*p * p*n) which is fine for small rank
        
        # More efficient: 
        # For each a, b: T_recon[a,b,:] = sum_r (U[a,r]*V[b,r]) * W[:,r]
        # Let M_ab = U[a,:] * V[b,:] (element-wise), then T_recon[a,b,:] = M_ab @ W.T
        # But we need to check against T[a,b,:]
        
        # Even more efficient: check only the non-zero entries of T
        # T[a,b,c] is non-zero only when a=i*m+j, b=j*p+k, c=k*n+i
        # There are exactly n*m*p = 50 such entries
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U_np[a, r] * V_np[b, r] * W_np[c, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Also need to check that T_recon[a,b,c] = 0 for entries where T[a,b,c] = 0
        # This is more expensive. Let's do a sparse check.
        # For a random subset of zero entries, verify they are close to 0.
        
        # Actually, for correctness we need ALL entries to match.
        # Let's do a full check but efficiently.
        # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        
        # Build the reconstructed tensor
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U_np[:,r], V_np[:,r], W_np[:,r])
        
        if not np.allclose(T_recon, T, atol=1e-9):
            return False
        
        return True
    
    # Try to find a lower rank decomposition
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Strategy: try structured decompositions with lower rank
    # Key insight: the tensor has a specific structure.
    # For M_{2,5,5}, we can try to exploit the small n=2.
    
    # One approach: use the fact that for each (i,k) pair in the output,
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a sum of m=5 products.
    
    # Let's try a direct search for small ranks with integer/half-integer entries.
    # We'll use a randomized approach with structured initialization.
    
    def try_rank(r):
        """Try to find a rank-r decomposition."""
        if r >= best_rank:
            return None
        
        # Use random restarts with structured patterns
        np.random.seed(42 + r)
        
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small random values
            U = np.random.randint(-2, 3, size=(n*m, r)).astype(float) * 0.5
            V = np.random.randint(-2, 3, size=(m*p, r)).astype(float) * 0.5
            W = np.random.randint(-2, 3, size=(p*n, r)).astype(float) * 0.5
            
            # Simple local search: try flipping individual entries
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Compute current error
                T_recon = np.zeros((n*m, m*p, p*n))
                for ri in range(r):
                    T_recon += np.einsum('a,b,c->abc', U[:,ri], V[:,ri], W[:,ri])
                error = np.sum((T_recon - T)**2)
                
                if error < 1e-9:
                    return (U.tolist(), V.tolist(), W.tolist(), r)
                
                # Try a random perturbation
                mat = np.random.choice([U, V, W])
                row = np.random.randint(mat.shape[0])
                col = np.random.randint(mat.shape[1])
                old_val = mat[row, col]
                mat[row, col] += np.random.choice([-1, -0.5, 0.5, 1])
                
                # Check if error decreased
                T_recon_new = np.zeros((n*m, m*p, p*n))
                for ri in range(r):
                    T_recon_new += np.einsum('a,b,c->abc', U[:,ri], V[:,ri], W[:,ri])
                error_new = np.sum((T_recon_new - T)**2)
                
                if error_new > error + 1e-12:
                    mat[row, col] = old_val  # revert
        
        return None
    
    # Try ranks from schoolbook-1 down to a reasonable lower bound
    # Lower bound for matrix multiplication rank is at least max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # But we can try a few ranks below 50
    
    for r in range(rank_sb - 1, max(20, rank_sb - 20), -1):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(r)
        if result is not None:
            U_res, V_res, W_res, rank_res = result
            if verify_decomposition(U_res, V_res, W_res, rank_res):
                best_rank = rank_res
                best_U, best_V, best_W = U_res, V_res, W_res
                break
    
    # If we found a better decomposition, return it
    if best_rank < rank_sb:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Otherwise return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
