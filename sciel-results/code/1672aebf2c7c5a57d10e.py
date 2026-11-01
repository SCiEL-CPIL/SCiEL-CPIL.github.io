"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a tensor decomposition for 2x5 by 5x5 matrix multiplication.
    Uses a search strategy to find a lower-rank decomposition.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Start with schoolbook (rank 50)
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    best_U = [[0] * schoolbook_rank for _ in range(n * m)]
    best_V = [[0] * schoolbook_rank for _ in range(m * p)]
    best_W = [[0] * schoolbook_rank for _ in range(p * n)]
    
    # Fill schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # Try to find a better decomposition using random initialization + optimization
    # For matrix multiplication A(2x5) * B(5x5) = C(2x5)
    # The tensor has dimensions (n*m, m*p, p*n) = (10, 25, 10)
    # We want to find rank-r decomposition: T = sum_{t=1}^r u_t ⊗ v_t ⊗ w_t
    
    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        # Build the tensor T[i,j,k] where i in [0,n*m), j in [0,m*p), k in [0,p*n)
        # T[i,j,k] = 1 if and only if the (i,j,k) entry of the matmul tensor is 1
        # For matmul: C[i, k] = sum_j A[i, j] * B[j, k]
        # In our indexing: i = row of A (0..n*m-1), j = col of A / row of B (0..m*p-1), k = col of B (0..p*n-1)
        # Actually the tensor convention: T[a, b, c] where a indexes A, b indexes B, c indexes C
        # A has n*m entries, B has m*p entries, C has p*n entries
        # T[a, b, c] = 1 iff the product of A[a] and B[b] contributes to C[c]
        
        # Let's decode indices:
        # a = i*m + j1 where i in [0,n), j1 in [0,m) -> A[i, j1]
        # b = j2*p + k1 where j2 in [0,m), k1 in [0,p) -> B[j2, k1]
        # c = k2*n + i2 where k2 in [0,p), i2 in [0,n) -> C[i2, k2]
        # The contribution is non-zero iff j1 == j2 (the inner dimension matches)
        # So T[a, b, c] = 1 iff j1 == j2, i.e., (a % m) == (b // p)
        
        # Verify: sum_t U[a,t] * V[b,t] * W[c,t] should equal T[a,b,c]
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Compute the reconstructed tensor
        # Reconstructed[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        # This is equivalent to: for each t, outer product of U[:,t], V[:,t], W[:,t]
        
        # Efficient check: compute for all (a,b,c) where T[a,b,c] should be 1
        # and check that reconstruction matches
        
        # Build expected tensor
        expected = np.zeros((n * m, m * p, p * n), dtype=float)
        for a in range(n * m):
            for b in range(m * p):
                j1 = a % m
                j2 = b // p
                if j1 == j2:
                    # Find which c values this contributes to
                    i = a // m
                    k1 = b % p
                    # c = k1 * n + i
                    c = k1 * n + i
                    expected[a, b, c] = 1
        
        # Reconstruct
        reconstructed = np.zeros((n * m, m * p, p * n), dtype=float)
        for t in range(rank):
            reconstructed += np.einsum('a,b,c->abc', U_arr[:, t], V_arr[:, t], W_arr[:, t])
        
        diff = np.max(np.abs(reconstructed - expected))
        return diff < tol, diff
    
    ok, diff = verify_decomposition(best_U, best_V, best_W)
    
    # Try random search for lower rank
    np.random.seed(42)
    
    for target_rank in range(40, 49):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Simple gradient-free check: just verify if this random init happens to work (it won't)
            # Instead, let's try a structured approach
            
            # Actually, for exact integer/half-integer solutions, random float search won't work well.
            # Let's try a different approach: start from schoolbook and try to merge terms.
            
            # Try structured decomposition: share some terms between the two rows of A
            # A has 2 rows, each of length 5. B is 5x5.
            # C[i, k] = sum_j A[i, j] * B[j, k]
            # We can write this as: for each j, we compute A[i, j] * B[j, :] for i=0,1
            # That gives us 5 * 2 = 10 products per k, total 50.
            
            # Alternative: treat A as a 2x5 matrix. For each column j of A (length 2),
            # we multiply by row j of B (length 5). The result is a 2x5 matrix.
            # This is rank 5 decomposition of the 2x5 matmul if we could do outer products,
            # but each "term" is a 2x5 outer product, which isn't a single rank-1 tensor.
            
            # A rank-1 tensor term is u ⊗ v ⊗ w where u has length 10, v has length 25, w has length 10.
            # For it to contribute to matmul, we need:
            # u[i*m+j] * v[j'*p+k] * w[k'*n+i'] is non-zero only when j==j' and i==i' and k==k'
            # So each rank-1 term can only contribute to one (i, j, k) triple where j is the shared index.
            # This means we need at least 50 rank-1 terms for an exact decomposition with this structure.
            
            # Wait, that's not right. A rank-1 tensor u⊗v⊗w has entries u[a]*v[b]*w[c] for all a,b,c.
            # It doesn't have to be zero except at specific positions. The sum over all rank-1 terms
            # must equal the matmul tensor, which has 1s at specific positions and 0s elsewhere.
            
            # So we need: sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c] for all a,b,c
            # where T[a,b,c] = 1 if (a%m)==(b//p) and c == (b%p)*n + (a//m), else 0.
            
            # This is a system of equations. The minimum rank is the tensor rank.
            # For 2x5x5, the tensor rank is known to be... let me think.
            # The bilinear complexity of 2x5 by 5x5 matmul.
            # Lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
            # Upper bound: schoolbook = 50
            # Known results: for m x n x p matmul, the rank is between max(mn, np, pm) and mnp.
            
            # Let me try to construct a rank-25 decomposition.
            # If we can achieve rank 25, that would be optimal (matching the lower bound).
            
            # Construction idea: 
            # C[i, k] = sum_j A[i, j] * B[j, k]
            # Let's define for each j (0..4) and each (i, k) pair, but that's 5*10=50 terms.
            
            # Alternative: use the fact that we have 2 rows.
            # Define U vectors that are sparse, V vectors that are sparse, W vectors that are sparse.
            
            # Try: for each j in 0..4, create 5 terms (one for each k in 0..4)
            # Term (j, k): U has 1s at positions (0, j) and (1, j) -> indices j and 5+j
            #              V has 1 at position (j, k) -> index j*5+k
            #              W has 1s at positions (k, 0) and (k, 1) -> indices k*2 and k*2+1
            # This gives: U[j]*V[j*5+k]*W[k*2] contributes to C[0,k] via A[0,j]*B[j,k]
            #            U[5+j]*V[j*5+k]*W[k*2+1] contributes to C[1,k] via A[1,j]*B[j,k]
            # But we need separate terms for each (i,j,k) to get the right coefficients.
            
            # Actually, let's think differently. We want:
            # sum_t U[a,t] V[b,t] W[c,t] = T[a,b,c]
            
            # Let's try a decomposition where each term handles one (j, k) pair but both i values.
            # For fixed j, k: we need contributions to:
            #   (a=0*m+j, b=j*p+k, c=k*n+0) = (j, j*5+k, k*2)
            #   (a=1*m+j, b=j*p+k, c=k*n+1) = (5+j, j*5+k, k*2+1)
            
            # Can we do this with 1 term? We'd need:
            #   U[j,t]*V[j*5+k,t]*W[k*2,t] = 1
            #   U[5+j,t]*V[j*5+k,t]*W[k*2+1,t] = 1
            #   All other products = 0
            
            # If we set V[j*5+k,t] = 1, then we need:
            #   U[j,t]*W[k*2,t] = 1
            #   U[5+j,t]*W[k*2+1,t] = 1
            #   And for all other (a', b', c') not in {(j, j*5+k, k*2), (5+j, j*5+k, k*2+1)},
            #   U[a',t]*V[b',t]*W[c',t] = 0
            
            # Since V[j*5+k,t]=1 and V[b',t]=0 for b' != j*5+k, the second condition is automatically satisfied
            # for b' != j*5+k. For b' = j*5+k, we need U[a',t]*W[c',t]=0 for all (a',c') except the two.
            
            # Set U[j,t]=1, U[5+j,t]=1, all other U[a',t]=0.
            # Then we need W[k*2,t]=1, W[k*2+1,t]=1, all other W[c',t]=0.
            # Check: U[j,t]*V[j*5+k,t]*W[k*2,t] = 1*1*1 = 1 ✓
            #        U[5+j,t]*V[j*5+k,t]*W[k*2+1,t] = 1*1*1 = 1 ✓
            #        For other c': W[c',t]=0 so product is 0 ✓
            #        For other a': U[a',t]=0 so product is 0 ✓
            
            # This works! Each (j,k) pair needs 1 term, total 5*5=25 terms.
            
            # Let's construct this rank-25 decomposition!
            pass
    
    # Construct rank-25 decomposition
    rank25 = m * p  # 5 * 5 = 25
    U25 = [[0] * rank25 for _ in range(n * m)]
    V25 = [[0] * rank25 for _ in range(m * p)]
    W25 = [[0] * rank25 for _ in range(p * n)]
    
    t = 0
    for j in range(m):
        for k in range(p):
            # U[j, t] = 1 (for i=0)
            # U[m+j, t] = 1 (for i=1)
            # V[j*p+k, t] = 1
            # W[k*n, t] = 1 (for i=0)
            # W[k*n+1, t] = 1 (for i=1)
            U25[j][t] = 1
            U25[m + j][t] = 1
            V25[j * p + k][t] = 1
            W25[k * n][t] = 1
            W25[k * n + 1][t] = 1
            t += 1
    
    ok25, diff25 = verify_decomposition(U25, V25, W25)
    if ok25:
        best_rank = rank25
        best_U = U25
        best_V = V25
        best_W = W25
    
    # Try to go even lower? Lower bound is max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # So 25 is optimal if achievable.
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
