"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # 110s safety margin within 150s limit
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    
    def build_tensor(n, m, p):
        """Build the true matrix multiplication tensor."""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        return T
    
    def verify_decomp(U, V, W, n, m, p):
        """Verify that the decomposition exactly reconstructs the matrix mult tensor."""
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        T_true = build_tensor(n, m, p)
        return np.allclose(T_recon, T_true, atol=1e-10)
    
    def schoolbook_decomp(n, m, p):
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    def block_strassen_decomp(n, m, p):
        """
        Block-Strassen decomposition for n×m×p matrix multiplication.
        Split A into 2 blocks of rows, B into 2 blocks of cols, C into 2×2 blocks.
        Uses 7 multiplications instead of 8.
        """
        # For n×m×p: A is n×m, B is m×p, C is n×p
        # Split A: A1 (n1×m), A2 (n2×m) where n1+n2=n
        # Split B: B1 (m×p1), B2 (m×p2) where p1+p2=p
        # C11 = A1*B1, C12 = A1*B2, C21 = A2*B1, C22 = A2*B2
        # Strassen: M1=(A1+A2)*B1, M2=A1*(B2-B1), M3=A2*(B1+B2), M4=A2*B2
        #           M5=(A1+A2)*(B2-B1), M6=(A1-A2)*B1, M7=(A1-A2)*(B2+B1)
        # C11 = M4+M1+M2+M5, C12=M3+M5, C21=M2+M4, C22=M1-M3+M4+M7
        
        n1 = n // 2
        n2 = n - n1
        p1 = p // 2
        p2 = p - p1
        
        # Each "block multiplication" is a sub-problem
        # M1: (n1+n2)×m * m×p1 -> (n1+n2)×p1, rank = (n1+n2)*m*p1
        # M2: n1×m * m×p2 -> n1×p2, rank = n1*m*p2
        # M3: n2×m * m×p1 -> n2×p1, rank = n2*m*p1
        # M4: n2×m * m×p2 -> n2×p2, rank = n2*m*p2
        # M5: (n1+n2)×m * m×p2 -> (n1+n2)×p2, rank = (n1+n2)*m*p2
        # M6: n1×m * m×p1 -> n1×p1, rank = n1*m*p1
        # M7: n1×m * m×p2 -> n1×p2, rank = n1*m*p2
        
        ranks = [
            (n1+n2)*m*p1,  # M1
            n1*m*p2,       # M2
            n2*m*p1,       # M3
            n2*m*p2,       # M4
            (n1+n2)*m*p2,  # M5
            n1*m*p1,       # M6
            n1*m*p2,       # M7
        ]
        total_rank = sum(ranks)
        
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        col = 0
        # M1 = (A1+A2) * B1
        # A1+A2: rows 0..n-1, cols 0..m-1 (all rows)
        # B1: rows 0..m-1, cols 0..p1-1
        # Result: (n1+n2)×p1, contributes to C11 (rows 0..n1-1, cols 0..p1-1)
        # and C21 (rows n1..n-1, cols 0..p1-1)
        for i in range(n1+n2):
            for j in range(m):
                for k in range(p1):
                    U[i*m + j, col] = 1  # A[i,j] (all rows)
                    V[j*p + k, col] = 1  # B[j,k] for k<p1
                    # Contribution to C[i,k] for i<n1 (C11) and i>=n1 (C21)
                    W[k*n + i, col] = 1
                    col += 1
        
        # M2 = A1 * (B2-B1)
        # A1: rows 0..n1-1
        # B2-B1: B2 has cols p1..p-1, B1 has cols 0..p1-1
        # So (B2-B1)[j,k] = B[j,p1+k] - B[j,k] for k=0..p2-1
        # Result: n1×p2, contributes to C11 (rows 0..n1-1, cols 0..p1-1) and C12 (rows 0..n1-1, cols p1..p-1)
        for i in range(n1):
            for j in range(m):
                for k in range(p2):
                    U[i*m + j, col] = 1  # A[i,j]
                    # B2-B1: B[j,p1+k] - B[j,k]
                    V[j*p + (p1+k), col] = 1   # +B[j,p1+k]
                    V[j*p + k, col] = -1        # -B[j,k]
                    # Contributes to C11: C[i,k] and C12: C[i,p1+k]
                    W[k*n + i, col] = 1         # C11 part
                    W[(p1+k)*n + i, col] = 1    # C12 part
                    col += 1
        
        # M3 = A2 * (B1+B2)
        # A2: rows n1..n-1
        # B1+B2: B1 has cols 0..p1-1, B2 has cols p1..p-1
        # (B1+B2)[j,k] = B[j,k] + B[j,p1+k] for k=0..p1-1 (only p1 cols since p1<=p2 typically)
        # Actually B1 is m×p1 and B2 is m×p2, so B1+B2 doesn't make sense directly.
        # Let me reconsider: M3 = A2*(B1+B2) where we pad B1 to m×p2 or B2 to m×p1
        # Standard Strassen for square: M3 = A2*(B1+B2)
        # For rectangular: M3 = A2 * (B1 padded + B2) - this gets complicated.
        # Let me use a simpler approach.
        
        # Actually, let me just use the schoolbook for now and try a different reduction.
        return None, None, None
    
    # Start with schoolbook
    U, V, W = schoolbook_decomp(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if not verify_decomp(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to reduce rank using null-space elimination
    # If columns are linearly dependent, we can potentially remove some
    
    def try_nullspace_reduction(U, V, W, time_limit):
        """Try to reduce rank by exploiting linear dependencies."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        start = time.time()
        rank = U.shape[1]
        
        # Try to find if any column is a linear combination of others
        # This is expensive, so we do a simple check
        
        # For each column, check if it's in the span of others
        # Use QR decomposition on U
        for _ in range(5):
            if time.time() - start > time_limit:
                break
            
            # Check rank of U
            u_rank = np.linalg.matrix_rank(U)
            if u_rank < rank:
                # There are linearly dependent columns
                # Find which columns can be removed
                # This requires more careful handling
                pass
            break
        
        return U, V, W, rank
    
    # Try random perturbations and optimization
    # This is a heuristic approach
    
    rng = np.random.RandomState(42)
    
    for target_rank in range(best_rank - 1, max(best_rank - 10, 60), -1):
        if time.time() > deadline - 30:
            break
        
        # Try to find a decomposition of rank target_rank
        # Initialize with random values
        success = False
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            
            U_rand = rng.randn(n*m, target_rank) * 0.1
            V_rand = rng.randn(m*p, target_rank) * 0.1
            W_rand = rng.randn(p*n, target_rank) * 0.1
            
            # Use iterative refinement
            # Minimize ||T - sum U[:,t] outer V[:,t] outer W[:,t]||
            # This is a non-convex problem, so we use a simple approach
            
            # Gradient descent on the factors
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Compute residual
                T_recon = np.einsum('at,bt,ct->abc', U_rand, V_rand, W_rand)
                T_true = build_tensor(n, m, p)
                residual = T_true - T_recon
                
                # Gradients
                # dL/dU[a,t] = sum_{b,c} residual[a,b,c] * V[b,t] * W[c,t]
                dU = np.einsum('abc,bt,ct->at', residual, V_rand, W_rand)
                dV = np.einsum('abc,at,ct->bt', residual, U_rand, W_rand)
                dW = np.einsum('abc,at,bt->ct', residual, U_rand, V_rand)
                
                # Update
                U_rand -= lr * dU
                V_rand -= lr * dV
                W_rand -= lr * dW
                
                # Check convergence
                if np.linalg.norm(residual) < 1e-8:
                    success = True
                    break
            
            if success:
                # Verify exact decomposition
                if verify_decomp(U_rand, V_rand, W_rand, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                    break
    
    # Use the best decomposition found
    U, V, W = best_U, best_V, best_W
    
    if not verify_decomp(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = schoolbook_decomp(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
