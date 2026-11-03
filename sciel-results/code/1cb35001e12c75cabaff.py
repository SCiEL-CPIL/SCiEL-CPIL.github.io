"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Schoolbook decomposition (rank = n*m*p = 63)
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return U, V, W, rank
    
    # Strassen decomposition for 2x2 x 2x2 (rank 7)
    def strassen_2x2():
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,0]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3 = A[0,0] * (B[1,0] - B[1,1])
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # M4 = A[1,1] * (B[0,1] - B[0,0])
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1,1]
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[1][4] = 1; W[0][4] = -1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[1,0])
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[3][5] = 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[0,1] + B[1,1])
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W, 7
    
    # Verify a decomposition by checking the tensor identity
    def verify(U, V, W, tol=1e-9):
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check all (i, j, k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # T[a,b,c] should be 1
                    val = 0.0
                    for col in range(rank):
                        val += U_arr[a, col] * V_arr[b, col] * W_arr[c, col]
                    if abs(val - 1.0) > tol:
                        return False
        # Check some zero entries to ensure no spurious contributions
        # For efficiency, check a sample of zero entries
        import random
        random.seed(42)
        for _ in range(100):
            a = random.randint(0, n*m-1)
            b = random.randint(0, m*p-1)
            c = random.randint(0, p*n-1)
            # Check if this is a valid (i,j,k) triple
            i1, j1 = divmod(a, m)
            j2, k1 = divmod(b, p)
            k2, i2 = divmod(c, n)
            if j1 != j2 or i1 != i2 or k1 != k2:
                # Should be 0
                val = 0.0
                for col in range(rank):
                    val += U_arr[a, col] * V_arr[b, col] * W_arr[c, col]
                if abs(val) > tol:
                    return False
        return True
    
    # Try to build a decomposition using Strassen-like blocks
    # Strategy: Treat 3x3x7 as combining smaller blocks
    # For 3x3x3, we can use a known rank-21 decomposition (or schoolbook 27)
    # For 3x3x4, we can try to find a lower rank
    
    # Let's try a direct approach: use the schoolbook but attempt rank reduction
    # via linear algebra on the tensor
    
    def build_tensor(n, m, p):
        """Build the matrix multiplication tensor as a 3D array."""
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    T = build_tensor(n, m, p)
    
    # Start with schoolbook
    U_school, V_school, W_school, rank_school = schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    # Try to reduce rank by finding linear dependencies
    # We can try various target ranks below 63
    # For each target rank r, try to find U, V, W with r columns
    
    def try_rank_reduction(target_rank, max_time=30):
        """Try to find a decomposition with target_rank factors."""
        if time.time() > deadline - 10:
            return None
        
        U_arr = np.array(U_school, dtype=float)
        V_arr = np.array(V_school, dtype=float)
        W_arr = np.array(W_school, dtype=float)
        
        # Try random projections to reduce rank
        # This is a heuristic: project the 63 columns to target_rank
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            # Create a random mixing matrix
            mix = np.random.randn(rank_school, target_rank)
            # New factors are linear combinations of old ones
            U_new = U_arr @ mix
            V_new = V_arr @ mix
            W_new = W_arr @ mix
            
            # This won't work directly because we need the same mixing for all
            # Let's try a different approach: iterative optimization
            
            # Actually, let's try a simpler approach: 
            # Use the fact that for small matrices, we can use known decompositions
            pass
        
        return None
    
    # For 3x3x7, let's try a block approach
    # Split the 7 columns of B into 3+4
    # Compute A(3x3) * B1(3x3) using a 3x3x3 decomposition
    # Compute A(3x3) * B2(3x4) using a 3x3x4 decomposition
    
    # For 3x3x3, the known minimum rank is 21 (from literature)
    # For 3x3x4, we need to find a decomposition
    
    # Let's use a practical approach: try to find a valid decomposition
    # with rank less than 63 using numerical optimization
    
    def optimize_decomposition(target_rank, n_iter=100, tol=1e-8):
        """Optimize U, V, W to minimize ||T - sum U_c ⊗ V_c ⊗ W_c||."""
        if time.time() > deadline - 15:
            return None
        
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # Alternating least squares
        for it in range(n_iter):
            if time.time() > deadline - 15:
                break
            
            # Update U: for each row a, solve least squares
            for a in range(n*m):
                # T[a, b, c] = sum_col U[a, col] * V[b, col] * W[c, col]
                # For fixed a, this is linear in U[a, :]
                # Form the design matrix
                B_mat = np.zeros((m*p*p*n, target_rank))
                for b in range(m*p):
                    for c in range(p*n):
                        B_mat[b*p*n + c, :] = V[b, :] * W[c, :]
                rhs = T[a, :, :].flatten()
                U[a, :] = np.linalg.lstsq(B_mat, rhs, rcond=None)[0]
            
            # Update V
            for b in range(m*p):
                B_mat = np.zeros((n*m*p*n, target_rank))
                for a in range(n*m):
                    for c in range(p*n):
                        B_mat[a*p*n + c, :] = U[a, :] * W[c, :]
                rhs = T[:, b, :].flatten()
                V[b, :] = np.linalg.lstsq(B_mat, rhs, rcond=None)[0]
            
            # Update W
            for c in range(p*n):
                B_mat = np.zeros((n*m*m*p, target_rank))
                for a in range(n*m):
                    for b in range(m*p):
                        B_mat[a*m*p + b, :] = U[a, :] * V[b, :]
                rhs = T[:, :, c].flatten()
                W[c, :] = np.linalg.lstsq(B_mat, rhs, rcond=None)[0]
        
        # Check if we have a valid decomposition
        U_list = [[float(x) for x in U[a, :]] for a in range(n*m)]
        V_list = [[float(x) for x in V[b, :]] for b in range(m*p)]
        W_list = [[float(x) for x in W[c, :]] for c in range(p*n)]
        
        if verify(U_list, V_list, W_list, tol=1e-6):
            return U_list, V_list, W_list, target_rank
        
        return None
    
    # Try some target ranks
    for target_rank in [50, 45, 40, 35, 30, 28, 25, 21]:
        if time.time() > deadline - 20:
            break
        result = optimize_decomposition(target_rank, n_iter=50)
        if result is not None:
            U_list, V_list, W_list, r = result
            if r < best_rank:
                best_U, best_V, best_W, best_rank = U_list, V_list, W_list, r
                print(f"Found rank {r} decomposition!")
    
    # If we found a better decomposition, return it
    if best_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Otherwise, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
