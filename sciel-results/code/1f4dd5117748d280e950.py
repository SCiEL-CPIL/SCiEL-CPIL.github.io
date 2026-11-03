import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def check_solution(U, V, W):
        """Check if U, V, W form an exact decomposition."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        # Check that other combinations give 0
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        if b != c:
                            continue  # Only need to check b==c case for non-trivial
                        for i in range(n):
                            for k in range(p):
                                if a == i and d == k:
                                    continue  # Already checked
                                s = 0
                                for r in range(rank):
                                    s += U_arr[a*m+b, r] * V_arr[c*p+d, r] * W_arr[k*n+i, r]
                                if abs(s) > 1e-9:
                                    return False
        return True
    
    def to_half_int(x):
        """Round to nearest half-integer."""
        return round(2*x) / 2
    
    def try_rank(R, max_restarts=10, max_iters=1000):
        """Try to find a decomposition of rank R."""
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize with small random values
            np.random.seed(42 + restart)
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            # Simple gradient descent
            lr = 0.01
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Compute gradient numerically (expensive but simple)
                # Actually, let's use a simpler approach: ALS
                # Fix U and V, solve for W
                # For each (i,k): W[k*n+i, r] appears in equations for all j
                # Σ_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all j
                
                # This is a linear system in W for each (i,k)
                # Let's set up the system
                W_new = np.zeros((p*n, R))
                for i in range(n):
                    for k in range(p):
                        # We have m equations:
                        # For j=0..m-1: Σ_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                        # Let A_jr = U[i*m+j, r] * V[j*p+k, r]
                        # Then: Σ_r A_jr * W[k*n+i, r] = 1 for all j
                        # This is: M * W[k*n+i, :] = 1_m where M[j,r] = A_jr
                        M = np.zeros((m, R))
                        for j in range(m):
                            for r in range(R):
                                M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        # Solve M * w = 1 (least squares)
                        try:
                            w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W_new[k*n+i, :] = w
                        except:
                            W_new[k*n+i, :] = W[k*n+i, :]
                W = W_new
                
                # Similarly solve for U and V
                U_new = np.zeros((n*m, R))
                for i in range(n):
                    for j in range(m):
                        # Equations involving U[i*m+j, r]:
                        # For all k: Σ_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                        # Let B_kr = V[j*p+k, r] * W[k*n+i, r]
                        # Then: Σ_r U[i*m+j, r] * B_kr = 1 for all k
                        M = np.zeros((p, R))
                        for k in range(p):
                            for r in range(R):
                                M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        try:
                            u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U_new[i*m+j, :] = u
                        except:
                            U_new[i*m+j, :] = U[i*m+j, :]
                U = U_new
                
                V_new = np.zeros((m*p, R))
                for j in range(m):
                    for k in range(p):
                        # Equations involving V[j*p+k, r]:
                        # For all i: Σ_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                        # Let C_ir = U[i*m+j, r] * W[k*n+i, r]
                        # Then: Σ_r V[j*p+k, r] * C_ir = 1 for all i
                        M = np.zeros((n, R))
                        for i in range(n):
                            for r in range(R):
                                M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        try:
                            v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V_new[j*p+k, :] = v
                        except:
                            V_new[j*p+k, :] = V[j*p+k, :]
                V = V_new
                
                # Check convergence
                if it % 10 == 0:
                    err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                err += (s - 1)**2
                    if err < 1e-6:
                        break
            
            # Round to half-integers
            U_hi = np.round(U * 2) / 2
            V_hi = np.round(V * 2) / 2
            W_hi = np.round(W * 2) / 2
            
            # Convert to lists
            U_list = U_hi.tolist()
            V_list = V_hi.tolist()
            W_list = W_hi.tolist()
            
            # Check exactness
            if check_solution(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    def try_structured_rank(R, max_restarts=5):
        """
        Try a structured ansatz using block-circulant / polynomial structure.
        
        Idea: For matrix multiplication, we can exploit the fact that the tensor
        has a convolutional structure. We use circulant matrices of size L = max(n, p)
        for the outer dimensions, and handle the middle dimension separately.
        
        The ansatz: 
        - U is block-structured: U[i*m+j, :] depends on i and j via a circulant pattern
        - V is block-structured: V[j*p+k, :] depends on j and k via a circulant pattern  
        - W is block-structured: W[k*n+i, :] depends on k and i via a circulant pattern
        
        We parameterize each block by a small number of coefficients (the polynomial terms).
        """
        L = max(n, p)  # Size of circulant blocks
        
        # Number of free parameters per "block"
        # For rank R, if we use circulant structure of size L, we need R to be a multiple of L
        # Or we can use a more flexible structure
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            np.random.seed(1000 + restart)
            
            # Initialize structured matrices
            # U[i*m + j, r] where r = block_idx * L + circ_idx
            # We'll use R columns, organized as (R // L) blocks of size L
            
            nblocks = R // L
            if nblocks * L < R:
                nblocks += 1
            
            # Parameterize U as: U[i*m+j, block*L + circ] = sum over polynomial terms
            # For simplicity, let's use a direct circulant structure
            # U[i*m+j, r] = f_U(i, j, r mod L, r // L)
            
            # Initialize random coefficients for the polynomial structure
            # U_block[i, j, b, c] where b = block, c = circulant shift
            U_coeff = np.random.randn(n, m, nblocks, L) * 0.5
            V_coeff = np.random.randn(m, p, nblocks, L) * 0.5
            W_coeff = np.random.randn(p, n, nblocks, L) * 0.5
            
            # Build U, V, W from coefficients
            def build_U(coeff):
                U_mat = np.zeros((n*m, R))
                for i in range(n):
                    for j in range(m):
                        for b in range(nblocks):
                            for c in range(L):
                                r = b * L + c
                                if r < R:
                                    U_mat[i*m+j, r] = coeff[i, j, b, c]
                return U_mat
            
            def build_V(coeff):
                V_mat = np.zeros((m*p, R))
                for j in range(m):
                    for k in range(p):
                        for b in range(nblocks):
                            for c in range(L):
                                r = b * L + c
                                if r < R:
                                    V_mat[j*p+k, r] = coeff[j, k, b, c]
                return V_mat
            
            def build_W(coeff):
                W_mat = np.zeros((p*n, R))
                for k in range(p):
                    for i in range(n):
                        for b in range(nblocks):
                            for c in range(L):
                                r = b * L + c
                                if r < R:
                                    W_mat[k*n+i, r] = coeff[k, i, b, c]
                return W_mat
            
            U = build_U(U_coeff)
            V = build_V(V_coeff)
            W = build_W(W_coeff)
            
            # Now optimize using ALS on the full matrices
            for it in range(200):
                if time.time() > deadline:
                    return None
                
                # Solve for W
                W_new = np.zeros((p*n, R))
                for i in range(n):
                    for k in range(p):
                        M = np.zeros((m, R))
                        for j in range(m):
                            for r in range(R):
                                M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        try:
                            w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W_new[k*n+i, :] = w
                        except:
                            W_new[k*n+i, :] = W[k*n+i, :]
                W = W_new
                
                # Solve for U
                U_new = np.zeros((n*m, R))
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, R))
                        for k in range(p):
                            for r in range(R):
                                M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        try:
                            u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U_new[i*m+j, :] = u
                        except:
                            U_new[i*m+j, :] = U[i*m+j, :]
                U = U_new
                
                # Solve for V
                V_new = np.zeros((m*p, R))
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, R))
                        for i in range(n):
                            for r in range(R):
                                M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        try:
                            v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V_new[j*p+k, :] = v
                        except:
                            V_new[j*p+k, :] = V[j*p+k, :]
                V = V_new
                
                # Check convergence
                if it % 20 == 0:
                    err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                err += (s - 1)**2
                    if err < 1e-6:
                        break
            
            # Round to half-integers
            U_hi = np.round(U * 2) / 2
            V_hi = np.round(V * 2) / 2
            W_hi = np.round(W * 2) / 2
            
            U_list = U_hi.tolist()
            V_list = V_hi.tolist()
            W_list = W_hi.tolist()
            
            if check_solution(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try structured ranks first (these are more likely to succeed with structure)
    for R in range(20, 101, 5):
        if time.time() > deadline:
            break
        result = try_structured_rank(R, max_restarts=3)
        if result is not None:
            return result
    
    # Fall back to unstructured search
    for R in range(25, 101, 5):
        if time.time() > deadline:
            break
        result = try_rank(R, max_restarts=5, max_iters=500)
        if result is not None:
            return result
    
    return schoolbook()
