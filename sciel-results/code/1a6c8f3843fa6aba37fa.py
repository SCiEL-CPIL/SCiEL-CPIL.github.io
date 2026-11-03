import time
import numpy as np
from itertools import product

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 125
    
    # Precompute index arrays for vectorized operations
    ij_idx = np.array([(i*m+j) for i in range(n) for j in range(m)])
    jk_idx = np.array([(j*p+k) for j in range(m) for k in range(p)])
    ki_idx = np.array([(k*n+i) for k in range(p) for i in range(n)])
    
    # Build target tensor: T[i,j,k] = 1
    # Reshape to matrices for each mode
    # Mode 1: (i*m+j) x (j*p+k) -> for each c, U[:,c] @ V[:,c].T should help
    # Actually, let's use the tensor formulation:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    def verify_fast(U, V, W):
        """Fast vectorized verification using einsum."""
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape to proper dimensions
        U_mat = U.reshape(n, m, -1)  # (n, m, rank)
        V_mat = V.reshape(m, p, -1)  # (m, p, rank)
        W_mat = W.reshape(p, n, -1)  # (p, n, rank)
        
        # Compute all (i,j,k) values at once
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Use einsum: 'ijc,jkc,kic->ijk'
        T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
        max_err = np.max(np.abs(T - 1.0))
        return float(max_err)
    
    def try_algebraic_construction(rank, seed=0):
        """
        Try to construct a rank-r decomposition using algebraic structure.
        For 5x5x5, we can try to exploit the fact that 5 = 4+1 or other decompositions.
        
        Strategy: Use the identity that for small matrices, we can sometimes find
        decompositions by exploiting additive/multiplicative structure.
        """
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        # For rank near 49, try a structured approach:
        # Split indices into groups and use block structure
        # For 5x5x5, try rank = 49 = 7^2 approach
        
        # Try a direct construction based on the observation that
        # matrix multiplication can be decomposed using fewer terms
        # by exploiting linearity in certain indices.
        
        # Method 1: Try to find a decomposition where each "tensor slice" 
        # has low rank structure
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Try a structured initialization based on modular arithmetic
        # For rank r, assign each term to cover a subset of (i,j,k) triples
        
        # Use a greedy approach: for each (i,j,k), ensure it's covered
        # by at least one term c where U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        # This is a set cover problem, but we can try a simple heuristic
        # First, initialize with random values
        U = np.random.randn(n*m, rank) * 0.3
        V = np.random.randn(m*p, rank) * 0.3
        W = np.random.randn(p*n, rank) * 0.3
        
        # Then try to optimize using gradient descent on a smooth loss
        # Loss = sum_{i,j,k} (1 - sum_c U[i,j,c]*V[j,k,c]*W[k,i,c])^2
        # + regularization to encourage sparse/structured solutions
        
        lr = 0.01
        for it in range(200):
            if time.time() > deadline - 5:
                return None
            
            # Compute gradients
            U_mat = U.reshape(n, m, rank)
            V_mat = V.reshape(m, p, rank)
            W_mat = W.reshape(p, n, rank)
            
            # T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
            T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
            err = T - 1.0
            
            # Gradient w.r.t. U[i,j,c] = err[i,j,k] * V[j,k,c] * W[k,i,c]
            # Sum over k
            grad_U = np.einsum('ijk,jkc,kic->ijc', err, V_mat, W_mat)
            grad_V = np.einsum('ijk,ijc,kic->jkc', err, U_mat, W_mat)
            grad_W = np.einsum('ijk,ijc,jkc->kic', err, U_mat, V_mat)
            
            # Update
            U -= lr * grad_U.reshape(n*m, rank)
            V -= lr * grad_V.reshape(m*p, rank)
            W -= lr * grad_W.reshape(p*n, rank)
            
            # Check convergence
            max_err = np.max(np.abs(T))
            if max_err < 1e-8:
                # Round to half-integers and verify
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify_fast(U_r, V_r) < 1e-8:
                    return U_r, V_r, W_r
        
        return None
    
    def try_decomposition(rank, max_iters=50, seed=0, init_U=None, init_V=None, init_W=None, tol=1e-8):
        """Vectorized alternating minimization for rank decomposition."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        # Initialize with small random values or provided initialization
        if init_U is not None and init_V is not None and init_W is not None:
            U = init_U.copy()
            V = init_V.copy()
            W = init_W.copy()
        else:
            # Use a better initialization: mix of random and structured
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Add some structure: ensure each row has at least one non-zero
            for i in range(n*m):
                if np.all(np.abs(U[i]) < 0.1):
                    U[i, np.random.randint(rank)] = 1.0
            for i in range(m*p):
                if np.all(np.abs(V[i]) < 0.1):
                    V[i, np.random.randint(rank)] = 1.0
            for i in range(p*n):
                if np.all(np.abs(W[i]) < 0.1):
                    W[i, np.random.randint(rank)] = 1.0
        
        prev_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_max_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U: for each (i,j), solve U[i*m+j, :] @ B = ones(p)
            # where B[c, k] = V[j*p+k, c] * W[k*n+i, c]
            for i in range(n):
                for j in range(m):
                    # Build B: (rank, p)
                    # B[c, k] = V[j*p+k, c] * W[k*n+i, c]
                    B = np.zeros((rank, p))
                    for k in range(p):
                        B[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve V[j*p+k, :] @ A = ones(n)
            # where A[c, i] = U[i*m+j, c] * W[k*n+i, c]
            for j in range(m):
                for k in range(p):
                    A = np.zeros((rank, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve W[k*n+i, :] @ C = ones(m)
            # where C[c, j] = U[i*m+j, c] * V[j*p+k, c]
            for k in range(p):
                for i in range(n):
                    C = np.zeros((rank, m))
                    for j in range(m):
                        C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Compute error
            max_err = verify_fast(U, V, W)
            
            # Track best
            if max_err < best_max_err:
                best_max_err = max_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if max_err < tol:
                return U, V, W, max_err
            
            if max_err > prev_err * 1.1 and it > 10:
                # Diverging, break early
                break
            prev_err = max_err
        
        # If we didn't converge but got close, return best found
        if best_max_err < 0.1:
            return best_U, best_V, best_W, best_max_err
        
        return None
    
    def polish_decomposition(U, V, W, rank, max_iters=100):
        """High-precision polish pass with tighter tolerance."""
        if time.time() > deadline - 5:
            return None
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    B = np.zeros((rank, p))
                    for k in range(p):
                        B[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((rank, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((rank, m))
                    for j in range(m):
                        C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            max_err = verify_fast(U, V, W)
            
            if max_err < 1e-12:
                break
            
            if max_err > prev_err * 1.05 and it > 30:
                break
            prev_err = max_err
        
        return U, V, W, verify_fast(U, V, W)
    
    def try_direct_construction(rank):
        """
        Try to directly construct a rank-r decomposition using known results.
        For 5x5x5 matrix multiplication, the minimal rank is known to be 49
        (Strassen's algorithm gives 49 for 5x5x5).
        
        We'll try to find the explicit decomposition.
        """
        if time.time() > deadline - 10:
            return None
        
        # For 5x5x5, Strassen's algorithm achieves rank 49
        # The decomposition is complex, so we'll use a search-based approach
        # with strong initialization
        
        best_result = None
        best_err = float('inf')
        
        # Try multiple random initializations with the alternating minimization
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            
            result = try_decomposition(rank, max_iters=60, seed=seed, tol=1e-7)
            if result is not None:
                U, V, W, err = result
                if err < best_err:
                    best_err = err
                    best_result = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-7:
                    break
        
        if best_result is not None:
            U, V, W = best_result
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            max_err = verify_fast(U_r, V_r, W_r)
            if max_err < 1e-8:
                return U_r, V_r, W_r, max_err
        
        return None
    
    # Schoolbook fallback
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = (U_school, V_school, W_school)
    
    # Focus on ranks near the known minimum (49 for 5x5x5)
    # Try a range of ranks, prioritizing those near 49
    candidate_ranks = list(range(45, 55)) + list(range(35, 45)) + list(range(25, 35))
    # Remove duplicates and sort
    candidate_ranks = sorted(set(candidate_ranks))
    
    for rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        # Try direct construction first for this rank
        direct_result = try_direct_construction(rank)
        if direct_result is not None:
            U_r, V_r, W_r, err = direct_result
            best_rank = rank
            best_result = (U_r, V_r, W_r)
            # Try to go lower
            if best_rank <= 45:
                break
            continue
        
        # Try alternating minimization with multiple seeds
        for seed in range(10):
            if time.time() > deadline - 15:
                break
            
            result = try_decomposition(rank, max_iters=50, seed=seed + rank*100, tol=1e-6)
            if result is not None:
                U, V, W, err = result
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                # Verify rounded version
                max_err = verify_fast(U_r, V_r, W_r)
                if max_err < 1e-8:
                    best_rank = rank
                    best_result = (U_r, V_r, W_r)
                    break  # Found valid decomposition for this rank
        
        if best_rank < rank_schoolbook:
            # Found improvement, try to go lower
            if best_rank <= 45:
                break
    
    # If we found a valid decomposition, try to polish it
    if best_rank < rank_schoolbook:
        U, V, W = best_result
        polished = polish_decomposition(U.copy(), V.copy(), W.copy(), best_rank)
        if polished is not None:
            U_p, V_p, W_p, err_p = polished
            U_pr = np.round(U_p * 2) / 2
            V_pr = np.round(V_p * 2) / 2
            W_pr = np.round(W_p * 2) / 2
            max_err = verify_fast(U_pr, V_pr, W_pr)
            if max_err < 1e-8:
                best_result = (U_pr, V_pr, W_pr)
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
