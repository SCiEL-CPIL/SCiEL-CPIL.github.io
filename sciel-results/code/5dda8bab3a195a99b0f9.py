"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = 5, 5, 5
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
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
    
    def verify(U, V, W, rank):
        """Verify exact decomposition. Returns max error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def round_to_half(x):
        """Round to nearest half-integer."""
        return round(x * 2) / 2.0
    
    def verify_rounded(U_r, V_r, W_r, rank):
        """Verify rounded decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_r[i*m+j][c] * V_r[j*p+k][c] * W_r[k*n+i][c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def try_rank(rank, max_restarts=8, max_iters=150):
        """Try to find exact rank decomposition via alternating minimization with structured init."""
        if time.time() > deadline - 20:
            return None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                return None
            
            # Structured initialization: use polynomial basis for better conditioning
            np.random.seed(rank * 1000 + restart * 7 + 42)
            
            # Initialize with a mix of structured and random
            if restart < 3:
                # Polynomial-style structured init
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for c in range(rank):
                    # Assign each column a "pattern"
                    i_pat = c % n
                    j_pat = (c // n) % m
                    k_pat = (c // (n*m)) % p
                    for i in range(n):
                        for j in range(m):
                            U[i*m+j, c] = (1.0 if i == i_pat else 0.0) + 0.1 * np.random.randn()
                    for j in range(m):
                        for k in range(p):
                            V[j*p+k, c] = (1.0 if j == j_pat else 0.0) + 0.1 * np.random.randn()
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i, c] = (1.0 if k == k_pat else 0.0) + 0.1 * np.random.randn()
            else:
                # Random init with small integer/half-integer values
                U = np.random.choice([0, 0.5, 1, -0.5, -1], size=(n*m, rank))
                V = np.random.choice([0, 0.5, 1, -0.5, -1], size=(m*p, rank))
                W = np.random.choice([0, 0.5, 1, -0.5, -1], size=(p*n, rank))
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                # Update U: for each (i,j), solve linear system
                # U[i*m+j, :] @ B.T = ones(p) where B[c,k] = V[j*p+k,c]*W[k*n+i,c]
                for i in range(n):
                    for j in range(m):
                        B = np.array([[V[j*p+k, c] * W[k*n+i, c] for k in range(p)] for c in range(rank)])
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), solve linear system
                # V[j*p+k, :] @ A.T = ones(n) where A[c,i] = U[i*m+j,c]*W[k*n+i,c]
                for j in range(m):
                    for k in range(p):
                        A = np.array([[U[i*m+j, c] * W[k*n+i, c] for i in range(n)] for c in range(rank)])
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i), solve linear system
                # W[k*n+i, :] @ C.T = ones(m) where C[c,j] = U[i*m+j,c]*V[j*p+k,c]
                for k in range(p):
                    for i in range(n):
                        C = np.array([[U[i*m+j, c] * V[j*p+k, c] for j in range(m)] for c in range(rank)])
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence every 5 iterations
                if it % 5 == 4 or it == max_iters - 1:
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                                err = abs(val - 1.0)
                                if err > max_err:
                                    max_err = err
                    
                    if max_err < 1e-6:
                        # Round to half-integers and verify
                        U_r = np.array([[round_to_half(x) for x in U[i]] for i in range(n*m)])
                        V_r = np.array([[round_to_half(x) for x in V[j]] for j in range(m*p)])
                        W_r = np.array([[round_to_half(x) for x in W[k]] for k in range(p*n)])
                        err_r = verify_rounded(U_r, V_r, W_r, rank)
                        if err_r < 1e-8:
                            return {
                                "U": U_r.tolist(),
                                "V": V_r.tolist(),
                                "W": W_r.tolist()
                            }
                    
                    # If close but not exact, do a polish pass
                    if max_err < 1e-3:
                        # Run additional iterations with tighter tolerance
                        for polish_it in range(30):
                            if time.time() > deadline - 3:
                                break
                            for i in range(n):
                                for j in range(m):
                                    B = np.array([[V[j*p+k, c] * W[k*n+i, c] for k in range(p)] for c in range(rank)])
                                    try:
                                        U[i*m+j, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                                    except:
                                        pass
                            for j in range(m):
                                for k in range(p):
                                    A = np.array([[U[i*m+j, c] * W[k*n+i, c] for i in range(n)] for c in range(rank)])
                                    try:
                                        V[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                                    except:
                                        pass
                            for k in range(p):
                                for i in range(n):
                                    C = np.array([[U[i*m+j, c] * V[j*p+k, c] for j in range(m)] for c in range(rank)])
                                    try:
                                        W[k*n+i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                                    except:
                                        pass
                            max_err = 0
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                                        err = abs(val - 1.0)
                                        if err > max_err:
                                            max_err = err
                            if max_err < 1e-8:
                                U_r = np.array([[round_to_half(x) for x in U[i]] for i in range(n*m)])
                                V_r = np.array([[round_to_half(x) for x in V[j]] for j in range(m*p)])
                                W_r = np.array([[round_to_half(x) for x in W[k]] for k in range(p*n)])
                                err_r = verify_rounded(U_r, V_r, W_r, rank)
                                if err_r < 1e-8:
                                    return {
                                        "U": U_r.tolist(),
                                        "V": V_r.tolist(),
                                        "W": W_r.tolist()
                                    }
                        break
        
        return None
    
    # Try ranks in a strategic order: known bounds suggest ~49-55 for 5x5x5
    # Start with lower ranks that are more likely to succeed
    rank_order = [49, 50, 51, 52, 53, 54, 55, 56, 58, 60, 65, 70, 80, 100, 125]
    
    for rank in rank_order:
        if time.time() > deadline - 25:
            break
        if rank >= 125:
            # Fall back to schoolbook
            return schoolbook()
        
        # Allocate more restarts for lower ranks
        restarts = 6 if rank <= 55 else 4
        iters = 120 if rank <= 55 else 80
        
        result = try_rank(rank, max_restarts=restarts, max_iters=iters)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
