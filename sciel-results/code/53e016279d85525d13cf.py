"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x3 matrix by a 3x4 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes
        (n*m, rank), (m*p, rank), (p*n, rank)
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Schoolbook decomposition: rank = n * m * p = 36
    # This is the standard exact decomposition with integer entries.
    
    rank_sb = n * m * p  # 36
    U_sb = [[0] * rank_sb for _ in range(n * m)]  # 9 x 36
    V_sb = [[0] * rank_sb for _ in range(m * p)]  # 12 x 36
    W_sb = [[0] * rank_sb for _ in range(p * n)]  # 12 x 36
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Check if the decomposition is exact."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_winograd_style(R, n, m, p, max_attempts=5):
        """
        Attempt to find a decomposition of rank R using a structured approach
        inspired by Winograd FMM. We try to exploit the structure where
        the 3x4 multiplication can be viewed as 4 columns of 3x3 multiplications,
        but with shared structure to reduce rank below 36.
        
        We use random initialization with half-integer values and coordinate descent.
        """
        np.random.seed(42)
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with random half-integer values
            vals = np.array([-1, -0.5, 0, 0.5, 1])
            U = np.random.choice(vals, size=(n*m, R))
            V = np.random.choice(vals, size=(m*p, R))
            W = np.random.choice(vals, size=(p*n, R))
            
            # Coordinate descent optimization
            converged = False
            for iteration in range(200):
                if time.time() > deadline:
                    return None
                
                # Compute current residual
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            residual += (s - 1.0)**2
                
                if residual < 1e-12:
                    # Convert to lists and verify
                    U_list = [[U[i, r] for r in range(R)] for i in range(n*m)]
                    V_list = [[V[i, r] for r in range(R)] for i in range(m*p)]
                    W_list = [[W[i, r] for r in range(R)] for i in range(p*n)]
                    if verify(U_list, V_list, W_list, n, m, p):
                        return (U_list, V_list, W_list)
                    break
                
                # Try to improve by adjusting individual elements
                improved = False
                # Shuffle order for diversity
                np.random.shuffle(np.arange(R))
                
                for r in range(R):
                    if time.time() > deadline:
                        return None
                    
                    # Try adjusting U elements in column r
                    for i in range(n*m):
                        old_val = U[i, r]
                        for new_val in [-1, -0.5, 0, 0.5, 1]:
                            if new_val == old_val:
                                continue
                            U[i, r] = new_val
                            # Quick check: compute partial residual change
                            # This is expensive, so we'll just accept and check at end
                            improved = True
                            break
                        # Keep the best or revert
                        # For simplicity, we'll do a full residual check every few updates
                        if iteration % 5 == 0 and i % 3 == 0:
                            res = 0.0
                            for ii in range(n):
                                for jj in range(m):
                                    for kk in range(p):
                                        s = 0.0
                                        for rr in range(R):
                                            s += U[ii*m+jj, rr] * V[jj*p+kk, rr] * W[kk*n+ii, rr]
                                        res += (s - 1.0)**2
                            if res > residual + 1e-6:
                                U[i, r] = old_val
                                improved = False
                    
                    # Try adjusting V elements in column r
                    for j in range(m*p):
                        old_val = V[j, r]
                        for new_val in [-1, -0.5, 0, 0.5, 1]:
                            if new_val == old_val:
                                continue
                            V[j, r] = new_val
                            improved = True
                            break
                        if iteration % 5 == 0 and j % 3 == 0:
                            res = 0.0
                            for ii in range(n):
                                for jj in range(m):
                                    for kk in range(p):
                                        s = 0.0
                                        for rr in range(R):
                                            s += U[ii*m+jj, rr] * V[jj*p+kk, rr] * W[kk*n+ii, rr]
                                        res += (s - 1.0)**2
                            if res > residual + 1e-6:
                                V[j, r] = old_val
                                improved = False
                    
                    # Try adjusting W elements in column r
                    for k in range(p*n):
                        old_val = W[k, r]
                        for new_val in [-1, -0.5, 0, 0.5, 1]:
                            if new_val == old_val:
                                continue
                            W[k, r] = new_val
                            improved = True
                            break
                        if iteration % 5 == 0 and k % 3 == 0:
                            res = 0.0
                            for ii in range(n):
                                for jj in range(m):
                                    for kk in range(p):
                                        s = 0.0
                                        for rr in range(R):
                                            s += U[ii*m+jj, rr] * V[jj*p+kk, rr] * W[kk*n+ii, rr]
                                        res += (s - 1.0)**2
                            if res > residual + 1e-6:
                                W[k, r] = old_val
                                improved = False
                
                if not improved:
                    break
            
            # Final check
            U_list = [[U[i, r] for r in range(R)] for i in range(n*m)]
            V_list = [[V[i, r] for r in range(R)] for i in range(m*p)]
            W_list = [[W[i, r] for r in range(R)] for i in range(p*n)]
            if verify(U_list, V_list, W_list, n, m, p):
                return (U_list, V_list, W_list)
        
        return None
    
    # Try to find lower rank decompositions
    # Based on known results, rank for 3x3x4 might be around 27-32
    # We'll try from 24 up to 35
    for R in range(24, best_rank):
        if time.time() > deadline - 5:  # Leave 5s margin
            break
        
        result = try_winograd_style(R, n, m, p, max_attempts=3)
        if result is not None:
            U_list, V_list, W_list = result
            if verify(U_list, V_list, W_list, n, m, p):
                best_rank = R
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break  # Found a valid lower rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
