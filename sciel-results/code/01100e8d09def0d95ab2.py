"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140  # safety margin before 150s kill

    # Strategy:
    # 1. Try to find rank < 28 via structured search exploiting the block structure
    # 2. Fall back to rank 28 Strassen decomposition
    
    # The tensor: T[i,j,k] = delta (all ones) for i in 0..1, j in 0..3, k in 0..3
    # We need: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1 for all (i,j,k)
    
    # Approach: Use the fact that n=2 is small.
    # For each (j,k) pair, the equation is:
    #   sum_t U[0,j,t]*V[j,k,t]*W[k,0,t] = 1
    #   sum_t U[1,j,t]*V[j,k,t]*W[k,1,t] = 1
    # 
    # Let's try a structured ansatz:
    # Partition j into {0,1} and {2,3}, k into {0,1} and {2,3}
    # This gives 4 blocks of 2x2x2, each needing rank 7 (Strassen)
    # Total: 28
    #
    # To go below 28, we need cross-block sharing.
    # Let's try rank 26 with a parameterized family and solve linear constraints.
    
    # For a given rank r, we have r "terms". Each term t has:
    # U_t: vector in R^{n*m} = R^8
    # V_t: vector in R^{m*p} = R^{16}
    # W_t: vector in R^{p*n} = R^8
    #
    # The constraint is: for all (i,j,k): sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    # This is 2*4*4 = 32 scalar equations.
    #
    # Total unknowns: r*(8+16+8) = 32r
    # For r=26: 832 unknowns, 32 equations -> very underdetermined
    # But the equations are nonlinear (trilinear).
    #
    # Alternative: Fix U and W, solve for V linearly.
    # Or use the "slice rank" approach.
    
    # Let me try a different structured approach:
    # Since n=2, write:
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    #
    # The 2x4x4 tensor can be seen as 4 copies of the 2x4x1 "row" tensor
    # (one for each k), each of which has rank 2 (since it's just outer product structure).
    # But they share the A factors...
    
    # Let's try: for rank r, set up the trilinear system and use scipy to solve.
    # Actually, let's try a more direct approach:
    
    # Known result: The bilinear complexity of 2x4x4 matrix multiplication is 26.
    # Let me try to construct it explicitly.
    
    # One approach: Use the fact that 2x4x4 can be embedded in 4x4x4.
    # 4x4x4 has rank 49 (Strassen). But that's too high.
    
    # Let's try a computational approach:
    # For rank r, we can fix the structure of U and W (using the block structure)
    # and solve for V.
    
    # Actually, let me try a simpler approach:
    # The current rank-28 solution uses 4 independent 2x2x2 Strassen decompositions.
    # Each uses 7 terms. Total 28.
    # 
    # Can we merge any terms across blocks?
    # Block (j_off, k_off) term t:
    #   U[i, j+j_off, t] = u_i
    #   V[j+j_off, k+k_off, t] = v_jk  
    #   W[k+k_off, i, t] = w_k
    #
    # Two terms from different blocks can be merged if they have the same
    # (u, v, w) pattern up to scaling... but they operate on different (j,k) ranges,
    # so they can't directly merge.
    #
    # However, we can use "cross-block" terms that contribute to multiple blocks.
    #
    # Let me try: rank 27 or 26 with a numerical search.
    
    def try_rank(r, max_time=30):
        """Try to find a rank-r decomposition."""
        t_start = time.time()
        
        # Use a structured initialization based on the rank-28 solution
        # then try to merge/remove terms
        
        # Start with the rank-28 solution
        U28, V28, W28 = _strassen_28(n, m, p)
        
        # Try to find a rank-r sub-decomposition by selecting r of the 28 terms
        # and adjusting coefficients
        # This won't work directly since we need all 28 for exactness
        
        # Alternative: parameterize and use scipy.optimize
        # For small r, use random restarts with gradient descent
        
        for attempt in range(20):
            if time.time() - t_start > max_time or time.time() > deadline - 20:
                break
            
            # Random initialization with structure
            # Use the block structure: each term contributes to specific (j,k) blocks
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Initialize with random values scaled appropriately
            np.random.seed(42 + attempt)
            U[:] = np.random.randn(n*m, r) * 0.5
            V[:] = np.random.randn(m*p, r) * 0.5
            W[:] = np.random.randn(p*n, r) * 0.5
            
            # Use scipy's least_squares to minimize the residual
            from scipy.optimize import least_squares
            
            def residual(params):
                U_p = params[:n*m*r].reshape(n*m, r)
                V_p = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_p = params[n*m*r + m*p*r:].reshape(p*n, r)
                
                res = np.zeros(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for t in range(r):
                                s += U_p[i*m+j, t] * V_p[j*p+k, t] * W_p[k*n+i, t]
                            res[idx] = s - 1.0
                            idx += 1
                return res
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(residual, x0, method='lm', max_nfev=5000,
                                       xtol=1e-12, ftol=1e-12, gtol=1e-12)
                
                if result.cost < 1e-20:
                    U_sol = result.x[:n*m*r].reshape(n*m, r)
                    V_sol = result.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                    W_sol = result.x[n*m*r + m*p*r:].reshape(p*n, r)
                    
                    # Round to nearest half-integer and verify
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    # Verify
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for t in range(r):
                                    s += U_r[i*m+j, t] * V_r[j*p+k, t] * W_r[k*n+i, t]
                                if abs(s - 1.0) > 1e-9:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    
                    if ok:
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()
            except Exception:
                continue
        
        return None
    
    # Try ranks 26 and 27
    for r in [26, 27]:
        if time.time() > deadline - 40:
            break
        result = try_rank(r, max_time=35)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: rank 28
    U28, V28, W28 = _strassen_28(n, m, p)
    return {"U": U28, "V": V28, "W": W28}


def _strassen_28(n, m, p):
    """Construct the rank-28 Strassen-based decomposition."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    blocks = [
        {'j_offset': 0, 'k_offset': 0},
        {'j_offset': 2, 'k_offset': 0},
        {'j_offset': 0, 'k_offset': 2},
        {'j_offset': 2, 'k_offset': 2}
    ]
    
    col = 0
    for block in blocks:
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j][col] += coeff
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k][col] += coeff
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i][col] += coeff
            col += 1
    
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    
    return U, V, W
