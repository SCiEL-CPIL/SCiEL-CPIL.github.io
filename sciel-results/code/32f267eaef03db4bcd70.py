<<<<<<< SEARCH
"""Matrix multiplication 2x4x6 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 4, 6]

# Deadline guard
DEADLINE = 120.0

def construct():
=======
"""Matrix multiplication 2x4x6 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# Deadline guard
DEADLINE = 115.0  # Safety margin

def _strassen_decomposition():
    """Return the rank-42 Strassen-based decomposition."""
    n, m, p = 2, 4, 6
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    strassen_defs = [
        ({'X00': 1, 'X11': 1}, {'Y00': 1, 'Y11': 1}),
        ({'X10': 1, 'X11': 1}, {'Y00': 1}),
        ({'X00': 1}, {'Y01': 1, 'Y11': -1}),
        ({'X11': 1}, {'Y10': 1, 'Y00': -1}),
        ({'X00': 1, 'X01': 1}, {'Y11': 1}),
        ({'X10': 1, 'X00': -1}, {'Y00': 1, 'Y01': 1}),
        ({'X01': 1, 'X11': -1}, {'Y10': 1, 'Y11': 1}),
    ]
    
    output_coeffs = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],
        [(2, 1), (4, 1)],
        [(1, 1), (3, 1)],
        [(0, 1), (1, -1), (2, 1), (5, 1)],
    ]
    
    for g in range(3):
        for mi, (x_coeff, y_coeff) in enumerate(strassen_defs):
            A_idx_map = {'X00': 0, 'X01': 1, 'X10': 4, 'X11': 5}
            B_idx_map = {
                'Y00': 0 * p + 2 * g,
                'Y01': 0 * p + 2 * g + 1,
                'Y10': 1 * p + 2 * g,
                'Y11': 1 * p + 2 * g + 1,
            }
            
            for x_name, x_val in x_coeff.items():
                U[A_idx_map[x_name]][col] += x_val
            for y_name, y_val in y_coeff.items():
                V[B_idx_map[y_name]][col] += y_val
            
            w_indices = [4 * g, 4 * g + 2, 4 * g + 1, 4 * g + 3]
            for zi, w_idx in enumerate(w_indices):
                for mult_idx, coeff in output_coeffs[zi]:
                    if mult_idx == mi:
                        W[w_idx][col] += coeff
            
            col += 1
        
        for mi, (x_coeff, y_coeff) in enumerate(strassen_defs):
            A_idx_map = {'X00': 2, 'X01': 3, 'X10': 6, 'X11': 7}
            B_idx_map = {
                'Y00': 2 * p + 2 * g,
                'Y01': 2 * p + 2 * g + 1,
                'Y10': 3 * p + 2 * g,
                'Y11': 3 * p + 2 * g + 1,
            }
            
            for x_name, x_val in x_coeff.items():
                U[A_idx_map[x_name]][col] += x_val
            for y_name, y_val in y_coeff.items():
                V[B_idx_map[y_name]][col] += y_val
            
            w_indices = [4 * g, 4 * g + 2, 4 * g + 1, 4 * g + 3]
            for zi, w_idx in enumerate(w_indices):
                for mult_idx, coeff in output_coeffs[zi]:
                    if mult_idx == mi:
                        W[w_idx][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = delta[i,k] should equal sum_r U[i,r]*V[j,r]*W[k,r]
    # where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    
    # Build the target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i,j] * B[j,k] contributes to C[i,k]
                # W index is k*n + i
                T_target[i*m+j, j*p+k, k*n+i] = 1
    
    # Compute the decomposed tensor
    # T_decomp[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # Efficient: for each r, outer product of U[:,r], V[:,r], W[:,r]
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(len(U[0])):
        T_decomp += np.einsum('i,j,k->ijk', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    return np.allclose(T_decomp, T_target, atol=1e-10)

def _try_lower_rank_search(start_rank, max_rank, n, m, p, deadline):
    """Search for a lower-rank decomposition using random initialization and refinement."""
    best_result = None
    
    for rank in range(start_rank, max_rank + 1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Random sparse initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Scale down
            U *= 0.1
            V *= 0.1
            W *= 0.1
            
            # Iterate: minimize ||sum_r U[:,r]⊗V[:,r]⊗W[:,r] - T||^2
            # Using alternating least squares
            T_target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i*m+j, j*p+k, k*n+i] = 1
            
            converged = False
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Update U: for each r, solve least squares
                for r in range(rank):
                    # Fix V[:,r] and W[:,r], update U[:,r]
                    # T_decomp[i,j,k] = U[i,r]*V[j,r]*W[k,r] + (other terms)
                    # For fixed r, the contribution is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                    # We want: sum_r' U[i,r']*V[j,r']*W[k,r'] ≈ T[i,j,k]
                    # The residual for updating U[:,r] is:
                    # R[i,j,k] = T[i,j,k] - sum_{r'≠r} U[i,r']*V[j,r']*W[k,r']
                    # Then U[:,r] should minimize sum_{j,k} (R[i,j,k] - U[i,r]*V[j,r]*W[k,r])^2
                    
                    # Compute residual
                    R = T_target.copy()
                    for r2 in range(rank):
                        if r2 != r:
                            R -= np.einsum('i,j,k->ijk', U[:,r2], V[:,r2], W[:,r2])
                    
                    # Now solve: minimize sum_{j,k} (R[i,j,k] - U[i,r]*V[j,r]*W[k,r])^2
                    # This is a linear system for U[:,r]
                    # Let M[i, (j,k)] = V[j,r]*W[k,r]
                    # Then minimize ||R_flat - U[:,r] @ M||^2
                    # Solution: U[:,r] = R_flat^T @ M @ (M^T @ M)^{-1}
                    
                    M = np.outer(V[:,r], W[:,r]).reshape(m*p, p*n)
                    # R_flat[i, (j,k)] = R[i,j,k]
                    R_flat = R.reshape(n*m, m*p*p*n)
                    
                    # Solve (M^T M) U[:,r] = M^T R_flat
                    MTM = M.T @ M
                    MTR = M.T @ R_flat.T
                    try:
                        U[:,r] = np.linalg.solve(MTM + 1e-8*np.eye(rank), MTR.T).T
                    except:
                        pass
                
                # Update V similarly
                for r in range(rank):
                    R = T_target.copy()
                    for r2 in range(rank):
                        if r2 != r:
                            R -= np.einsum('i,j,k->ijk', U[:,r2], V[:,r2], W[:,r2])
                    
                    # For V[:,r]: minimize sum_{i,k} (R[i,j,k] - U[i,r]*V[j,r]*W[k,r])^2
                    # M[j, (i,k)] = U[i,r]*W[k,r]
                    M = np.outer(U[:,r], W[:,r]).reshape(n*m, p*n)
                    R_flat = R.transpose(1,0,2).reshape(m*p, n*m*p*n)
                    
                    MTM = M.T @ M
                    MTR = M.T @ R_flat.T
                    try:
                        V[:,r] = np.linalg.solve(MTM + 1e-8*np.eye(rank), MTR.T).T
                    except:
                        pass
                
                # Update W similarly
                for r in range(rank):
                    R = T_target.copy()
                    for r2 in range(rank):
                        if r2 != r:
                            R -= np.einsum('i,j,k->ijk', U[:,r2], V[:,r2], W[:,r2])
                    
                    # For W[:,r]: minimize sum_{i,j} (R[i,j,k] - U[i,r]*V[j,r]*W[k,r])^2
                    # M[k, (i,j)] = U[i,r]*V[j,r]
                    M = np.outer(U[:,r], V[:,r]).reshape(n*m, m*p)
                    R_flat = R.transpose(2,0,1).reshape(p*n, n*m*m*p)
                    
                    MTM = M.T @ M
                    MTR = M.T @ R_flat.T
                    try:
                        W[:,r] = np.linalg.solve(MTM + 1e-8*np.eye(rank), MTR.T).T
                    except:
                        pass
                
                # Check convergence
                T_decomp = np.zeros((n*m, m*p, p*n))
                for r in range(rank):
                    T_decomp += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
                err = np.linalg.norm(T_decomp - T_target)
                if err < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Convert to exact fractions if possible
                U_list = [[int(round(U[i,r]*2))/2 for r in range(rank)] for i in range(n*m)]
                V_list = [[int(round(V[i,r]*2))/2 for r in range(rank)] for i in range(m*p)]
                W_list = [[int(round(W[i,r]*2))/2 for r in range(rank)] for i in range(p*n)]
                
                # Verify
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    return best_result
                elif _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    return best_result
        
        # If we found a valid decomposition at this rank, return it
        if best_result is not None:
            return best_result
    
    return best_result

def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x6 matrix.
    """
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    # First, get the baseline rank-42 solution
    baseline = _strassen_decomposition()
    
    # Try to find a lower-rank solution
    # Theoretical lower bound is at least max(n*m, m*p, p*n) = 24
    # Known results suggest rank might be around 28-35 for this problem
    search_start = 28
    search_end = 41
    deadline = start_time + DEADLINE - 5  # Leave 5 seconds for verification
    
    found = _try_lower_rank_search(search_start, search_end, n, m, p, deadline)
    
    if found is not None:
        return found
    
    # Fall back to the rank-42 solution
    return baseline
>>>>>>> REPLACE

