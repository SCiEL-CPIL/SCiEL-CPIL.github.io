import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Precompute target tensor indices
    # T[u, v, w] = 1 where u=i*m+j, v=j*p+k, w=k*n+i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_indices.append((u, v, w))
                
    TARGET_TENSOR = np.zeros((size_u, size_v, size_w))
    for u, v, w in target_indices:
        TARGET_TENSOR[u, v, w] = 1.0

    # Schoolbook solution as fallback
    best_rank = n * m * p # 48
    U_school = np.zeros((size_u, best_rank))
    V_school = np.zeros((size_v, best_rank))
    W_school = np.zeros((size_w, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_solution = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    def check_and_snap(U, V, W):
        # Snap to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Verify exactness
        # T = sum_r U[:,r] V[:,r] W[:,r]
        T_check = np.zeros((size_u, size_v, size_w))
        for r in range(U.shape[1]):
            T_check += np.einsum('u,v,w->uvw', U_snap[:, r], V_snap[:, r], W_snap[:, r])
            
        if np.allclose(T_check, TARGET_TENSOR, atol=1e-9):
            return True, U_snap, V_snap, W_snap
        return False, None, None, None

    def solve_for_W(U, V, R):
        # Solve linear system for W given U, V
        # Equations: sum_r U[u,r] V[v,r] W[w,r] = T[u,v,w]
        # Group by w (which is k*n+i)
        # For a fixed w (i.e., fixed i, k), we have m equations (one for each j)
        # Unknowns are W[w, 0..R-1]
        
        W = np.zeros((size_w, R))
        
        # Iterate over all w
        for w in range(size_w):
            # w = k*n + i
            k = w // n
            i = w % n
            
            # Collect equations for this w
            # Equations correspond to j in 0..m-1
            # u = i*m + j
            # v = j*p + k
            # T[u,v,w] should be 1 for all j?
            # Yes, for a fixed (i,k), T[i*m+j, j*p+k, k*n+i] = 1 for all j.
            
            A = np.zeros((m, R))
            b = np.zeros(m)
            
            for j in range(m):
                u = i * m + j
                v = j * p + k
                # Coefficients for W[w, r] are U[u, r] * V[v, r]
                A[j, :] = U[u, :] * V[v, :]
                b[j] = TARGET_TENSOR[u, v, w] # Should be 1.0
            
            # Solve A W[w,:] = b
            # If R > m, system is underdetermined. Use least squares or minimum norm.
            # If R <= m, system is overdetermined or square. Check consistency.
            
            try:
                if R <= m:
                    # Overdetermined or square
                    # Check if solution exists
                    # Use least squares
                    W_w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    # Check if residuals are small
                    if np.linalg.norm(A @ W_w - b) > 1e-9:
                        return None
                else:
                    # Underdetermined. Find minimum norm solution?
                    # Or just check if a half-integer solution exists?
                    # Minimum norm solution is A.T (A A.T)^-1 b
                    # But we need half-integer.
                    # Let's find the minimum norm solution and check if it's close to half-integers.
                    W_w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    # Note: lstsq returns minimum norm solution for underdetermined systems?
                    # Actually, for underdetermined, it finds a solution with minimum norm.
                    # We might need to search for other solutions.
                    # For now, let's just check this one.
            except np.linalg.LinAlgError:
                return None
            
            W[w, :] = W_w
            
        # Snap W to half-integers and verify
        W_snap = np.round(W * 2) / 2
        
        # Verify full tensor
        T_check = np.zeros((size_u, size_v, size_w))
        for r in range(R):
            T_check += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W_snap[:, r])
            
        if np.allclose(T_check, TARGET_TENSOR, atol=1e-9):
            return W_snap
        else:
            return None

    # Search for lower ranks
    # We try ranks from 47 down to 40
    # We use random U, V with sparse 0/1 or 0/0.5/1 entries
    
    np.random.seed(42)
    
    for R in range(47, 39, -1):
        if time.time() > deadline - 5:
            break
            
        num_tries = 100
        for seed in range(num_tries):
            if time.time() > deadline - 5:
                break
            
            # Generate random U, V
            # Strategy: Start with schoolbook and zero out some columns?
            # Or random sparse matrices.
            # Let's try random matrices with entries in {0, 1}
            # To ensure some structure, we can use the schoolbook U, V and perturb.
            
            U = np.zeros((size_u, R))
            V = np.zeros((size_v, R))
            
            # Initialize with random 0/1
            # Each row of U should have some non-zeros.
            # Let's try: for each r, pick a random i,j for U and random j,k for V?
            # This is too restrictive.
            
            # Better: Random 0/1 matrices with density 0.5
            U = (np.random.rand(size_u, R) < 0.5).astype(float)
            V = (np.random.rand(size_v, R) < 0.5).astype(float)
            
            # Solve for W
            W_sol = solve_for_W(U, V, R)
            
            if W_sol is not None:
                # Found a solution!
                best_rank = R
                best_solution = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_sol.tolist()
                }
                break # Move to lower R
                
    return best_solution

# EVOLVE-BLOCK-END
