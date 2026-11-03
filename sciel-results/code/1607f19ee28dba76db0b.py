import numpy as np
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T is defined by T[i, j, k] = 1 for all i, j, k
    # The decomposition condition is sum_r U[i, j, r] V[j, k, r] W[i, k, r] = 1
    
    # We try to find a decomposition of rank r
    # We iterate r from a low bound to the schoolbook rank
    # Lower bound from flattening is max(n*m, m*p, p*n) = 12
    # But tensor rank is higher. Let's try from 15 upwards.
    
    max_rank = n * m * p
    found_solution = None
    
    # Helper function to verify a solution
    def verify(U, V, W, r):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Helper to check if values are close to half-integers
    def is_half_integer(arr, tol=1e-6):
        doubled = 2 * arr
        rounded = np.round(doubled)
        if np.allclose(doubled, rounded, atol=tol):
            return rounded / 2.0
        return None

    # Try different ranks
    for r in range(15, max_rank + 1):
        # Try a few random initializations
        for attempt in range(100):
            # Initialize V and W with small random integers
            V = np.random.randint(-1, 2, size=(m * p, r)).astype(float)
            W = np.random.randint(-1, 2, size=(p * n, r)).astype(float)
            
            # Add some non-zero values to avoid singularity
            if np.all(V == 0): V[0,0] = 1
            if np.all(W == 0): W[0,0] = 1
            
            U = np.zeros((n * m, r))
            valid = True
            
            # Iterate ALS steps
            for step in range(10):
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        # A_ij is p x r
                        A_ij = np.zeros((p, r))
                        for k in range(p):
                            for r_idx in range(r):
                                A_ij[k, r_idx] = V[j * p + k, r_idx] * W[k * n + i, r_idx]
                        
                        b = np.ones(p)
                        try:
                            # Solve A_ij u = b
                            # Use lstsq for robustness
                            u, residuals, rank, s = np.linalg.lstsq(A_ij, b, rcond=None)
                            # Check if solution is exact
                            if np.allclose(A_ij @ u, b, atol=1e-9):
                                U[i * m + j, :] = u
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
                
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        B_jk = np.zeros((n, r))
                        for i in range(n):
                            for r_idx in range(r):
                                B_jk[i, r_idx] = U[i * m + j, r_idx] * W[k * n + i, r_idx]
                        
                        b = np.ones(n)
                        try:
                            v, residuals, rank, s = np.linalg.lstsq(B_jk, b, rcond=None)
                            if np.allclose(B_jk @ v, b, atol=1e-9):
                                V[j * p + k, :] = v
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
                
                # Solve for W
                for i in range(n):
                    for k in range(p):
                        C_ik = np.zeros((m, r))
                        for j in range(m):
                            for r_idx in range(r):
                                C_ik[j, r_idx] = U[i * m + j, r_idx] * V[j * p + k, r_idx]
                        
                        b = np.ones(m)
                        try:
                            w, residuals, rank, s = np.linalg.lstsq(C_ik, b, rcond=None)
                            if np.allclose(C_ik @ w, b, atol=1e-9):
                                W[k * n + i, :] = w
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                # Check if we can round to half-integers
                U_rounded = is_half_integer(U)
                V_rounded = is_half_integer(V)
                W_rounded = is_half_integer(W)
                
                if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                    # Verify the rounded solution
                    if verify(U_rounded, V_rounded, W_rounded, r):
                        # Convert to lists of lists for the interface
                        U_list = U_rounded.tolist()
                        V_list = V_rounded.tolist()
                        W_list = W_rounded.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
        
    # Fallback: Schoolbook algorithm
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
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
