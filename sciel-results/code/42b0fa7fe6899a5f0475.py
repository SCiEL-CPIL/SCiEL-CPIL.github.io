import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x5 matrix.
    
    Tries to find a lower-rank decomposition than schoolbook (rank 60)
    using structured search, falling back to schoolbook if none found.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

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

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build target tensor
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1.0
        
        # Compute decomposition
        T_comp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        return np.allclose(T_comp, T_target, atol=tol)

    # Start with schoolbook as the guaranteed fallback
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Verify schoolbook works
    if not verify(best_result["U"], best_result["V"], best_result["W"]):
        pass  # Should never happen

    # Try to find lower-rank decompositions using a direct algebraic approach
    # Strategy: For each candidate rank r < 60, try to solve the system of
    # polynomial equations for U, V, W entries.
    
    # The system: for each (i,j,k) with i in [0,n), j in [0,m), k in [0,p):
    #   sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    # For all other (a,b,c) combinations where the index pattern doesn't match:
    #   sum_t U[a, t] * V[b, t] * W[c, t] = 0
    
    # This is a system of trilinear equations. We'll try a randomized approach:
    # 1. Randomly initialize U, V, W with small integer/half-integer values
    # 2. Use gradient-based or local search to minimize the residual
    
    def try_rank(r, max_iters=5000):
        """Try to find a rank-r decomposition using local search."""
        if time.time() > deadline - 10:
            return None
        
        # Number of constraints
        num_constraints = n * m * m * p * p * n  # Total tensor size
        # But we only need to satisfy the 60 non-zero entries and ensure others are 0
        
        # Initialize with random small values
        np.random.seed(42 + r)
        
        # Use a structured initialization based on schoolbook terms
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Assign schoolbook terms to first min(r, 60) columns
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < r:
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
        
        # For remaining columns, add small random perturbations
        for c in range(col, r):
            U[:, c] = np.random.choice([-0.5, 0.5], size=n * m)
            V[:, c] = np.random.choice([-0.5, 0.5], size=m * p)
            W[:, c] = np.random.choice([-0.5, 0.5], size=p * n)
        
        # Local search: try to reduce the residual
        best_residual = np.inf
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute residual
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            T_target = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i * m + j, j * p + k, k * n + i] = 1.0
            
            residual = np.sum((T_comp - T_target) ** 2)
            
            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if residual < 1e-12:
                break
            
            # Try flipping a single entry
            if np.random.random() < 0.33:
                a = np.random.randint(n * m)
                c = np.random.randint(r)
                U[a, c] = -U[a, c] if U[a, c] != 0 else 0.5
            elif np.random.random() < 0.66:
                b = np.random.randint(m * p)
                c = np.random.randint(r)
                V[b, c] = -V[b, c] if V[b, c] != 0 else 0.5
            else:
                c = np.random.randint(p * n)
                d = np.random.randint(r)
                W[c, d] = -W[c, d] if W[c, d] != 0 else 0.5
        
        if best_residual < 1e-12:
            # Convert to lists with integer/half-integer entries
            U_list = [[round(float(x), 1) for x in row] for row in best_U]
            V_list = [[round(float(x), 1) for x in row] for row in best_V]
            W_list = [[round(float(x), 1) for x in row] for row in best_W]
            
            # Verify
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try ranks from 59 down to 20
    for r in range(best_rank - 1, 19, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    # Final verification
    if verify(best_result["U"], best_result["V"], best_result["W"]):
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
