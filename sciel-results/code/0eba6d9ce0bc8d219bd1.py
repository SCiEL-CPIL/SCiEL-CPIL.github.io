import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Target tensor: T[i, j, k, j'] = delta(j, j')
    # We want sum_r W[r, k, i] U[r, i, j] V[r, k, j'] = T[i, j, k, j']
    
    # Number of variables for U: r * n * m
    # Number of variables for V: r * m * p
    # Number of variables for W: r * p * n
    
    def solve_als(r, max_iters=100):
        # Initialize randomly
        U = np.random.randn(r, n, m)
        V = np.random.randn(r, m, p)
        W = np.random.randn(r, p, n)
        
        # Precompute indices for efficiency?
        # For each step, we build a linear system.
        
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Solve for U
            # Variable: U_flat of size r*n*m
            # Equation: sum_j U[r, i, j] * (sum_k,j' W[r, k, i] V[r, k, j'] * Target[i, j, k, j']) = ...
            # This is not quite right.
            # The target is T[i, j, k, j'].
            # The term in the sum for a specific (i, k, j, j') is:
            # sum_r W[r, k, i] U[r, i, j] V[r, k, j']
            # We want this to equal T[i, j, k, j'].
            
            # Let's vectorize.
            # For fixed r, i, k:
            # The matrix of coefficients for U[r, i, :] is related to V[r, :, k] and W[r, k, i].
            # Actually, U[r, i, j] is a scalar.
            
            # Let's build the linear system for U.
            # Unknowns: U[r, i, j] for all r, i, j.
            # For each (i, k, j, j'), we have an equation.
            # LHS: sum_r W[r, k, i] U[r, i, j] V[r, k, j']
            # RHS: T[i, j, k, j']
            
            # This is a linear system in U.
            # A_U has rows indexed by (i, k, j, j') and columns by (r, i, j).
            # A_U[(i, k, j, j'), (r, i', j')] = W[r, k, i] * V[r, k, j'] * delta(i, i') * delta(j, j')?
            # No, U is indexed by (r, i, j).
            # The term U[r, i, j] appears in the equation for (i, k, j, j') with coefficient W[r, k, i] V[r, k, j'].
            # So A_U has non-zero entries only when the row index (i, k, j, j') matches the column index (r, i, j) in terms of i and j.
            
            # Let's just use a dense matrix for small r.
            # Num rows: n * p * m * m = 4 * 5 * 5 * 5 = 500
            # Num cols: r * n * m = r * 20
            
            A_U = np.zeros((n * p * m * m, r * n * m))
            b_U = np.zeros(n * p * m * m)
            
            idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            target = 1.0 if j == jp else 0.0
                            b_U[idx] = target
                            # Coefficients for U
                            for r in range(r):
                                # U[r, i, j] is at col index r*n*m + i*m + j
                                col_idx = r * n * m + i * m + j
                                A_U[idx, col_idx] = W[r, k, i] * V[r, k, jp]
                            idx += 1
                            
            # Solve
            sol_U, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
            U = sol_U.reshape(r, n, m)
            
            # Solve for V
            A_V = np.zeros((n * p * m * m, r * m * p))
            b_V = np.zeros(n * p * m * m)
            idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            target = 1.0 if j == jp else 0.0
                            b_V[idx] = target
                            for r in range(r):
                                # V[r, jp, k] is at col index r*m*p + jp*p + k
                                col_idx = r * m * p + jp * p + k
                                A_V[idx, col_idx] = W[r, k, i] * U[r, i, j]
                            idx += 1
                            
            sol_V, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
            V = sol_V.reshape(r, m, p)
            
            # Solve for W
            A_W = np.zeros((n * p * m * m, r * p * n))
            b_W = np.zeros(n * p * m * m)
            idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            target = 1.0 if j == jp else 0.0
                            b_W[idx] = target
                            for r in range(r):
                                # W[r, k, i] is at col index r*p*n + k*n + i
                                col_idx = r * p * n + k * n + i
                                A_W[idx, col_idx] = U[r, i, j] * V[r, k, jp]
                            idx += 1
                            
            sol_W, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
            W = sol_W.reshape(r, p, n)
            
            # Check error
            err = 0.0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            val = 0.0
                            for r in range(r):
                                val += W[r, k, i] * U[r, i, j] * V[r, k, jp]
                            target = 1.0 if j == jp else 0.0
                            err += (val - target) ** 2
                            
            if err < 1e-8:
                # Round to half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify
                err2 = 0.0
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            for jp in range(m):
                                val = 0.0
                                for r in range(r):
                                    val += W_int[r, k, i] * U_int[r, i, j] * V_int[r, k, jp]
                                target = 1.0 if j == jp else 0.0
                                err2 += (val - target) ** 2
                                
                if err2 < 1e-8:
                    return U_int, V_int, W_int
                    
        return None

    # Try ranks
    best_result = None
    for r in range(10, 100):
        if time.time() > deadline:
            break
        res = solve_als(r, max_iters=50)
        if res is not None:
            best_result = res
            break # Found a solution, return it. Since we iterate up, this is minimal for this search.
            
    if best_result is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((rank, n, m))
        V = np.zeros((rank, m, p))
        W = np.zeros((rank, p, n))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[col, i, j] = 1
                    V[col, j, k] = 1
                    W[col, k, i] = 1
                    col += 1
        best_result = (U, V, W)
        
    U_mat, V_mat, W_mat = best_result
    
    # Convert to lists of lists as required by the interface
    # U shape (n*m, rank) -> U_mat is (rank, n, m).
    # We need U[i*m+j, r] = U_mat[r, i, j]
    U_out = [[0.0] * len(U_mat) for _ in range(n * m)]
    for r in range(len(U_mat)):
        for i in range(n):
            for j in range(m):
                U_out[i * m + j][r] = U_mat[r, i, j]
                
    V_out = [[0.0] * len(V_mat) for _ in range(m * p)]
    for r in range(len(V_mat)):
        for j in range(m):
            for k in range(p):
                V_out[j * p + k][r] = V_mat[r, j, k]
                
    W_out = [[0.0] * len(W_mat) for _ in range(p * n)]
    for r in range(len(W_mat)):
        for k in range(p):
            for i in range(n):
                W_out[k * n + i][r] = W_mat[r, k, i]
                
    return {"U": U_out, "V": V_out, "W": W_out}
