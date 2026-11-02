import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0

    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R

    U_sb, V_sb, W_sb, R_sb = schoolbook()
    best_U, best_V, best_W, best_R = U_sb, V_sb, W_sb, R_sb

    def verify(U, V, W, R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integer(mat, tol=1e-6):
        snapped = np.round(mat * 2) / 2
        if np.max(np.abs(mat - snapped)) < tol:
            return snapped
        return mat

    # Algebraic elimination: for each rank R, try sparse support patterns
    # where U and V have entries from {-1, -0.5, 0, 0.5, 1} with limited nonzeros per row,
    # then solve the linear system for W.
    
    # The key insight: fix U and V, then for each output (i,k), the equation is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
    # This is a linear system in W[k*n+i, :] for each fixed (i,k).
    
    # Actually, let's reformulate. For fixed U, V, the tensor equation is:
    # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Group by (i, k): for each r, the coefficient of W[k*n+i, r] is sum_j U[i*m+j, r] * V[j*p+k, r]
    # Wait, that's not right either. Let me re-derive.
    
    # The matrix multiplication tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all i,j,k.
    
    # For fixed U and V, this is a system of n*m*p equations in p*n*R unknowns (W entries).
    # But the equations are coupled. Let's think of it differently.
    
    # For each r, define a matrix M_r of size n x p where M_r[i,k] = sum_j U[i*m+j, r] * V[j*p+k, r]
    # Then the equation for (i,k) is: sum_r M_r[i,k] * W[k*n+i, r] = 1
    # This gives n*p equations in n*p*R unknowns.
    
    # Actually, let me just build the full linear system.
    # Variables: W[k*n+i, r] for all k, i, r. Total: p*n*R variables.
    # Equations: for each (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Total: n*m*p equations.
    
    # Build the system A * w = b where w is the flattened W matrix.
    # For equation (i,j,k): the coefficient of W[k*n+i, r] is U[i*m+j, r] * V[j*p+k, r]
    
    def try_algebraic_elimination(R, max_patterns=200):
        """Try to find a decomposition of rank R by fixing sparse U,V and solving for W."""
        if time.time() > deadline - 10:
            return None
        
        # Generate random sparse U and V patterns
        # Each row of U (size R) should have a few nonzeros
        # Each row of V (size R) should have a few nonzeros
        
        rng = np.random.default_rng(int(time.time() * 1000) % 1000000)
        
        # Try various sparsity levels
        for trial in range(max_patterns):
            if time.time() > deadline - 10:
                break
            
            # Random sparse U: each row has 1-3 nonzeros
            U = np.zeros((n * m, R))
            for row in range(n * m):
                nnz = rng.integers(1, min(4, R + 1))
                cols = rng.choice(R, size=nnz, replace=False)
                for c in cols:
                    U[row, c] = rng.choice([-1, -0.5, 0.5, 1])
            
            # Random sparse V: each row has 1-3 nonzeros
            V = np.zeros((m * p, R))
            for row in range(m * p):
                nnz = rng.integers(1, min(4, R + 1))
                cols = rng.choice(R, size=nnz, replace=False)
                for c in cols:
                    V[row, c] = rng.choice([-1, -0.5, 0.5, 1])
            
            # Build linear system for W
            # Variables: W[k*n+i, r] indexed by (k, i, r) -> flat index = (k*n+i)*R + r
            # Equations: (i, j, k) -> flat index = (i*m+j)*p + k
            n_eqs = n * m * p
            n_vars = p * n * R
            
            A = np.zeros((n_eqs, n_vars))
            b = np.ones(n_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            coeff = U[i * m + j, r] * V[j * p + k, r]
                            if coeff != 0:
                                var_idx = (k * n + i) * R + r
                                A[eq_idx, var_idx] = coeff
                        eq_idx += 1
            
            # Solve the linear system
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=1e-10)
            except Exception:
                continue
            
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            
            W = W_flat.reshape(p * n, R)
            
            # Verify
            if verify(U, V, W, R):
                # Try to snap to half-integers
                U_snapped = snap_to_half_integer(U)
                V_snapped = snap_to_half_integer(V)
                W_snapped = snap_to_half_integer(W)
                
                if verify(U_snapped, V_snapped, W_snapped, R):
                    return U_snapped, V_snapped, W_snapped
                
                # Check if W is close to half-integers
                W_close = snap_to_half_integer(W, tol=1e-4)
                if np.max(np.abs(W - W_close)) < 1e-4:
                    if verify(U, V, W_close, R):
                        return U, V, W_close
            
            # Also try: maybe U and V need adjustment. Use the solution W to inform
            # a refinement, but for now just check if we got a valid solution.
        
        return None

    # Try ranks from lower bound up to schoolbook
    min_rank_to_try = 14  # n*p = 14 is a lower bound
    
    for R in range(min_rank_to_try, min(R_sb + 1, 42)):
        if time.time() > deadline - 15:
            break
        
        result = try_algebraic_elimination(R, max_patterns=50)
        if result is not None:
            U_sol, V_sol, W_sol = result
            return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
    
    # If no better solution found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
