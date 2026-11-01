"""Rewritten program for matrix multiplication 2,4,5 rank-minimal decomposition."""
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a low-rank decomposition of the 2x4x5 matrix multiplication tensor."""
    n, m, p = DIMENSIONS  # 2, 4, 5
    nm, mp, pn = n * m, m * p, p * n  # 8, 20, 10
    
    # Try to find a decomposition of rank R
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Start with a reasonable rank and try to solve
    for R in range(16, 41):
        try:
            result = _solve_decomposition(n, m, p, R)
            if result is not None:
                U, V, W = result
                # Verify the decomposition
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
        except Exception:
            continue
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _solve_decomposition(n, m, p, R):
    """Try to find a rank-R decomposition using iterative optimization."""
    nm, mp, pn = n * m, m * p, p * n
    
    import random
    random.seed(42)
    
    # Initialize with random values
    U = [[random.uniform(-1, 1) for _ in range(R)] for _ in range(nm)]
    V = [[random.uniform(-1, 1) for _ in range(R)] for _ in range(mp)]
    W = [[random.uniform(-1, 1) for _ in range(R)] for _ in range(pn)]
    
    # Use alternating minimization
    for iteration in range(1000):
        # Update U
        U = _update_U(V, W, n, m, p, R)
        # Update V
        V = _update_V(U, W, n, m, p, R)
        # Update W
        W = _update_W(U, V, n, m, p, R)
        
        # Check convergence
        if _verify(U, V, W, n, m, p):
            # Try to make entries integer or half-integer
            U, V, W = _round_to_half_integers(U, V, W)
            if _verify(U, V, W, n, m, p):
                return U, V, W
    
    return None

def _update_U(V, W, n, m, p, R):
    """Update U by solving a linear system for each (i,j)."""
    nm = n * m
    U = [[0.0] * R for _ in range(nm)]
    
    for i in range(n):
        for j in range(m):
            # For each k, we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a linear system in U[i*m+j, :]
            # Build the system: for each k, sum_r U[r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
            A = np.zeros((p, R))
            b = np.ones(p)
            for k in range(p):
                for r in range(R):
                    A[k, r] = V[j * p + k][r] * W[k * n + i][r]
            
            # Solve least squares
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i * m + j] = list(sol)
            except:
                U[i * m + j] = [1.0 / R] * R
    
    return U

def _update_V(U, W, n, m, p, R):
    """Update V by solving a linear system for each (j,k)."""
    mp = m * p
    V = [[0.0] * R for _ in range(mp)]
    
    for j in range(m):
        for k in range(p):
            # For each i, we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a linear system in V[j*p+k, :]
            A = np.zeros((n, R))
            b = np.ones(n)
            for i in range(n):
                for r in range(R):
                    A[i, r] = U[i * m + j][r] * W[k * n + i][r]
            
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[j * p + k] = list(sol)
            except:
                V[j * p + k] = [1.0 / R] * R
    
    return V

def _update_W(U, V, n, m, p, R):
    """Update W by solving a linear system for each (k,i)."""
    pn = p * n
    W = [[0.0] * R for _ in range(pn)]
    
    for k in range(p):
        for i in range(n):
            # For each j, we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a linear system in W[k*n+i, :]
            A = np.zeros((m, R))
            b = np.ones(m)
            for j in range(m):
                for r in range(R):
                    A[j, r] = U[i * m + j][r] * V[j * p + k][r]
            
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[k * n + i] = list(sol)
            except:
                W[k * n + i] = [1.0 / R] * R
    
    return W

def _round_to_half_integers(U, V, W):
    """Round entries to nearest half-integer and verify."""
    def round_matrix(M):
        return [[round(x * 2) / 2 for x in row] for row in M]
    
    U_r = round_matrix(U)
    V_r = round_matrix(V)
    W_r = round_matrix(W)
    
    return U_r, V_r, W_r
# EVOLVE-BLOCK-END
