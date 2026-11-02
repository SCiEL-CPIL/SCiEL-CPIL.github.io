import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10
    
    deadline = time.time() + 130  # safety margin
    
    def validate(U, V, W, tol=1e-9):
        """Check if decomposition is valid: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)"""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def snap_half_ints(M):
        """Snap matrix entries to nearest half-integer."""
        return [[round(x * 2) / 2 for x in row] for row in M]
    
    def snap_ints(M):
        """Snap matrix entries to nearest integer."""
        return [[round(x) for x in row] for row in M]
    
    def try_rank(rank, seed=0, max_iters=500):
        """
        Try to find valid decomposition of given rank using alternating linear solves.
        """
        import random
        rng = random.Random(seed)
        
        # Initialize with small random values
        U = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(nm)]
        V = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(mp)]
        W = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(pn)]
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each (i,j), solve A * U_row = ones(p)
            # A[k,c] = V[j*p+k][c] * W[k*n+i][c]
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for c in range(rank):
                            A[k, c] = V[j*p+k][c] * W[k*n+i][c]
                    b = np.ones(p)
                    try:
                        # Use least squares (not NNLS) to allow negative values
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        U[i*m+j] = sol.tolist()
                    except:
                        pass
            
            # Update V: for each (j,k), solve A * V_row = ones(n)
            # A[i,c] = U[i*m+j][c] * W[k*n+i][c]
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for c in range(rank):
                            A[i, c] = U[i*m+j][c] * W[k*n+i][c]
                    b = np.ones(n)
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        V[j*p+k] = sol.tolist()
                    except:
                        pass
            
            # Update W: for each (k,i), solve A * W_row = ones(m)
            # A[j,c] = U[i*m+j][c] * V[j*p+k][c]
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            A[j, c] = U[i*m+j][c] * V[j*p+k][c]
                    b = np.ones(m)
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i] = sol.tolist()
                    except:
                        pass
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
            
            if err < 1e-12:
                # Found a solution! Try to snap to half-integers
                U_r = snap_half_ints(U)
                V_r = snap_half_ints(V)
                W_r = snap_half_ints(W)
                if validate(U_r, V_r, W_r, tol=1e-9):
                    return U_r, V_r, W_r
                
                U_i = snap_ints(U)
                V_i = snap_ints(V)
                W_i = snap_ints(W)
                if validate(U_i, V_i, W_i, tol=1e-9):
                    return U_i, V_i, W_i
        
        # If best_err is very small, try snapping the best solution found
        if best_err < 1e-8 and best_U is not None:
            U_r = snap_half_ints(best_U)
            V_r = snap_half_ints(best_V)
            W_r = snap_half_ints(best_W)
            if validate(U_r, V_r, W_r, tol=1e-9):
                return U_r, V_r, W_r
            
            U_i = snap_ints(best_U)
            V_i = snap_ints(best_V)
            W_i = snap_ints(best_W)
            if validate(U_i, V_i, W_i, tol=1e-9):
                return U_i, V_i, W_i
        
        return None
    
    def try_structured_init(rank, seed=0):
        """
        Try with structured initialization that respects tensor symmetries.
        """
        import random
        rng = random.Random(seed)
        
        # Initialize U with identity-like pattern for first n*m entries
        U = [[0.0 for _ in range(rank)] for _ in range(nm)]
        V = [[0.0 for _ in range(rank)] for _ in range(mp)]
        W = [[0.0 for _ in range(rank)] for _ in range(pn)]
        
        # Set up a base pattern: for each c, assign a "role"
        for c in range(rank):
            # Randomly assign which (i,j) this column contributes to in U
            i = c % n
            j = (c // n) % m
            U[i*m+j][c] = 1.0
            
            # For V, assign to (j,k)
            k = c % p
            V[j*p+k][c] = 1.0
            
            # For W, assign to (k,i)
            W[k*n+i][c] = 1.0
        
        # Add some random perturbation
        for i in range(nm):
            for c in range(rank):
                U[i][c] += rng.uniform(-0.1, 0.1)
        for j in range(mp):
            for c in range(rank):
                V[j][c] += rng.uniform(-0.1, 0.1)
        for k in range(pn):
            for c in range(rank):
                W[k][c] += rng.uniform(-0.1, 0.1)
        
        # Now run alternating optimization
        max_iters = 300
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for c in range(rank):
                            A[k, c] = V[j*p+k][c] * W[k*n+i][c]
                    b = np.ones(p)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i*m+j] = sol.tolist()
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for c in range(rank):
                            A[i, c] = U[i*m+j][c] * W[k*n+i][c]
                    b = np.ones(n)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j*p+k] = sol.tolist()
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            A[j, c] = U[i*m+j][c] * V[j*p+k][c]
                    b = np.ones(m)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i] = sol.tolist()
                    except:
                        pass
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
            
            if err < 1e-12:
                U_r = snap_half_ints(U)
                V_r = snap_half_ints(V)
                W_r = snap_half_ints(W)
                if validate(U_r, V_r, W_r, tol=1e-9):
                    return U_r, V_r, W_r
                U_i = snap_ints(U)
                V_i = snap_ints(V)
                W_i = snap_ints(W)
                if validate(U_i, V_i, W_i, tol=1e-9):
                    return U_i, V_i, W_i
        
        if best_err < 1e-8 and best_U is not None:
            U_r = snap_half_ints(best_U)
            V_r = snap_half_ints(best_V)
            W_r = snap_half_ints(best_W)
            if validate(U_r, V_r, W_r, tol=1e-9):
                return U_r, V_r, W_r
            U_i = snap_ints(best_U)
            V_i = snap_ints(best_V)
            W_i = snap_ints(best_W)
            if validate(U_i, V_i, W_i, tol=1e-9):
                return U_i, V_i, W_i
        
        return None
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = 15
    # But tighter: for matrix multiplication, rank >= max(n*m, m*p, p*n) is a lower bound
    # Actually the true lower bound is max(n*m, m*p, p*n) for the standard tensor rank
    
    best_result = None
    
    for R in range(15, 31):
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            
            result = try_rank(R, seed=seed, max_iters=300)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
        
        # Also try structured initialization
        for seed in range(3):
            if time.time() > deadline - 5:
                break
            
            result = try_structured_init(R, seed=seed)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
    
    # Fallback: return schoolbook decomposition (rank 30)
    rank = n * m * p  # 30
    U = [[0.0 for _ in range(rank)] for _ in range(nm)]
    V = [[0.0 for _ in range(rank)] for _ in range(mp)]
    W = [[0.0 for _ in range(rank)] for _ in range(pn)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][r] = 1.0
                V[j*p+k][r] = 1.0
                W[k*n+i][r] = 1.0
                r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
