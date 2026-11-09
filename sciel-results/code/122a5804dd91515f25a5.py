"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
from fractions import Fraction

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if len(U) > 0 else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 0.0
                            for c in range(rank):
                                lhs += U[i*m+j][c] * V[jp*p+k][c] * W[kp*n+ip][c]
                            rhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            if abs(lhs - rhs) > tol:
                                return False
    return True

def _try_round_to_rational(U, V, W, n, m, p):
    """Try to round entries to integer or half-integer values and verify."""
    for denom in [1, 2, 4, 8]:
        U_r = np.round(np.array(U) * denom) / denom
        V_r = np.round(np.array(V) * denom) / denom
        W_r = np.round(np.array(W) * denom) / denom
        
        if _verify(U_r, V_r, W_r, n, m, p, tol=1e-6):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
    return None

def _loss_and_grad(x, n, m, p, r):
    """
    Compute loss and gradient for tensor decomposition.
    x is a flattened array of length (n*m + m*p + p*n) * r
    Order: U (n*m*r), V (m*p*r), W (p*n*r)
    """
    U = x[:n*m*r].reshape(n*m, r)
    V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
    W = x[n*m*r + m*p*r :].reshape(p*n, r)
    
    loss = 0.0
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            rhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            s = np.dot(np.dot(U[i*m+j], V[jp*p+k]), W[kp*n+ip])
                            err = s - rhs
                            loss += err * err
                            dU[i*m+j] += 2 * err * (V[jp*p+k] * W[kp*n+ip])
                            dV[jp*p+k] += 2 * err * (U[i*m+j] * W[kp*n+ip])
                            dW[kp*n+ip] += 2 * err * (U[i*m+j] * V[jp*p+k])
                            
    grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    return loss, grad

def _strassen_4x4():
    """Return Strassen decomposition for 4x4 matrix multiplication (rank 7)."""
    # This is a known rank-7 decomposition for 4x4
    # Using the standard Strassen-like construction
    n = 4
    r = 7
    
    # Define the 7 terms for 4x4 multiplication
    # We'll use a construction based on splitting into 2x2 blocks
    U = np.zeros((n*n, r))
    V = np.zeros((n*n, r))
    W = np.zeros((n*n, r))
    
    # 2x2 Strassen has rank 7, we can extend to 4x4
    # Actually, let's use a simpler approach: direct construction
    # For 4x4, we can use rank 16 (schoolbook) or try lower
    
    # Let's just return a schoolbook for now and improve
    for i in range(n):
        for j in range(n):
            for k in range(n):
                col = i*n + j*n + k  # This won't work, need different indexing
    # This is getting complex. Let's use a different approach.
    return None

def _try_direct_construction(r, n, m, p, deadline, seed):
    """Try to find a rank-r decomposition with structured initialization."""
    if time.time() > deadline - 10:
        return None
        
    np.random.seed(seed)
    
    # Try multiple initialization strategies
    strategies = [
        'random', 'sparse', 'structured'
    ]
    
    for strat in strategies:
        if time.time() > deadline - 5:
            break
            
        if strat == 'random':
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
        elif strat == 'sparse':
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            # Randomly place non-zeros
            for c in range(r):
                # Each column has some pattern
                U[np.random.choice(n*m, size=min(3, n*m), replace=False), c] = np.random.choice([-1, 1], size=min(3, n*m))
                V[np.random.choice(m*p, size=min(3, m*p), replace=False), c] = np.random.choice([-1, 1], size=min(3, m*p))
                W[np.random.choice(p*n, size=min(3, p*n), replace=False), c] = np.random.choice([-1, 1], size=min(3, p*n))
        else:  # structured
            # Try to exploit the rectangular structure
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Assign each (i,j,k) triple to a rank component
            # For 2x4x6, we have 48 triples
            # Try to group them
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Assign to rank components in a structured way
                        # This is heuristic
                        pass
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def fun(x):
            loss, grad = _loss_and_grad(x, n, m, p, r)
            return loss, grad
        
        try:
            res = opt.minimize(fun, x0, jac=True, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50})
            
            if res.fun < 1e-4:
                x_sol = res.x
                U_sol = x_sol[:n*m*r].reshape(n*m, r)
                V_sol = x_sol[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_sol = x_sol[n*m*r + m*p*r :].reshape(p*n, r)
                
                rounded = _try_round_to_rational(U_sol, V_sol, W_sol, n, m, p)
                if rounded is not None:
                    return rounded
                    
        except Exception:
            pass
            
    return None

def _embed_and_project(n, m, p, deadline):
    """
    Try to embed 2x4x6 into a larger problem with known low-rank decomposition.
    Strategy: 2x4x6 can be seen as 2 * (4x6), and 4x6 = 4x(3+3) = 4x3 + 4x3
    If we can find rank-r decompositions for 4x3x3, we can combine them.
    """
    if time.time() > deadline - 5:
        return None
    
    # 4x3x3 schoolbook rank is 12
    # Try to find lower rank for 4x3x3
    # Then 2x4x6 would be 2 * rank(4x3x3) if we can separate the two 4x3 blocks
    
    # Actually, 2x4x6 = (2x4) x (4x6)
    # We can write the 4x6 as two 4x3 matrices side by side
    # So C = A * [B1 | B2] = [A*B1 | A*B2]
    # This doesn't directly help with tensor rank
    
    # Alternative: use the fact that 2x4x6 tensor can be embedded in 4x4x4
    # 2x4x6 has 48 entries, 4x4x4 has 64
    # If we have a rank-r decomposition for 4x4x4, we can project
    
    # For now, let's try direct optimization with better initializations
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin before 120s limit
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: for 2x4x6, the tensor rank is at least 
    # max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # But practical lower bounds are higher. Let's try from 47 down.
    
    # Try ranks from 47 down to 30
    for r in range(best_rank - 1, 29, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple seeds for each rank
        for seed in range(3):
            if time.time() > deadline:
                break
                
            result = _try_direct_construction(r, n, m, p, deadline, seed + r * 100)
            if result is not None:
                U_list, V_list, W_list = result
                best_rank = r
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                print(f"Found rank {r} decomposition")
                found = True
                break
        
        if found:
            break
            
    return best_result
# EVOLVE-BLOCK-END
