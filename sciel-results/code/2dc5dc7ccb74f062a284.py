"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

import numpy as np
import time

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give an exact decomposition of the matmul tensor."""
    rank = len(U[0]) if U else 0
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    T[a, b, c] += U[a][r] * V[b][r] * W[c][r]
    # Check against expected
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    return False
    # Check all other entries are zero
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    if abs(T[a, b, c]) > 1e-9:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition using numerical optimization
    # Start with schoolbook rank and try to reduce
    
    # First, let's try known small ranks
    # For 2x3 * 3x6, let's try ranks from 24 downward
    
    best_rank = n * m * p  # schoolbook
    best_decomp = None
    
    # Generate schoolbook decomposition as fallback
    U_sb = [[0] * best_rank for _ in range(n * m)]
    V_sb = [[0] * best_rank for _ in range(m * p)]
    W_sb = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank decompositions
    # Use a simple approach: try various ranks and use iterative refinement
    
    for target_rank in range(24, best_rank):  # try ranks from 24 up to 35
        if time.time() > deadline:
            break
        
        # Try random initialization and gradient descent
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + target_rank * 100 + attempt)
            
            # Random initialization
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Build target tensor
            T_target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        T_target[a, b, c] = 1.0
            
            # Iterative refinement: minimize ||T - sum U[:,r] V[:,r] W[:,r]||^2
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Compute current tensor
                T_curr = np.einsum('ar,br,cr->abc', U, V, W)
                residual = T_target - T_curr
                
                # Gradients
                dU = np.einsum('abc,br,cr->ar', residual, V, W)
                dV = np.einsum('abc,ar,cr->br', residual, U, W)
                dW = np.einsum('abc,ar,br->cr', residual, U, V)
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                err = np.linalg.norm(residual)
                if err < 1e-6:
                    # Try to round to exact values
                    U_rounded = np.round(U * 2) / 2  # half-integers
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify
                    if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                        found = True
                        best_rank = target_rank
                        best_decomp = (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                        break
                    
                    # Also try integer rounding
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
                        found = True
                        best_rank = target_rank
                        best_decomp = (U_int.tolist(), V_int.tolist(), W_int.tolist())
                        break
                    
                    # Try more aggressive rounding
                    for mult in [1, 2, 4]:
                        U_r = np.round(U * mult) / mult
                        V_r = np.round(V * mult) / mult
                        W_r = np.round(W * mult) / mult
                        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                            found = True
                            best_rank = target_rank
                            best_decomp = (U_r.tolist(), V_r.tolist(), W_r.tolist())
                            break
                    if found:
                        break
                
                if err < 1e-8:
                    break
            
            if found:
                break
        
        if found:
            break
    
    U, V, W = best_decomp
    return {"U": U, "V": V, "W": W}
