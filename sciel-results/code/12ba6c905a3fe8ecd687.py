"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_rational(mat, max_denom=100):
    """Round entries to rationals with small denominators."""
    from fractions import Fraction
    rounded = []
    for row in mat:
        rrow = []
        for val in row:
            # Try to find a fraction with small denominator
            best = None
            best_err = float('inf')
            for d in range(1, max_denom+1):
                num = round(val * d)
                frac = Fraction(num, d)
                err = abs(val - float(frac))
                if err < best_err:
                    best_err = err
                    best = frac
            rrow.append(float(best))
        rounded.append(rrow)
    return rounded

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Try rank 14 first
    for target_rank in [14, 15, 16, 17, 18]:
        if time.time() > deadline - 30:
            break
        
        # Use random initialization and gradient descent
        best_U, best_V, best_W = None, None, None
        
        n_attempts = 5 if target_rank <= 14 else 3
        for attempt in range(n_attempts):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(42 + attempt)
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Gradient descent
            lr = 0.01
            for step in range(5000):
                if time.time() > deadline - 15:
                    break
                
                # Compute gradient
                dU = np.zeros((nm, target_rank))
                dV = np.zeros((mp, target_rank))
                dW = np.zeros((pn, target_rank))
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                            # dU[i*m+j,r] += (T - 1) * V[j*p+k,r] * W[k*n+i,r]
                            # dV[j*p+k,r] += (T - 1) * U[i*m+j,r] * W[k*n+i,r]
                            # dW[k*n+i,r] += (T - 1) * U[i*m+j,r] * V[j*p+k,r]
                            T_val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                            err = T_val - 1.0
                            dU[i*m+j] += err * V[j*p+k] * W[k*n+i]
                            dV[j*p+k] += err * U[i*m+j] * W[k*n+i]
                            dW[k*n+i] += err * U[i*m+j] * V[j*p+k]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                if step % 100 == 0:
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                                max_err = max(max_err, abs(T_val - 1.0))
                    if max_err < 1e-6:
                        break
            
            # Check if we got a good solution
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        max_err = max(max_err, abs(T_val - 1.0))
            
            if max_err < 1e-6:
                # Round to rationals and verify
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                # Try rounding to simple rationals
                for max_denom in [1, 2, 4, 8, 16, 32, 64, 100]:
                    U_r = _round_to_rational(U_list, max_denom)
                    V_r = _round_to_rational(V_list, max_denom)
                    W_r = _round_to_rational(W_list, max_denom)
                    
                    if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-9):
                        return {"U": U_r, "V": V_r, "W": W_r}
                
                # If rounding didn't work, use the float values (should be fine for exact check)
                if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we couldn't find rank target_rank, try next
        if target_rank >= 18:
            # Fall back to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(nm)]
            V = [[0] * rank for _ in range(mp)]
            W = [[0] * rank for _ in range(pn)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][column] = 1
                        V[j*p+k][column] = 1
                        W[k*n+i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
    
    # Final fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
