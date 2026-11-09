"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Build target tensor T[a,b,c] where a=(i,j), b=(j',k), c=(i',k')
    # T[a,b,c] = 1 if j=j' and i=i' and k=k', else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank = 54
    
    # Try to find a lower rank decomposition using gradient-based optimization
    # Start from rank 27 (theoretical lower bound) up to schoolbook rank - 1
    for rank in range(27, best_rank):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        num_tries = min(5, max(1, int((deadline - 15 - time.time()) / 4)))
        
        for attempt in range(num_tries):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(rank * 10000 + attempt * 1000)
            U = np.random.randn(n*m, rank) * 0.3
            V = np.random.randn(m*p, rank) * 0.3
            W = np.random.randn(p*n, rank) * 0.3
            
            # Simple gradient descent with adaptive learning rate
            lr = 0.02
            for step in range(300):
                if time.time() > deadline - 15:
                    break
                
                approx = np.einsum('ar,br,cr->abc', U, V, W)
                diff = approx - T
                error = np.sum(diff**2)
                
                if error < 1e-10:
                    # Try rounding to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    approx_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    if np.allclose(approx_r, T, atol=1e-8):
                        U_frac = [[Fraction(U_r[a,r]).limit_denominator(2) for r in range(rank)] for a in range(n*m)]
                        V_frac = [[Fraction(V_r[b,r]).limit_denominator(2) for r in range(rank)] for b in range(m*p)]
                        W_frac = [[Fraction(W_r[c,r]).limit_denominator(2) for r in range(rank)] for c in range(p*n)]
                        best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                        best_rank = rank
                        break
                
                # Gradients
                gU = np.einsum('br,cr,abc->ar', V, W, diff)
                gV = np.einsum('ar,cr,abc->br', U, W, diff)
                gW = np.einsum('ar,br,abc->cr', U, V, diff)
                
                # Scale learning rate based on gradient magnitude
                scale = max(1.0, np.linalg.norm(gU) / 10)
                U -= lr * gU / scale
                V -= lr * gV / scale
                W -= lr * gW / scale
                
                # Decay learning rate
                lr *= 0.995
            
            if best_result is not None:
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U_out = [[float(x) for x in row] for row in best_result["U"]]
        V_out = [[float(x) for x in row] for row in best_result["V"]]
        W_out = [[float(x) for x in row] for row in best_result["W"]]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook decomposition of rank n*m*p
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
