"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Tensor T[i, j, k] = 1 for all valid indices
    # i: 0..n-1, j: 0..m-1, k: 0..p-1
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Try ranks from low to high
    for rank in range(10, best_rank + 1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(trial * 1000 + rank)
            
            # Initialize with small random values
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Gradient descent to minimize sum of squared errors
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                
                # Compute current tensor
                error = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] = 1
                            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            val = 0.0
                            for c in range(rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err = val - 1.0
                            error += err * err
                            
                            for c in range(rank):
                                grad_U[i*m+j, c] += 2 * err * V[j*p+k, c] * W[k*n+i, c]
                                grad_V[j*p+k, c] += 2 * err * U[i*m+j, c] * W[k*n+i, c]
                                grad_W[k*n+i, c] += 2 * err * U[i*m+j, c] * V[j*p+k, c]
                
                if error < 1e-10:
                    break
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate
                if step % 100 == 99:
                    lr *= 0.95
            
            if error < 1e-8:
                # Found a solution! Try to snap to rationals
                # Convert to fractions with small denominators
                U_frac = np.zeros((n*m, rank), dtype=object)
                V_frac = np.zeros((m*p, rank), dtype=object)
                W_frac = np.zeros((p*n, rank), dtype=object)
                
                success = True
                for idx in range(n*m):
                    for c in range(rank):
                        frac = Fraction(U[idx, c]).limit_denominator(100)
                        U_frac[idx, c] = frac
                for idx in range(m*p):
                    for c in range(rank):
                        frac = Fraction(V[idx, c]).limit_denominator(100)
                        V_frac[idx, c] = frac
                for idx in range(p*n):
                    for c in range(rank):
                        frac = Fraction(W[idx, c]).limit_denominator(100)
                        W_frac[idx, c] = frac
                
                # Verify the snapped solution
                verify_ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = Fraction(0)
                            for c in range(rank):
                                val += U_frac[i*m+j, c] * V_frac[j*p+k, c] * W_frac[k*n+i, c]
                            if val != 1:
                                verify_ok = False
                                break
                        if not verify_ok:
                            break
                    if not verify_ok:
                        break
                
                if verify_ok:
                    # Convert to list of lists with float representation
                    U_list = [[float(U_frac[i, c]) for c in range(rank)] for i in range(n*m)]
                    V_list = [[float(V_frac[i, c]) for c in range(rank)] for i in range(m*p)]
                    W_list = [[float(W_frac[i, c]) for c in range(rank)] for i in range(p*n)]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    break  # Found minimum rank, stop searching
                
        if best_result is not None:
            break
    
    if best_result is None:
        # Fall back to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
