"""Matrix multiplication 2x3x4 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    n, m, p = 2, 3, 4
    nmp = n * m * p  # 24
    
    # Schoolbook decomposition (rank 24)
    def schoolbook():
        rank = nmp
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
    
    best = schoolbook()
    best_rank = nmp
    
    # Try to find a better decomposition using linear algebra
    def try_solve(U, V, r):
        """Try to solve for W given U and V."""
        W = [[Fraction(0)]*r for _ in range(p*n)]
        for i in range(n):
            for k in range(p):
                # For fixed (i,k), solve: sum_r (U[i*m+j][r] * V[j*p+k][r]) * W[k*n+i][r] = 1 for all j
                A = [[Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) for r in range(r)] for j in range(m)]
                # Solve A * x = 1
                try:
                    A_np = np.array([[float(a) for a in row] for row in A], dtype=float)
                    ones = np.ones(m)
                    x = np.linalg.solve(A_np, ones)
                    # Convert to fractions
                    x_frac = []
                    valid = True
                    for r_idx in range(r):
                        val = Fraction(x[r_idx]).limit_denominator(1000)
                        if abs(float(val) - x[r_idx]) > 1e-6:
                            valid = False
                            break
                        x_frac.append(val)
                    if valid:
                        W[k*n+i] = x_frac
                    else:
                        return None
                except np.linalg.LinAlgError:
                    return None
        # Verify the solution
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(r):
                        total += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * W[k*n+i][r]
                    if total != 1:
                        return None
        return W
    
    # Try to find a decomposition with rank < 24
    for target_rank in range(23, 11, -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few random U and V
        for trial in range(10):
            if time.time() > deadline - 30:
                break
            
            # Initialize U and V with random small integer values
            U = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(m*p)]
            
            W = try_solve(U, V, target_rank)
            if W is not None:
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if best_rank < 24:
            break
    
    return best
