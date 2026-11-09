"""Program for matrix multiplication tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Trivial decomposition with rank n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: i*m + j
                U[i * m + j][r] = 1
                # Index in V: j*p + k
                V[j * p + k][r] = 1
                # Index in W: k*n + i
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for other combinations, the sum should be 0
    
    # Check all (i, j, k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                if abs(val - 1.0) > 1e-10:
                    return False
    
    # Check that non-matching triples give 0
    # U[i1*m + j1, r] * V[j2*p + k2, r] * W[k3*n + i2, r]
    # This should be 0 unless i1=i2, j1=j2, k1=k2
    for i1 in range(n):
        for j1 in range(m):
            for k1 in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i1 == i2 and j1 == j2 and k1 == k2:
                                continue
                            val = np.sum(U[i1*m + j1, :] * V[j2*p + k2, :] * W[k1*n + i2, :])
                            if abs(val) > 1e-10:
                                return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Start with the trivial decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Try to find lower rank decompositions
    # For matrix multiplication, the minimum rank is at least max(n*m, m*p, p*n)
    # and at most n*m*p
    # For 2x3x10: min possible is 30, max is 60
    
    # Try ranks from lower bound to current best
    min_possible = max(n*m, m*p, p*n)  # 30 for 2x3x10
    
    for target_rank in range(min_possible, best_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Try random initialization and optimization
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U_rand = np.random.randn(n*m, target_rank)
            V_rand = np.random.randn(m*p, target_rank)
            W_rand = np.random.randn(p*n, target_rank)
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_rand[i*m + j, :] * V_rand[j*p + k, :] * W_rand[k*n + i, :])
                        if abs(val - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Check non-matching triples
            for i1 in range(n):
                for j1 in range(m):
                    for k1 in range(p):
                        for i2 in range(n):
                            for j2 in range(m):
                                for k2 in range(p):
                                    if i1 == i2 and j1 == j2 and k1 == k2:
                                        continue
                                    val = np.sum(U_rand[i1*m + j1, :] * V_rand[j2*p + k2, :] * W_rand[k1*n + i2, :])
                                    if abs(val) > 1e-8:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Found a valid decomposition
            best_rank = target_rank
            best_U = U_rand.tolist()
            best_V = V_rand.tolist()
            best_W = W_rand.tolist()
            break
    
    if best_rank < rank:
        U = best_U
        V = best_V
        W = best_W
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
