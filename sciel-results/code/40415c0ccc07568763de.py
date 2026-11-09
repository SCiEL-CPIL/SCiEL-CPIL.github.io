"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Start with trivial decomposition (rank = n*m*p)
    base_rank = n * m * p
    best_U = [[0] * base_rank for _ in range(n * m)]
    best_V = [[0] * base_rank for _ in range(m * p)]
    best_W = [[0] * base_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # Try to reduce rank by merging terms
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # We want to find minimal rank with integer/half-integer entries
    
    # Strategy: Try rank values from lower bound upward
    # Lower bound for matrix multiplication is max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # But we can potentially do better with clever decompositions
    
    # For 2x3 * 3x9, let's try various rank values
    # First, verify the trivial solution works
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # T[i,j,k] should equal sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1  # delta(i,j,k) in the multiplication tensor
                    actual = 0
                    for r in range(rank):
                        actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Try to find better decompositions using structured approaches
    # For small matrices, we can try random search with integer coefficients
    
    best_rank = base_rank
    best_result = (best_U, best_V, best_W)
    
    # Try structured approaches for lower ranks
    # One approach: use the fact that we're multiplying (2,3) x (3,9)
    # Result is (2,9) matrix
    # Each output element C[i,k] = sum_j A[i,j] * B[j,k]
    # So we have 2*9 = 18 output elements, each is a sum of 3 products
    
    # Try rank = 18 (minimum possible since we have 18 outputs)
    # This would require each rank-1 term to contribute to exactly one output
    
    # Actually, let's think more carefully. The tensor has 2*3*9 = 54 non-zero entries.
    # A rank-1 term U[r] * V[r] * W[r] where U is (2,3), V is (3,9), W is (2,9)
    # contributes to all i,j,k where U[i,j,r]*V[j,k,r]*W[i,k,r] != 0
    
    # Let's try a direct search for small rank improvements
    # Start by trying to merge pairs of trivial terms
    
    # For now, try a few random searches with integer coefficients
    np.random.seed(42)
    
    # Try ranks from 30 down to 27 (reasonable lower bound)
    for target_rank in range(30, 26, -1):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        # Try random initialization with small integers
        for trial in range(50):
            if time.time() > deadline:
                break
            
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Initialize with random small integers
            for r in range(target_rank):
                U[:, r] = np.random.randint(-2, 3, size=n*m)
                V[:, r] = np.random.randint(-2, 3, size=m*p)
                W[:, r] = np.random.randint(-2, 3, size=p*n)
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1
                        actual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        if abs(actual - expected) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_result = (U.tolist(), V.tolist(), W.tolist())
                break
    
    # Convert best result to list format
    if isinstance(best_result[0], np.ndarray):
        U_final = best_result[0].tolist()
        V_final = best_result[1].tolist()
        W_final = best_result[2].tolist()
    else:
        U_final = best_result[0]
        V_final = best_result[1]
        W_final = best_result[2]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
