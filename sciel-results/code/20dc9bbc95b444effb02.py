import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook decomposition
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Lower bound for rank
    min_rank = max(n * m, m * p, p * n)
    
    # Strategy: Systematic search with structured initialization
    # Try ranks from lower bound up to schoolbook
    for target_rank in range(min_rank, best_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Number of attempts: more for lower ranks
        max_attempts = 3000 if target_rank <= 20 else 1500 if target_rank <= 25 else 500
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break
            
            # Generate random U, V with small integer entries
            # Use a mix of sparse and denser matrices
            if target_rank <= 18:
                density = 0.25
                max_val = 2
            elif target_rank <= 22:
                density = 0.3
                max_val = 2
            else:
                density = 0.35
                max_val = 3
            
            U = _random_sparse_matrix(n * m, target_rank, density, max_val)
            V = _random_sparse_matrix(m * p, target_rank, density, max_val)
            
            # Try to construct W
            W = [[0] * target_rank for _ in range(p * n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Compute M_ik[c] = sum_j U[i*m+j][c] * V[j*p+k][c]
                    M = [0] * target_rank
                    for j in range(m):
                        for c in range(target_rank):
                            M[c] += U[i * m + j][c] * V[j * p + k][c]
                    
                    # Find c where M[c] is in {1, -1, 2, -2}
                    found = False
                    for c in range(target_rank):
                        if M[c] == 1:
                            W[k * n + i][c] = 1
                            found = True
                            break
                        elif M[c] == -1:
                            W[k * n + i][c] = -1
                            found = True
                            break
                        elif M[c] == 2:
                            W[k * n + i][c] = 0.5
                            found = True
                            break
                        elif M[c] == -2:
                            W[k * n + i][c] = -0.5
                            found = True
                            break
                    
                    if not found:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify the decomposition
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        # If we didn't find one for this rank, try the next rank
    
    return best_result

def _random_sparse_matrix(rows, cols, density, max_val):
    """Generate a random sparse matrix with small integer entries."""
    mat = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            if random.random() < density:
                val = random.randint(-max_val, max_val)
                if val == 0:
                    val = 1
                row.append(val)
            else:
                row.append(0)
        mat.append(row)
    return mat

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
