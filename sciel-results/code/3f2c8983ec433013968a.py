"""Combined program for matrix multiplication 2x3x7 rank minimization."""
import numpy as np
import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 140  # safety margin before 150s kill

    # Schoolbook as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    def verify_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 1)

    def try_structured_search(target_rank, max_attempts=200):
        """
        Uses the decoupled linear system structure.
        For fixed U, V, the system for W decouples into n*p independent sub-systems.
        Each sub-system is M @ w = 1, where M is m x rank.
        We solve for w and check if it's close to integer.
        """
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Generate random U, V with entries in {0, 1} or {-1, 0, 1}
            # Using {0, 1} first as it corresponds to F2
            val_set = [0, 1] if attempt % 2 == 0 else [-1, 0, 1]
            
            U = [[random.choice(val_set) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.choice(val_set) for _ in range(target_rank)] for _ in range(mp)]
            
            # Convert to numpy for easier manipulation
            U_np = np.array(U).reshape(n, m, target_rank)
            V_np = np.array(V).reshape(m, p, target_rank)
            
            W_solution = [[0] * target_rank for _ in range(pn)]
            valid = True
            
            # Solve for each (i, k)
            for i in range(n):
                for k in range(p):
                    # Construct M: m x target_rank
                    # M[j, c] = U[i, j, c] * V[j, k, c]
                    M = U_np[i, :, :] * V_np[:, k, :]
                    
                    # Solve M @ w = 1
                    # If M has rank < m, there might be no solution or infinite.
                    # lstsq gives min-norm solution.
                    try:
                        w, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    # Check if w is close to integers
                    w_int = np.round(w)
                    if np.allclose(M @ w_int, np.ones(m), atol=1e-9):
                        # Store W[k, i, c]
                        for c in range(target_rank):
                            W_solution[k * n + i][c] = int(w_int[c])
                    else:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U, "V": V, "W": W_solution}
        
        return None

    # Try ranks from 41 down to 35
    for target_rank in range(41, 34, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_structured_search(target_rank, max_attempts=100)
        if result:
            best = result
            best_rank = target_rank
            # If we found a better rank, we can break if we want the minimum,
            # but we should keep searching for even lower if time permits?
            # The loop goes down, so we will find the lowest we can in the time limit.
            # However, if we find 41, we continue to 40.
            continue
    
    return best
# EVOLVE-BLOCK-END
