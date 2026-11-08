import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 4
    
    # Build the target tensor
    T_expected = np.zeros((9, 12, 12), dtype=int)
    for i in range(3):
        for j in range(3):
            for k in range(4):
                T_expected[i*3+j, j*4+k, k*3+i] = 1
    
    # Helper to verify a decomposition
    def verify(U, V, W):
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        return np.array_equal(T_recon, T_expected)
    
    # Start with the known rank-34 construction
    rank = 34
    U = np.zeros((9, rank), dtype=int)
    V = np.zeros((12, rank), dtype=int)
    W = np.zeros((12, rank), dtype=int)
    
    col = 0
    # Strassen Left (A11 * B1L)
    U[0, col] = 1; U[4, col] = 1; V[0, col] = 1; V[5, col] = 1; W[0, col] = 1; W[4, col] = 1; col += 1
    U[3, col] = 1; U[4, col] = 1; V[0, col] = 1; W[1, col] = 1; W[4, col] = -1; col += 1
    U[0, col] = 1; V[1, col] = 1; V[5, col] = -1; W[3, col] = 1; W[4, col] = 1; col += 1
    U[4, col] = 1; V[4, col] = 1; V[0, col] = -1; W[0, col] = 1; W[1, col] = 1; col += 1
    U[0, col] = 1; U[1, col] = 1; V[5, col] = 1; W[0, col] = -1; W[3, col] = 1; col += 1
    U[3, col] = 1; U[0, col] = -1; V[0, col] = 1; V[1, col] = 1; W[4, col] = 1; col += 1
    U[1, col] = 1; U[4, col] = -1; V[4, col] = 1; V[5, col] = 1; W[0, col] = 1; col += 1
    U[2, col] = 1; V[8, col] = 1; W[0, col] = 1; col += 1
    U[2, col] = 1; V[9, col] = 1; W[3, col] = 1; col += 1
    U[5, col] = 1; V[8, col] = 1; W[1, col] = 1; col += 1
    U[5, col] = 1; V[9, col] = 1; W[4, col] = 1; col += 1
    U[6, col] = 1; V[0, col] = 1; W[2, col] = 1; col += 1
    U[6, col] = 1; V[1, col] = 1; W[5, col] = 1; col += 1
    U[7, col] = 1; V[4, col] = 1; W[2, col] = 1; col += 1
    U[7, col] = 1; V[5, col] = 1; W[5, col] = 1; col += 1
    U[8, col] = 1; V[8, col] = 1; W[2, col] = 1; col += 1
    U[8, col] = 1; V[9, col] = 1; W[5, col] = 1; col += 1
    # Strassen Right (A11 * B1R)
    U[0, col] = 1; U[4, col] = 1; V[2, col] = 1; V[7, col] = 1; W[6, col] = 1; W[10, col] = 1; col += 1
    U[3, col] = 1; U[4, col] = 1; V[2, col] = 1; W[7, col] = 1; W[10, col] = -1; col += 1
    U[0, col] = 1; V[3, col] = 1; V[7, col] = -1; W[9, col] = 1; W[10, col] = 1; col += 1
    U[4, col] = 1; V[6, col] = 1; V[2, col] = -1; W[6, col] = 1; W[7, col] = 1; col += 1
    U[0, col] = 1; U[1, col] = 1; V[7, col] = 1; W[6, col] = -1; W[9, col] = 1; col += 1
    U[3, col] = 1; U[0, col] = -1; V[2, col] = 1; V[3, col] = 1; W[10, col] = 1; col += 1
    U[1, col] = 1; U[4, col] = -1; V[6, col] = 1; V[7, col] = 1; W[6, col] = 1; col += 1
    U[2, col] = 1; V[10, col] = 1; W[6, col] = 1; col += 1
    U[2, col] = 1; V[11, col] = 1; W[9, col] = 1; col += 1
    U[5, col] = 1; V[10, col] = 1; W[7, col] = 1; col += 1
    U[5, col] = 1; V[11, col] = 1; W[10, col] = 1; col += 1
    U[6, col] = 1; V[2, col] = 1; W[8, col] = 1; col += 1
    U[6, col] = 1; V[3, col] = 1; W[11, col] = 1; col += 1
    U[7, col] = 1; V[6, col] = 1; W[8, col] = 1; col += 1
    U[7, col] = 1; V[7, col] = 1; W[11, col] = 1; col += 1
    U[8, col] = 1; V[10, col] = 1; W[8, col] = 1; col += 1
    U[8, col] = 1; V[11, col] = 1; W[11, col] = 1; col += 1
    
    # Verify the rank-34 construction
    if not verify(U, V, W):
        # Fallback to trivial rank-36
        rank = 36
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((12, rank), dtype=int)
        W = np.zeros((12, rank), dtype=int)
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    U[i*3+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce to rank 33 by searching for alternative decompositions
    # Strategy: Try to merge two terms into one using linear combinations
    # For rank 33, we need to find U', V', W' of shape (9,33), (12,33), (12,33)
    
    # Approach: Use random perturbation + local search
    # Start from rank-34, try to eliminate one column by absorbing it into others
    
    best_rank = 34
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try various strategies to reduce rank
    for target_rank in [33]:
        if time.time() > deadline - 10:
            break
        
        # Strategy 1: Try to find a valid rank-33 decomposition by modifying the rank-34 one
        # We'll try to set one column to zero and adjust others
        
        for col_to_eliminate in range(34):
            if time.time() > deadline - 10:
                break
            
            # Try eliminating column col_to_eliminate
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            U_test[:, col_to_eliminate] = 0
            V_test[:, col_to_eliminate] = 0
            W_test[:, col_to_eliminate] = 0
            
            # Now we have a rank-33 structure but it's incorrect
            # We need to find adjustments to the remaining columns
            
            # Compute the error
            T_err = T_expected - np.einsum('ir,jr,kr->ijk', U_test, V_test, W_test, optimize=True)
            
            if np.allclose(T_err, 0):
                best_rank = target_rank
                best_U = U_test[:, :target_rank]
                best_V = V_test[:, :target_rank]
                best_W = W_test[:, :target_rank]
                break
            
            # Try to fix by adding small corrections to existing columns
            # This is a linear system problem
            
        if best_rank == target_rank:
            break
    
    # If we found a better rank, use it
    if best_rank < 34:
        U = best_U
        V = best_V
        W = best_W
        rank = best_rank
    else:
        # Stick with rank 34
        U, V, W = U, V, W
        rank = 34
    
    # Final verification
    if not verify(U, V, W):
        # Fallback to trivial rank-36
        rank = 36
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((12, rank), dtype=int)
        W = np.zeros((12, rank), dtype=int)
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    U[i*3+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
