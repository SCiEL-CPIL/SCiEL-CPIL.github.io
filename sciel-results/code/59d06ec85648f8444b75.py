"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Target tensor T[i, j, k, j'] = delta_{j, j'}
    # Decomposition: T[i, j, k, j'] = sum_c U[i, j, c] * V[j', k, c] * W[k, i, c]
    # Note: V index is j', k. In the prompt, V is (m*p, rank) with index j*p+k.
    # So V[j'*p+k, c].
    # W index is k, i. In the prompt, W is (p*n, rank) with index k*n+i.
    # So W[k*n+i, c].
    # U index is i, j. In the prompt, U is (n*m, rank) with index i*m+j.
    # So U[i*m+j, c].
    
    # The condition is:
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = 1 if j==j' else 0
    
    best_result = None
    best_rank = n * m * p
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # c corresponds to (i, j, k)
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try ranks
    for target_rank in range(30, 61):
        if time.time() > deadline - 20.0:
            break
        
        max_trials = 50
        for trial in range(max_trials):
            if time.time() > deadline - 10.0:
                break
            
            np.random.seed(trial * 1000 + target_rank)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            converged = False
            for iteration in range(100):
                if time.time() > deadline - 5.0:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Build A (m*p x rank) and b (m*p x 1)
                        A = np.zeros((m*p, target_rank))
                        b = np.zeros(m*p)
                        for jp in range(m):
                            for k in range(p):
                                idx = jp*p + k
                                b[idx] = 1.0 if jp == j else 0.0
                                for c in range(target_rank):
                                    A[idx, c] = V[jp*p+k, c] * W[k*n+i, c]
                        
                        try:
                            ATA = A.T @ A
                            ATb = A.T @ b
                            U[i*m+j, :] = np.linalg.solve(ATA + 1e-10*np.eye(target_rank), ATb)
                        except:
                            pass
                
                # Update V
                for jp in range(m):
                    for k in range(p):
                        A = np.zeros((n*m, target_rank))
                        b = np.zeros(n*m)
                        for i in range(n):
                            for j in range(m):
                                idx = i*m + j
                                b[idx] = 1.0 if j == jp else 0.0
                                for c in range(target_rank):
                                    A[idx, c] = U[i*m+j, c] * W[k*n+i, c]
                        
                        try:
                            ATA = A.T @ A
                            ATb = A.T @ b
                            V[jp*p+k, :] = np.linalg.solve(ATA + 1e-10*np.eye(target_rank), ATb)
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m*p, target_rank))
                        b = np.zeros(m*p)
                        for jp in range(m):
                            for j in range(m):
                                idx = jp*p + k # Wait, V index is j*p+k. Here j is the row of A?
                                # V[j*p+k, c]. The index in V is j*p+k.
                                # In the update for W, we fix k, i.
                                # We sum over j, j'.
                                # Term: U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c]
                                # We are solving for W[k*n+i, c].
                                # The equation is sum_c ... = delta_{j j'}
                                # For fixed k, i, we have equations for all j, j'.
                                # Number of equations: m*m.
                                # A should be (m*m, rank).
                                pass # Re-do this block correctly below
                
                # Correct Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m*m, target_rank))
                        b = np.zeros(m*m)
                        eq_idx = 0
                        for j in range(m):
                            for jp in range(m):
                                b[eq_idx] = 1.0 if j == jp else 0.0
                                for c in range(target_rank):
                                    A[eq_idx, c] = U[i*m+j, c] * V[jp*p+k, c]
                                eq_idx += 1
                        
                        try:
                            ATA = A.T @ A
                            ATb = A.T @ b
                            W[k*n+i, :] = np.linalg.solve(ATA + 1e-10*np.eye(target_rank), ATb)
                        except:
                            pass
                
                # Check error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for jp in range(m):
                                val = 0.0
                                for c in range(target_rank):
                                    val += U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c]
                                target = 1.0 if j == jp else 0.0
                                error += (val - target) ** 2
                
                if error < 1e-10:
                    best_result = (U, V, W)
                    best_rank = target_rank
                    converged = True
                    break
        
        if converged:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_result[0]]
    V_list = [[float(x) for x in row] for row in best_result[1]]
    W_list = [[float(x) for x in row] for row in best_result[2]]
    
    # Snap to half-integers
    for mat in [U_list, V_list, W_list]:
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                val = mat[i][j]
                half_int = round(val * 2) / 2
                if abs(val - half_int) < 1e-6:
                    mat[i][j] = half_int
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
