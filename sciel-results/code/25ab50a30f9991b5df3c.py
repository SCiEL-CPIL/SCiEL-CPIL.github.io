import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Target tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # Constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def verify(U, V, W):
        """Verify decomposition using Fractions for exactness."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if s != 1:
                        return False
        return True
    
    # Start with schoolbook
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    def try_find_decomposition(target_rank, max_restarts=30):
        """Try to find a rank=target_rank decomposition using vectorized gradient descent."""
        np.random.seed(42)
        for attempt in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize with random values
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Precompute index mappings
            # For each (i,j,k), we need U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
            idx_U = np.array([i*m+j for i in range(n) for j in range(m) for k in range(p)])
            idx_V = np.array([j*p+k for i in range(n) for j in range(m) for k in range(p)])
            idx_W = np.array([k*n+i for i in range(n) for j in range(m) for k in range(p)])
            
            # Vectorized gradient descent
            lr = 0.05
            for it in range(200):
                if time.time() > deadline:
                    return None
                
                # Compute predictions: pred[i,j,k] = sum_c U[idx_U[ijk],c] * V[idx_V[ijk],c] * W[idx_W[ijk],c]
                # For each (ijk, c): pred = sum_c U[idx_U[ijk],c] * V[idx_V[ijk],c] * W[idx_W[ijk],c]
                
                # Efficient computation:
                # U[idx_U] has shape (n*m*p, target_rank)
                # V[idx_V] has shape (n*m*p, target_rank)
                # W[idx_W] has shape (n*m*p, target_rank)
                
                U_sel = U[idx_U]  # (n*m*p, r)
                V_sel = V[idx_V]  # (n*m*p, r)
                W_sel = W[idx_W]  # (n*m*p, r)
                
                # pred[ijk] = sum_c U_sel[ijk,c] * V_sel[ijk,c] * W_sel[ijk,c]
                pred = np.sum(U_sel * V_sel * W_sel, axis=1)  # (n*m*p,)
                err = pred - 1.0  # (n*m*p,)
                
                # Gradients:
                # dU[idx_U[ijk], c] += 2 * err[ijk] * V_sel[ijk,c] * W_sel[ijk,c]
                # dV[idx_V[ijk], c] += 2 * err[ijk] * U_sel[ijk,c] * W_sel[ijk,c]
                # dW[idx_W[ijk], c] += 2 * err[ijk] * U_sel[ijk,c] * V_sel[ijk,c]
                
                # We need to scatter-add gradients back to U, V, W
                # Since multiple (ijk) can map to same row, we need to accumulate
                
                # For U: for each ijk, add 2*err[ijk]*V_sel[ijk,:]*W_sel[ijk,:] to U[idx_U[ijk],:]
                # Build gradient arrays
                grad_U = np.zeros((n*m, target_rank))
                grad_V = np.zeros((m*p, target_rank))
                grad_W = np.zeros((p*n, target_rank))
                
                # Accumulate using np.add.at for repeated indices
                U_grad_contrib = 2.0 * err[:, None] * V_sel * W_sel  # (n*m*p, r)
                V_grad_contrib = 2.0 * err[:, None] * U_sel * W_sel  # (n*m*p, r)
                W_grad_contrib = 2.0 * err[:, None] * U_sel * V_sel  # (n*m*p, r)
                
                np.add.at(grad_U, idx_U, U_grad_contrib)
                np.add.at(grad_V, idx_V, V_grad_contrib)
                np.add.at(grad_W, idx_W, W_grad_contrib)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            # Check if we found a solution
            U_sel = U[idx_U]
            V_sel = V[idx_V]
            W_sel = W[idx_W]
            pred = np.sum(U_sel * V_sel * W_sel, axis=1)
            max_err = np.max(np.abs(pred - 1.0))
            
            if max_err < 1e-5:
                # Convert to fractions and verify
                U_frac = [[Fraction(U[i][c]).limit_denominator(100) for c in range(target_rank)] for i in range(n*m)]
                V_frac = [[Fraction(V[i][c]).limit_denominator(100) for c in range(target_rank)] for i in range(m*p)]
                W_frac = [[Fraction(W[i][c]).limit_denominator(100) for c in range(target_rank)] for i in range(p*n)]
                
                if verify(U_frac, V_frac, W_frac):
                    return U_frac, V_frac, W_frac
        
        return None
    
    # Search from low rank upward
    for target_rank in range(24, best_rank + 1):
        if time.time() > deadline:
            break
        result = try_find_decomposition(target_rank, max_restarts=20)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    # Return as lists of Fractions
    U_list = [[x for x in row] for row in best_U]
    V_list = [[x for x in row] for row in best_V]
    W_list = [[x for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
