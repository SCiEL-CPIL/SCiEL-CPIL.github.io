"""Matrix multiplication 2,2,6 - rank-minimal exact tensor decomposition.

Strategy:
1. Build the multiplication tensor.
2. Try to find rank-r decompositions for r from a lower bound up to schoolbook-1.
3. For each rank, use multiple strategies:
   a. Random init + ALS (alternating least squares)
   b. Gradient-based optimization (L-BFGS or similar)
   c. Structured initialization exploiting tensor symmetry
4. Return the lowest-rank exact decomposition found, or schoolbook as fallback.
"""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Build the multiplication tensor
    T_full = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1.0
    
    # Precompute flattened views
    T_flat_U = T_full.reshape(nm, mp * pn)
    T_flat_V = T_full.transpose(1, 0, 2).reshape(mp, nm * pn)
    T_flat_W = T_full.transpose(2, 0, 1).reshape(pn, nm * mp)
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    def time_left():
        return deadline - time.time()
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-8):
        """Check if the decomposition is exact."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        max_err = 0.0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = np.dot(U[a], V[b] * W[c])
                    err = abs(val - T_full[a, b, c])
                    if err > max_err:
                        max_err = err
        return max_err < tol
    
    def compute_error(U, V, W):
        """Compute the Frobenius norm of the decomposition error."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        error = 0.0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = np.dot(U[a], V[b] * W[c])
                    error += (val - T_full[a, b, c]) ** 2
        return error
    
    def try_rank_als(rank, n_trials=20, seed=None):
        """Try to find a rank-r decomposition via random init + ALS."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        for trial in range(n_trials):
            if time_left() < 5:
                break
            
            # Random initialization with various scales
            scale = rng.choice([0.5, 1.0, 2.0])
            U = rng.randn(nm, rank) * scale
            V = rng.randn(mp, rank) * scale
            W = rng.randn(pn, rank) * scale
            
            # Alternating least squares
            converged = False
            for iteration in range(150):
                if time_left() < 5:
                    break
                
                # Update U
                M_U = np.zeros((mp * pn, rank))
                for b in range(mp):
                    for c in range(pn):
                        M_U[b * pn + c, :] = V[b, :] * W[c, :]
                
                try:
                    A_U = M_U.T @ M_U
                    B_U = M_U.T @ T_flat_U.T
                    U_new = np.linalg.solve(A_U, B_U).T
                except np.linalg.LinAlgError:
                    U_new = U + 0.01 * rng.randn(nm, rank)
                
                # Update V
                M_V = np.zeros((nm * pn, rank))
                for a in range(nm):
                    for c in range(pn):
                        M_V[a * pn + c, :] = U_new[a, :] * W[c, :]
                
                try:
                    A_V = M_V.T @ M_V
                    B_V = M_V.T @ T_flat_V.T
                    V_new = np.linalg.solve(A_V, B_V).T
                except np.linalg.LinAlgError:
                    V_new = V + 0.01 * rng.randn(mp, rank)
                
                # Update W
                M_W = np.zeros((nm * mp, rank))
                for a in range(nm):
                    for b in range(mp):
                        M_W[a * mp + b, :] = U_new[a, :] * V_new[b, :]
                
                try:
                    A_W = M_W.T @ M_W
                    B_W = M_W.T @ T_flat_W.T
                    W_new = np.linalg.solve(A_W, B_W).T
                except np.linalg.LinAlgError:
                    W_new = W + 0.01 * rng.randn(pn, rank)
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                error = compute_error(U, V, W)
                if error < 1e-10:
                    converged = True
                    break
            
            if converged:
                if verify(U, V, W):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
        
        return None
    
    def try_rank_gradient(rank, n_trials=5, seed=None):
        """Try to find a rank-r decomposition via gradient-based optimization."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        for trial in range(n_trials):
            if time_left() < 10:
                break
            
            # Random initialization
            U = rng.randn(nm, rank) * 0.5
            V = rng.randn(mp, rank) * 0.5
            W = rng.randn(pn, rank) * 0.5
            
            # Simple gradient descent with adaptive learning rate
            lr = 0.01
            best_error = compute_error(U, V, W)
            
            for iteration in range(500):
                if time_left() < 5:
                    break
                
                # Compute gradients
                dU = np.zeros((nm, rank))
                dV = np.zeros((mp, rank))
                dW = np.zeros((pn, rank))
                
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            err = np.dot(U[a], V[b] * W[c]) - T_full[a, b, c]
                            for r in range(rank):
                                dU[a, r] += err * V[b, r] * W[c, r]
                                dV[b, r] += err * U[a, r] * W[c, r]
                                dW[c, r] += err * U[a, r] * V[b, r]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                error = compute_error(U, V, W)
                if error < 1e-10:
                    if verify(U, V, W):
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                    break
                
                # Adapt learning rate
                if error > best_error:
                    lr *= 0.5
                else:
                    best_error = error
                    lr = min(lr * 1.1, 0.1)
            
            # Final check
            if verify(U, V, W):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        return None
    
    def try_rank_structured(rank, seed=None):
        """Try structured initialization exploiting tensor symmetry."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        # For 2x2 x 2x6, exploit the structure
        # The tensor has symmetries related to the matrix multiplication structure
        
        # Strategy: Start from a perturbed schoolbook-like structure
        # and try to compress it
        
        if rank >= 12:
            # Try a structured initialization based on the block structure
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            # Assign blocks based on the (i,j,k) structure
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < rank:
                            U[i * m + j, col] = 1.0
                            V[j * p + k, col] = 1.0
                            W[k * n + i, col] = 1.0
                            col += 1
            
            # Now try to optimize
            if time_left() > 10:
                result = try_rank_gradient(rank, n_trials=3, seed=seed)
                if result is not None:
                    return result
        
        return None
    
    # Try ranks from lower to higher
    start_rank = max(nm, mp, pn)  # = 12
    end_rank = n * m * p - 1  # = 23
    
    best_result = None
    for rank in range(start_rank, end_rank + 1):
        if time_left() < 15:
            break
        
        # Try ALS first (usually faster)
        result = try_rank_als(rank, n_trials=15, seed=rank * 1000)
        if result is not None:
            best_result = result
            break
        
        # Try gradient-based
        if time_left() > 20:
            result = try_rank_gradient(rank, n_trials=3, seed=rank * 2000)
            if result is not None:
                best_result = result
                break
        
        # Try structured
        if time_left() > 15:
            result = try_rank_structured(rank, seed=rank * 3000)
            if result is not None:
                best_result = result
                break
    
    # If no lower rank found, return schoolbook
    if best_result is None:
        best_result = schoolbook()
    
    return best_result
