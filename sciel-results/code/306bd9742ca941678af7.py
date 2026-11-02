"""Improved program for matrix multiplication 2x5x6 tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Schoolbook decomposition (rank 60) as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    def is_halfinteger(val):
        v = float(val)
        return abs(v * 2 - round(v * 2)) < 1e-6
    
    def to_halfint_list(mat):
        result = []
        for row in mat:
            r = []
            for val in row:
                v = float(val)
                half = round(v * 2) / 2.0
                r.append(half)
            result.append(r)
        return result
    
    def compute_residual(U, V, W):
        """Compute the residual tensor for the decomposition."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        residual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    residual[i, j, k] = val - 1.0
        return residual
    
    def gradient_descent(rank, max_iters=5000, lr=0.01, seed=42):
        """Use gradient descent to find a valid decomposition of given rank."""
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Try multiple initializations
        best_residual = np.inf
        best_U, best_V, best_W = None, None, None
        
        for init_trial in range(5):
            np.random.seed(seed + init_trial * 100)
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            for iteration in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Compute residual
                residual = compute_residual(U, V, W)
                res_norm = np.sum(residual ** 2)
                
                if res_norm < best_residual:
                    best_residual = res_norm
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if res_norm < 1e-12:
                    return U, V, W
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r = residual[i, j, k]
                            dU[i*m+j, :] += r * V[j*p+k, :] * W[k*n+i, :]
                            dV[j*p+k, :] += r * U[i*m+j, :] * W[k*n+i, :]
                            dW[k*n+i, :] += r * U[i*m+j, :] * V[j*p+k, :]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adaptive learning rate
                if res_norm < 1e-6:
                    lr *= 0.5
                elif res_norm > 1.0:
                    lr *= 0.5
                
                if iteration % 1000 == 0 and res_norm < 1e-4:
                    # Try to snap to half-integers and verify
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    if verify(U_snap, V_snap, W_snap):
                        return U_snap, V_snap, W_snap
            
            if best_residual < 1e-10:
                break
        
        if best_residual < 1e-10:
            # Try to snap to half-integers
            U_snap = np.round(best_U * 2) / 2
            V_snap = np.round(best_V * 2) / 2
            W_snap = np.round(best_W * 2) / 2
            if verify(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap
        
        return None
    
    def structured_search(rank, max_trials=20):
        """Try structured initializations for gradient descent."""
        for trial in range(max_trials):
            if time.time() > deadline - 15:
                break
            
            # Different initialization strategies
            if trial == 0:
                # Sparse random
                U = np.random.choice([-1, 0, 1], size=(n*m, rank))
                V = np.random.choice([-1, 0, 1], size=(m*p, rank))
                W = np.random.choice([-1, 0, 1], size=(p*n, rank))
            elif trial == 1:
                # Block structure
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for c in range(rank):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n*m)) % p
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
            elif trial == 2:
                # Strassen-like for 2x2 blocks
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                # Initialize with schoolbook pattern but with some sharing
                for c in range(min(rank, n*m*p)):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n*m)) % p
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
            else:
                # Random with varying density
                density = 0.2 + 0.1 * trial
                U = np.random.randn(n*m, rank)
                V = np.random.randn(m*p, rank)
                W = np.random.randn(p*n, rank)
                U[np.random.rand(*U.shape) < (1-density)] = 0
                V[np.random.rand(*V.shape) < (1-density)] = 0
                W[np.random.rand(*W.shape) < (1-density)] = 0
            
            result = gradient_descent(rank, max_iters=3000, lr=0.05, seed=trial*7)
            if result is not None:
                return result
        
        return None
    
    # Try to find a decomposition with rank < 60
    best_rank = schoolbook_rank
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Try ranks from lower to higher (we want the minimum)
    # Theoretical lower bound: at least max(n*m, m*p, p*n) / something
    # For 2x5x6: n*m=10, m*p=30, p*n=12, so rank >= 30 is a weak bound
    # But realistically, let's try from 40 up to 59
    
    candidate_ranks = list(range(40, schoolbook_rank))
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        result = structured_search(target_rank, max_trials=10)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_rank = target_rank
                best_decomp = (U, V, W)
                break  # Found the minimum, stop
    
    # If we found a better decomposition, use it
    if best_rank < schoolbook_rank:
        U, V, W = best_decomp
        return {"U": to_halfint_list(U), "V": to_halfint_list(V), "W": to_halfint_list(W)}
    
    # Otherwise, return schoolbook
    return {"U": to_halfint_list(U_sb), "V": to_halfint_list(V_sb), "W": to_halfint_list(W_sb)}
