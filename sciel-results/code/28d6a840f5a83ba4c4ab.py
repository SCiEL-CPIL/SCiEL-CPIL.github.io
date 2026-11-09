"""Optimized tensor decomposition search for 2x3 x 3x8 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 8]

def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p  # 48
    U_sb = np.zeros((n*m, rank_sb), dtype=np.float64)
    V_sb = np.zeros((m*p, rank_sb), dtype=np.float64)
    W_sb = np.zeros((p*n, rank_sb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    def verify_exact(U, V, W, tol=1e-9):
        """Check if the decomposition is exact."""
        Ur = U.reshape(n, m, -1)
        Vr = V.reshape(m, p, -1)
        Wr = W.reshape(p, n, -1)
        
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = Ur[i,j,:] * Vr[j,k,:] * Wr[k,i,:]
                    val = np.sum(prod)
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_round_to_half_int(U, V, W):
        """Round entries to nearest half-integer and verify."""
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        if verify_exact(U_r, V_r, W_r, tol=1e-8):
            return U_r, V_r, W_r, True
        return None, None, None, False
    
    def try_round_to_int(U, V, W):
        """Round entries to nearest integer and verify."""
        U_r = np.round(U).astype(np.float64)
        V_r = np.round(V).astype(np.float64)
        W_r = np.round(W).astype(np.float64)
        if verify_exact(U_r, V_r, W_r, tol=1e-8):
            return U_r, V_r, W_r, True
        return None, None, None, False
    
    def gradient_descent(R, max_iter=300, lr=0.05, seed=None, init=None):
        """Run gradient descent to find a decomposition of rank R."""
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        
        if init is not None:
            U, V, W = init
        else:
            U = rng.standard_normal((n*m, R))
            V = rng.standard_normal((m*p, R))
            W = rng.standard_normal((p*n, R))
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            Ur = U.reshape(n, m, R)
            Vr = V.reshape(m, p, R)
            Wr = W.reshape(p, n, R)
            
            # Compute S[i,j,k] = sum_r Ur[i,j,r]*Vr[j,k,r]*Wr[k,i,r]
            S = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        S[i,j,k] = np.sum(Ur[i,j,:] * Vr[j,k,:] * Wr[k,i,:])
            
            loss = np.sum((S - 1.0)**2)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-12:
                break
            
            # Compute gradients
            dU = np.zeros_like(Ur)
            dV = np.zeros_like(Vr)
            dW = np.zeros_like(Wr)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = 2*(S[i,j,k] - 1.0)
                        dU[i,j,:] += err * (Vr[j,k,:] * Wr[k,i,:])
                        dV[j,k,:] += err * (Ur[i,j,:] * Wr[k,i,:])
                        dW[k,i,:] += err * (Ur[i,j,:] * Vr[j,k,:])
            
            # Update with learning rate
            U -= lr * dU.reshape(n*m, R)
            V -= lr * dV.reshape(m*p, R)
            W -= lr * dW.reshape(p*n, R)
            
            # Adaptive learning rate
            if loss < best_loss * 0.5:
                lr = min(lr * 1.5, 0.5)
            elif loss > best_loss * 2:
                lr = max(lr * 0.5, 0.001)
        
        return best_U, best_V, best_W, best_loss
    
    def polish(U, V, W, max_iter=500, tol=1e-14):
        """High-precision polish pass with tighter tolerance."""
        R = U.shape[1]
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        lr = 0.01
        
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            Ur = U.reshape(n, m, R)
            Vr = V.reshape(m, p, R)
            Wr = W.reshape(p, n, R)
            
            S = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        S[i,j,k] = np.sum(Ur[i,j,:] * Vr[j,k,:] * Wr[k,i,:])
            
            loss = np.sum((S - 1.0)**2)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < tol:
                break
            
            dU = np.zeros_like(Ur)
            dV = np.zeros_like(Vr)
            dW = np.zeros_like(Wr)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = 2*(S[i,j,k] - 1.0)
                        dU[i,j,:] += err * (Vr[j,k,:] * Wr[k,i,:])
                        dV[j,k,:] += err * (Ur[i,j,:] * Wr[k,i,:])
                        dW[k,i,:] += err * (Ur[i,j,:] * Vr[j,k,:])
            
            U -= lr * dU.reshape(n*m, R)
            V -= lr * dV.reshape(m*p, R)
            W -= lr * dW.reshape(p*n, R)
            
            if loss < best_loss * 0.1:
                lr = min(lr * 1.3, 0.1)
            elif loss > best_loss * 3:
                lr = max(lr * 0.5, 0.0001)
        
        return best_U, best_V, best_W, best_loss
    
    # Try decreasing ranks from lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    ranks_to_try = list(range(24, 48))
    
    for R in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        # More attempts for lower ranks
        max_attempts = max(3, 40 - (R - 24))
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            seed = R * 1000 + attempt
            
            # Multiple initialization strategies
            if attempt % 3 == 0:
                # Standard random
                U, V, W, loss = gradient_descent(R, max_iter=200, lr=0.03, seed=seed)
            elif attempt % 3 == 1:
                # Scaled random
                rng = np.random.default_rng(seed + 7)
                U = rng.standard_normal((n*m, R)) * 0.5
                V = rng.standard_normal((m*p, R)) * 0.5
                W = rng.standard_normal((p*n, R)) * 0.5
                U, V, W, loss = gradient_descent(R, max_iter=200, lr=0.05, seed=None, init=(U, V, W))
            else:
                # Sparse-like initialization
                rng = np.random.default_rng(seed + 13)
                U = rng.standard_normal((n*m, R))
                V = rng.standard_normal((m*p, R))
                W = rng.standard_normal((p*n, R))
                # Zero out some entries to encourage sparsity
                mask = rng.random((n*m, R)) < 0.3
                U[mask] = 0
                mask = rng.random((m*p, R)) < 0.3
                V[mask] = 0
                mask = rng.random((p*n, R)) < 0.3
                W[mask] = 0
                U, V, W, loss = gradient_descent(R, max_iter=200, lr=0.04, seed=None, init=(U, V, W))
            
            if loss < 1e-8:
                # Polish with tighter tolerance
                U, V, W, loss = polish(U, V, W, max_iter=300, tol=1e-14)
                
                # Try rounding to half-integers
                Ur, Vr, Wr, success = try_round_to_half_int(U, V, W)
                if success:
                    best_rank = R
                    best_U, best_V, best_W = Ur, Vr, Wr
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                
                # Try rounding to integers
                Ur, Vr, Wr, success = try_round_to_int(U, V, W)
                if success:
                    best_rank = R
                    best_U, best_V, best_W = Ur, Vr, Wr
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            
            # Also check if close to half-integers
            if loss < 1e-6:
                Ur, Vr, Wr, success = try_round_to_half_int(U, V, W)
                if success:
                    best_rank = R
                    best_U, best_V, best_W = Ur, Vr, Wr
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # If we found a better rank but couldn't round exactly, still return best
    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Return schoolbook solution
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
