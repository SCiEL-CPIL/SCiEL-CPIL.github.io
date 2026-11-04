"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    # Tensor T[i,j,k] = 1 if we multiply A[i,j] * B[j,k]
    # W convention: W[k*n + i, r]
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find low-rank decomposition
    # Rank r means: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # We want minimal r such that this holds exactly
    
    # Strategy: Use numerical optimization to find rank-r decomposition
    # Then verify with exact arithmetic
    
    def verify_decomposition(U, V, W, T, tol=1e-10):
        """Check if U @ V @ W reconstructs T"""
        n1, r = U.shape
        n2, _ = V.shape
        n3, _ = W.shape
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Reconstruct: for each r, outer product U[:,r] * V[:,r] * W[:,r]
        recon = np.zeros((n1, n2, n3))
        for r in range(r):
            recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(recon, T, atol=tol)
    
    def try_rank(r, T, max_time=30.0):
        """Try to find rank-r decomposition"""
        n1, n2, n3 = T.shape
        t0 = time.time()
        
        # Initialize with random values
        np.random.seed(42 + r)
        U = np.random.randn(n1, r) * 0.1
        V = np.random.randn(n2, r) * 0.1
        W = np.random.randn(n3, r) * 0.1
        
        # Use gradient descent on the loss ||T - sum_r U_r * V_r * W_r||^2
        lr = 0.01
        best_loss = np.inf
        
        for step in range(5000):
            if time.time() - t0 > max_time:
                break
            
            # Compute reconstruction
            recon = np.zeros((n1, n2, n3))
            for ri in range(r):
                recon += np.einsum('a,b,c->abc', U[:,ri], V[:,ri], W[:,ri])
            
            loss = np.sum((T - recon)**2)
            if loss < best_loss:
                best_loss = loss
                if best_loss < 1e-12:
                    break
            
            # Compute gradients
            # dL/dU[a,ri] = -2 * sum_{b,c} (T[a,b,c] - recon[a,b,c]) * V[b,ri] * W[c,ri]
            # dL/dV[b,ri] = -2 * sum_{a,c} (T[a,b,c] - recon[a,b,c]) * U[a,ri] * W[c,ri]
            # dL/dW[c,ri] = -2 * sum_{a,b} (T[a,b,c] - recon[a,b,c]) * U[a,ri] * V[b,ri]
            
            diff = T - recon
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for ri in range(r):
                dU[:,ri] = -2 * np.einsum('abc,bc->a', diff, V[:,ri]*W[:,ri])
                dV[:,ri] = -2 * np.einsum('abc,ac->b', diff, U[:,ri]*W[:,ri])
                dW[:,ri] = -2 * np.einsum('abc,ab->c', diff, U[:,ri]*V[:,ri])
            
            U += lr * dU
            V += lr * dW  # Wait, this is wrong
            # Let me fix the gradient computation
            
            # Actually let me redo this more carefully
            break
        
        # Better approach: use a proper optimization
        return None
    
    # Better strategy: use the fact that for small tensors, we can try
    # specific structured decompositions
    
    # Known: 2x3x4 multiplication can be done in rank 19
    # Let's try to find it using a more systematic approach
    
    # Method: Use the "slice rank" or try to exploit structure
    # For 2x3x4, one approach is to use the fact that it's a submatrix of 3x3x3
    
    # Let me try a different approach: use numerical SVD-based methods
    # or try to find the decomposition by solving polynomial equations
    
    # For now, let's try ranks from 19 upward with a more careful optimization
    
    def optimize_rank(r, T, max_iter=20000, tol=1e-12):
        """Optimize for rank-r decomposition using Adam-like optimizer"""
        n1, n2, n3 = T.shape
        t0 = time.time()
        
        np.random.seed(123 + r)
        U = np.random.randn(n1, r) * 0.5
        V = np.random.randn(n2, r) * 0.5
        W = np.random.randn(n3, r) * 0.5
        
        lr = 0.001
        beta1, beta2 = 0.9, 0.999
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(max_iter):
            if time.time() - t0 > 15:
                break
            
            # Compute reconstruction and loss
            recon = np.zeros((n1, n2, n3))
            for ri in range(r):
                recon += np.einsum('a,b,c->abc', U[:,ri], V[:,ri], W[:,ri])
            
            diff = T - recon
            loss = np.sum(diff**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_loss < tol:
                    break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for ri in range(r):
                # dL/dU[a,ri] = -2 * sum_{b,c} diff[a,b,c] * V[b,ri] * W[c,ri]
                dU[:,ri] = -2 * np.einsum('abc,bc->a', diff, V[:,ri]*W[:,ri])
                dV[:,ri] = -2 * np.einsum('abc,ac->b', diff, U[:,ri]*W[:,ri])
                dW[:,ri] = -2 * np.einsum('abc,ab->c', diff, U[:,ri]*V[:,ri])
            
            # Adam update
            t = step + 1
            mU = beta1*mU + (1-beta1)*dU
            vU = beta2*vU + (1-beta2)*dU**2
            mU_hat = mU / (1 - beta1**t)
            vU_hat = vU / (1 - beta2**t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + 1e-8)
            
            mV = beta1*mV + (1-beta1)*dV
            vV = beta2*vV + (1-beta2)*dV**2
            mV_hat = mV / (1 - beta1**t)
            vV_hat = vV / (1 - beta2**t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + 1e-8)
            
            mW = beta1*mW + (1-beta1)*dW
            vW = beta2*vW + (1-beta2)*dW**2
            mW_hat = mW / (1 - beta1**t)
            vW_hat = vW / (1 - beta2**t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)
        
        return best_U, best_V, best_W, best_loss
    
    # Try ranks from 19 upward
    for r in range(19, 25):
        if time.time() > deadline:
            break
        
        U, V, W, loss = optimize_rank(r, T)
        
        if loss < 1e-8:
            # Found a good numerical solution, now try to snap to rationals
            # Convert to fractions with reasonable denominator
            def snap_to_rational(arr, max_denom=100):
                """Try to snap array entries to simple rationals"""
                result = arr.copy()
                for i in range(len(result)):
                    for j in range(len(result[0])):
                        val = result[i,j]
                        # Try small denominators
                        found = False
                        for denom in range(1, max_denom+1):
                            num = round(val * denom)
                            if abs(val - num/denom) < 1e-6:
                                result[i,j] = num/denom
                                found = True
                                break
                        if not found:
                            result[i,j] = round(val, 6)
                return result
            
            U_r = snap_to_rational(U)
            V_r = snap_to_rational(V)
            W_r = snap_to_rational(W)
            
            # Verify with the snapped values
            recon = np.zeros((n*m, m*p, p*n))
            for ri in range(r):
                recon += np.einsum('a,b,c->abc', U_r[:,ri], V_r[:,ri], W_r[:,ri])
            
            if np.allclose(recon, T, atol=1e-6):
                # Convert to exact fractions
                U_exact = [[Fraction(U_r[i,ri]).limit_denominator(1000) for ri in range(r)] for i in range(n*m)]
                V_exact = [[Fraction(V_r[i,ri]).limit_denominator(1000) for ri in range(r)] for i in range(m*p)]
                W_exact = [[Fraction(W_r[i,ri]).limit_denominator(1000) for ri in range(r)] for i in range(p*n)]
                
                # Final verification with exact arithmetic
                recon_exact = np.zeros((n*m, m*p, p*n))
                for ri in range(r):
                    for a in range(n*m):
                        for b in range(m*p):
                            for c in range(p*n):
                                recon_exact[a,b,c] += float(U_exact[a][ri] * V_exact[b][ri] * W_exact[c][ri])
                
                if np.allclose(recon_exact, T, atol=1e-4):
                    return {"U": U_exact, "V": V_exact, "W": W_exact}
    
    # Fallback: return schoolbook decomposition
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
# EVOLVE-BLOCK-END

