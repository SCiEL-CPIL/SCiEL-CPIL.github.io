"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication.
    
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    
    Condition: For all i, k, sum_j A[i,j] B[j,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
    This implies: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = delta[j, j'] for all j, j'?
    Actually, the condition derived was:
    sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] = delta[j, j']
    """
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    R = U.shape[1]
    
    for i in range(n):
        for k in range(p):
            # Construct the matrix M[j, j'] = sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r]
            M = np.zeros((m, m))
            for r in range(R):
                w_val = W[k * n + i, r]
                if w_val == 0: continue
                u_vec = U[i * m : (i + 1) * m, r] # shape (m,)
                v_vec = V[:, r] # shape (m*p,)
                # Extract v_vec for column k
                # V is indexed by j*p + k. So for fixed k, the indices are 0*p+k, 1*p+k, ...
                v_col = v_vec[k::p] # shape (m,)
                # M[j, j'] += w_val * u_vec[j] * v_col[j']
                M += w_val * np.outer(u_vec, v_col)
            
            # Check if M is Identity
            if not np.allclose(M, np.eye(m), atol=1e-9):
                return False
    return True

def _find_decomposition(n, m, p, target_rank, max_iters=2000, tol=1e-10):
    """Use Gradient Descent to find U, V, W for a given rank."""
    np.random.seed(42)
    R = target_rank
    
    # Initialize
    U = np.random.randn(n * m, R) * 0.1
    V = np.random.randn(m * p, R) * 0.1
    W = np.random.randn(p * n, R) * 0.1
    
    lr = 0.01
    
    for it in range(max_iters):
        # Compute Loss and Gradients
        # L = sum_{i,k} || sum_r W[k*n+i, r] u_{i,r} v_{k,r}^T - I_m ||_F^2
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        loss = 0.0
        
        for i in range(n):
            for k in range(p):
                # Current product for this i, k
                P = np.zeros((m, m))
                for r in range(R):
                    w_val = W[k * n + i, r]
                    u_vec = U[i * m : (i + 1) * m, r]
                    v_vec = V[k::p, r] # Extract column k from V[:, r]
                    P += w_val * np.outer(u_vec, v_vec)
                
                E = P - np.eye(m)
                loss += np.sum(E**2)
                
                if loss > 1e6: # Diverging
                    return None, None, None, loss
                
                # Gradients
                # dL/dW[k*n+i, r] = 2 * sum_{j,j'} E[j,j'] * u_vec[j] * v_vec[j']
                # dL/dU[i*m+j, r] = 2 * sum_k sum_{j'} E[j,j'] * W[k*n+i, r] * v_vec[j']
                # dL/dV[j'*p+k, r] = 2 * sum_i sum_j E[j,j'] * W[k*n+i, r] * u_vec[j]
                
                for r in range(R):
                    u_vec = U[i * m : (i + 1) * m, r]
                    v_vec = V[k::p, r]
                    w_val = W[k * n + i, r]
                    
                    # Grad W
                    gw = 2 * np.sum(E * np.outer(u_vec, v_vec))
                    grad_W[k * n + i, r] += gw
                    
                    # Grad U
                    # dP/dU[i*m+j, r] = w_val * e_j v_vec^T
                    # dL/dU = 2 * sum_{j',j} E[j,j'] * w_val * delta_{j,j} v_vec[j']
                    # = 2 * w_val * (E @ v_vec)
                    gu = 2 * w_val * (E @ v_vec)
                    grad_U[i * m : (i + 1) * m, r] += gu
                    
                    # Grad V
                    # dP/dV[j'*p+k, r] = w_val * u_vec e_{j'}^T
                    # dL/dV = 2 * sum_{j,j'} E[j,j'] * w_val * u_vec[j] delta_{j',j'}
                    # = 2 * w_val * (E.T @ u_vec)
                    gv = 2 * w_val * (E.T @ u_vec)
                    grad_V[k::p, r] += gv
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if loss < tol:
            break
            
    return U, V, W, loss

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Try ranks from low to high
    # Lower bound is roughly 14 for 2x4x4
    for R in range(14, 30):
        if time.time() > deadline:
            break
            
        U, V, W, loss = _find_decomposition(n, m, p, R)
        
        if U is None:
            continue
            
        # If loss is very small, try to snap to half-integers
        if loss < 1e-6:
            # Round to nearest half-integer
            U_half = np.round(U * 2) / 2
            V_half = np.round(V * 2) / 2
            W_half = np.round(W * 2) / 2
            
            # Verify exactness
            if _verify_decomposition(U_half, V_half, W_half, n, m, p):
                return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
            
            # If snapping failed, try rounding to integers
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if _verify_decomposition(U_int, V_int, W_int, n, m, p):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
            
    # Fallback: Schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
