"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _compute_T(U, V, W, n, m, p, rank):
    """Compute the reconstructed tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]."""
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        T += u[:, :, None] * v[None, :, :] * w.T[None, :, :]  # w.T is (n, p), so w.T[None,:,:] is (1,n,p) -> need (n,p) indexed as [k,i]
        # Actually w has shape (p, n), w[k,i] means row k, col i.
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        # Let me redo this properly
        pass
    
    # Redo with explicit loops for correctness, but vectorized over c
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return T

def _compute_T_fast(U, V, W, n, m, p, rank):
    """Faster computation using einsum."""
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        # w is (p,n), we need w[k,i] which is w.T[i,k]
        # So T += u[:,:,None] * v[None,:,:] * w.T[None,:,:]
        # u[:,:,None] is (n,m,1), v[None,:,:] is (1,m,p), w.T[None,:,:] is (1,n,p)
        # Broadcasting: (n,m,p)
        T += u[:, :, None] * v[None, :, :] * w.T[None, :, :]
    return T

def _compute_loss(U, V, W, n, m, p, rank):
    T = _compute_T_fast(U, V, W, n, m, p, rank)
    return np.sum((T - 1.0) ** 2)

def _compute_gradients(U, V, W, n, m, p, rank):
    """Compute gradients of loss = ||T - 1||^2 with respect to U, V, W."""
    T = _compute_T_fast(U, V, W, n, m, p, rank)
    err = T - 1.0  # (n, m, p)
    
    grad_U = np.zeros((n * m, rank), dtype=float)
    grad_V = np.zeros((m * p, rank), dtype=float)
    grad_W = np.zeros((p * n, rank), dtype=float)
    
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        
        # grad_U[i,j,c] = 2 * sum_k err[i,j,k] * v[j,k] * w[k,i]
        # = 2 * (err[:,j,:] * v[j,:] * w.T[:,j]) summed over k... 
        # Let's be more careful:
        # grad_U[i,j,c] = 2 * sum_k err[i,j,k] * v[j,k] * w[k,i]
        # For fixed i,j: sum_k err[i,j,k] * v[j,k] * w[k,i]
        # = sum_k err[i,j,k] * v[j,k] * w[k,i]
        # w[k,i] for fixed i is the i-th column of w, which is w[:,i]
        # So: sum_k err[i,j,k] * v[j,k] * w[:,i][k] = sum_k err[i,j,k] * v[j,k] * w[k,i]
        
        # Vectorized: for each c
        # grad_U[:, c] reshaped to (n,m): grad_U[i,j] = 2 * sum_k err[i,j,k] * v[j,k] * w[k,i]
        # = 2 * einsum('ijk,jk,ki->ij', err, v, w)
        
        grad_U[:, c] = 2.0 * np.einsum('ijk,jk,ki->ij', err, v, w).reshape(n * m)
        
        # grad_V[j,k,c] = 2 * sum_i err[i,j,k] * u[i,j] * w[k,i]
        # = 2 * einsum('ijk,ij,ki->jk', err, u, w)
        grad_V[:, c] = 2.0 * np.einsum('ijk,ij,ki->jk', err, u, w).reshape(m * p)
        
        # grad_W[k,i,c] = 2 * sum_j err[i,j,k] * u[i,j] * v[j,k]
        # = 2 * einsum('ijk,ij,jk->ki', err, u, v)
        grad_W[:, c] = 2.0 * np.einsum('ijk,ij,jk->ki', err, u, v).reshape(p * n)
    
    return grad_U, grad_V, grad_W

def _gradient_descent(U, V, W, n, m, p, rank, deadline, lr_init=0.05, max_iter=500):
    """Run gradient descent to minimize loss."""
    lr = lr_init
    best_loss = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        loss = _compute_loss(U, V, W, n, m, p, rank)
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if loss < 1e-14:
            break
        
        grad_U, grad_V, grad_W = _compute_gradients(U, V, W, n, m, p, rank)
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Adaptive learning rate
        new_loss = _compute_loss(U, V, W, n, m, p, rank)
        if new_loss > loss * 1.1:
            # Rejected, reduce lr
            lr *= 0.8
            U = best_U.copy()
            V = best_V.copy()
            W = best_W.copy()
        else:
            lr = min(lr * 1.01, 0.5)
    
    return best_U, best_V, best_W, best_loss

def _newton_polish(U, V, W, n, m, p, rank, deadline, max_iter=20):
    """Newton-style polish: use second-order information implicitly via line search."""
    lr = 0.01
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        loss = _compute_loss(U, V, W, n, m, p, rank)
        if loss < 1e-16:
            break
        
        grad_U, grad_V, grad_W = _compute_gradients(U, V, W, n, m, p, rank)
        
        # Simple backtracking line search
        step = lr
        for _ in range(10):
            U_new = U - step * grad_U
            V_new = V - step * grad_V
            W_new = W - step * grad_W
            new_loss = _compute_loss(U_new, V_new, W_new, n, m, p, rank)
            if new_loss < loss:
                U, V, W = U_new, V_new, W_new
                loss = new_loss
                break
            step *= 0.5
        
        lr *= 0.9
    
    return U, V, W, loss

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    deadline = time.time() + 130  # Leave 20s margin
    
    # Initialize with schoolbook
    best_rank = schoolbook_rank
    best_U = np.zeros((n * m, schoolbook_rank), dtype=float)
    best_V = np.zeros((m * p, schoolbook_rank), dtype=float)
    best_W = np.zeros((p * n, schoolbook_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # The known lower bound for matrix multiplication rank is around (n+m+p)^2 / something
    # For 2x5x6, let's try ranks from 25 upward
    found = False
    for rank in range(25, schoolbook_rank):
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        max_restarts = 3
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(seed=restart * 1000 + rank)
            
            # Random initialization with small scale
            scale = 1.0 / np.sqrt(rank)
            U = rng.randn(n * m, rank) * scale
            V = rng.randn(m * p, rank) * scale
            W = rng.randn(p * n, rank) * scale
            
            # Run gradient descent
            U, V, W, loss = _gradient_descent(U, V, W, n, m, p, rank, deadline, lr_init=0.1, max_iter=400)
            
            if loss < 1e-10:
                # Polish
                U, V, W, loss = _newton_polish(U, V, W, n, m, p, rank, deadline, max_iter=30)
            
            if loss < 1e-9:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
        
        if found:
            break
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

