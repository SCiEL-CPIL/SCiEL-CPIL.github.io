"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def _construct_target_tensor(n, m, p):
    """Construct the target multiplication tensor T."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_fast(U, V, W, T, tol=1e-8):
    """
    Verify if U, V, W decompose T.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=tol)

def _try_rational_snapping(U, V, W, T, max_denom=8):
    """
    Try to snap the float solution to rational numbers with small denominators.
    Returns True if a snapped version verifies exactly, and updates U, V, W in place.
    """
    # Try denominators 1, 2, 4, 8
    for denom in [1, 2, 4, 8]:
        step = 1.0 / denom
        U_snap = np.round(U / step) * step
        V_snap = np.round(V / step) * step
        W_snap = np.round(W / step) * step
        
        if _verify_fast(U_snap, V_snap, W_snap, T, tol=1e-9):
            U[:] = U_snap
            V[:] = V_snap
            W[:] = W_snap
            return True
            
    return False

def _als_polish(U, V, W, T, n, m, p, iterations=20):
    """
    Polish the solution using Alternating Least Squares.
    """
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    for _ in range(iterations):
        # Solve for U
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed V, W:
        # U = T_mat @ V_mat @ (V_mat^T V_mat)^-1
        T_mat = T.reshape(nm, mp*pn)
        
        V_mat = np.zeros((mp*pn, r))
        for b in range(mp):
            for c in range(pn):
                V_mat[b*pn+c, :] = V[b, :] * W[c, :]
                
        try:
            VtV = V_mat.T @ V_mat
            VtV += 1e-6 * np.eye(r)
            U = T_mat @ V_mat @ np.linalg.inv(VtV)
        except np.linalg.LinAlgError:
            pass
            
        # Solve for V
        T_mat_v = T.reshape(mp, nm*pn)
        U_mat = np.zeros((nm*pn, r))
        for a in range(nm):
            for c in range(pn):
                U_mat[a*pn+c, :] = U[a, :] * W[c, :]
        try:
            UtU = U_mat.T @ U_mat
            UtU += 1e-6 * np.eye(r)
            V = T_mat_v @ U_mat @ np.linalg.inv(UtU)
        except np.linalg.LinAlgError:
            pass
            
        # Solve for W
        T_mat_w = T.reshape(pn, nm*mp)
        UW_mat = np.zeros((nm*mp, r))
        for a in range(nm):
            for b in range(mp):
                UW_mat[a*mp+b, :] = U[a, :] * V[b, :]
        try:
            UWtUW = UW_mat.T @ UW_mat
            UWtUW += 1e-6 * np.eye(r)
            W = T_mat_w @ UW_mat @ np.linalg.inv(UWtUW)
        except np.linalg.LinAlgError:
            pass

    return U, V, W

def _structured_init(n, m, p, rank, seed):
    """
    Generate structured initializations based on the tensor's inherent structure.
    For 2x5x6, exploit the small n=2 dimension.
    """
    nm, mp, pn = n*m, m*p, p*n
    np.random.seed(seed)
    
    # Strategy 1: Initialize with block structure
    # For n=2, we can think of C = [row1; row2] where each row is a linear combination
    U = np.random.randn(nm, rank) * 0.1
    V = np.random.randn(mp, rank) * 0.1
    W = np.random.randn(pn, rank) * 0.1
    
    # Strategy 2: Sparse initialization - each rank term activates a few entries
    if seed % 2 == 1:
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for r in range(rank):
            # Each rank term connects a subset of (i,j) to (j,k) to (k,i)
            i1 = np.random.randint(0, n)
            i2 = np.random.randint(0, n) if np.random.rand() > 0.5 else i1
            js = np.random.choice(m, size=min(m, np.random.randint(1, m+1)), replace=False)
            ks = np.random.choice(p, size=min(p, np.random.randint(1, p+1)), replace=False)
            
            for j in js:
                for k in ks:
                    U[i1*m+j, r] = np.random.choice([-1, 1])
                    V[j*p+k, r] = np.random.choice([-1, 1])
                    W[k*n+i1, r] = np.random.choice([-1, 1])
                    if i2 != i1:
                        U[i2*m+j, r] = np.random.choice([-1, 1]) * 0.5
                        W[k*n+i2, r] = np.random.choice([-1, 1]) * 0.5
    
    # Strategy 3: Use known small-rank patterns for n=2
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    # Try to factor these two sums with shared B terms
    if seed % 3 == 0:
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Assign some ranks to row 0, some to row 1, some to both
        r0 = rank // 3
        r1 = rank // 3
        r_both = rank - r0 - r1
        
        for r in range(r0):
            js = np.random.choice(m, size=np.random.randint(1, m+1), replace=False)
            for j in js:
                U[0*m+j, r] = np.random.choice([-1, 1])
                V[j*p+np.random.randint(0,p), r] = np.random.choice([-1, 1])
                W[np.random.randint(0,p)*n+0, r] = np.random.choice([-1, 1])
        
        for r in range(r0, r0+r1):
            js = np.random.choice(m, size=np.random.randint(1, m+1), replace=False)
            for j in js:
                U[1*m+j, r] = np.random.choice([-1, 1])
                V[j*p+np.random.randint(0,p), r] = np.random.choice([-1, 1])
                W[np.random.randint(0,p)*n+1, r] = np.random.choice([-1, 1])
        
        for r in range(r0+r1, rank):
            js = np.random.choice(m, size=np.random.randint(1, m+1), replace=False)
            for j in js:
                U[0*m+j, r] = np.random.choice([-1, 1])
                U[1*m+j, r] = np.random.choice([-1, 1])
                V[j*p+np.random.randint(0,p), r] = np.random.choice([-1, 1])
                W[np.random.randint(0,p)*n+0, r] = np.random.choice([-1, 1])
                W[np.random.randint(0,p)*n+1, r] = np.random.choice([-1, 1])
    
    return U, V, W

def _gradient_descent_with_restart(U, V, W, T, n, m, p, rank, deadline, max_steps=500):
    """
    Run gradient descent with adaptive learning rate and early stopping.
    Returns (found, U, V, W) where found is True if a valid decomposition was found.
    """
    nm, mp, pn = n*m, m*p, p*n
    
    v = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
    m_ = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
    beta1, beta2 = 0.9, 0.999
    lr = 0.02
    t = 0
    best_loss = np.inf
    found = False
    
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
            
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T - recon
        loss = np.sum(R**2)
        
        if loss < best_loss:
            best_loss = loss
            if loss < 1e-10:
                found = True
                break
                
        # Gradients
        gU = -np.einsum('abc,br,cr->ar', R, V, W)
        gV = -np.einsum('abc,ar,cr->br', R, U, W)
        gW = -np.einsum('abc,ar,br->cr', R, U, V)
        
        grads = [gU, gV, gW]
        params = [U, V, W]
        
        t += 1
        for i in range(3):
            m_[i] = beta1 * m_[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)
            m_hat = m_[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)
            params[i] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            
        U = params[0]
        V = params[1]
        W = params[2]
        
        # Adaptive learning rate: decay when loss plateaus
        if step % 100 == 0 and step > 0:
            if best_loss > 1e-6:
                lr *= 0.7
            else:
                lr *= 0.5
        
        # Check for convergence
        if step % 20 == 0 and loss < 1e-8:
            if _verify_fast(U, V, W, T, tol=1e-9):
                found = True
                break
                
    return found, U, V, W

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T = _construct_target_tensor(n, m, p)
    
    # Schoolbook fallback (Rank 60)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try ranks from 30 to 59
    for rank in range(30, best_rank):
        if time.time() > deadline - 30:
            break
            
        found = False
        max_restarts = 15  # Reduced to allow more time per restart
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Use structured initialization
            U, V, W = _structured_init(n, m, p, rank, seed=42 + restart * 7)
            
            # Run gradient descent
            found, U, V, W = _gradient_descent_with_restart(U, V, W, T, n, m, p, rank, deadline, max_steps=400)
            
            if found:
                # Polish with ALS
                U, V, W = _als_polish(U, V, W, T, n, m, p, iterations=15)
                if _verify_fast(U, V, W, T, tol=1e-8):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    _try_rational_snapping(U, V, W, T)
                    break

        if found:
            break

    U_final = best_U
    V_final = best_V
    W_final = best_W
    
    U_list = [[float(x) for x in row] for row in U_final]
    V_list = [[float(x) for x in row] for row in V_final]
    W_list = [[float(x) for x in row] for row in W_final]
    
    return {"U": U_list, "V": V_list, "W": W_list}
