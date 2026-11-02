import time
import numpy as np
from scipy.optimize import minimize

def _verify(U, V, W, n, m, p):
    """Verify decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    U3 = np.array(U, dtype=float).reshape(n, m, -1)
    V3 = np.array(V, dtype=float).reshape(m, p, -1)
    W3 = np.array(W, dtype=float).reshape(p, n, -1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank32():
    """
    Build the known rank-32 decomposition: 4 Strassen blocks (cols 0-7) + 1 schoolbook (col 8).
    """
    n, m, p = 2, 2, 9
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1: (a00+a11)(b_{0,k0}+b_{1,k1})
        U[0][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        # M2: (a10+a11) * b_{0,k0}
        U[2][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0
        W[k0*n + 1][col] = 1.0; W[k1*n + 1][col] = -1.0
        col += 1
        # M3: a00 * (b_{0,k1} - b_{1,k1})
        U[0][col] = 1.0
        V[0*p + k1][col] = 1.0; V[1*p + k1][col] = -1.0
        W[k1*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        # M4: a11 * (b_{1,k0} - b_{0,k0})
        U[3][col] = 1.0
        V[1*p + k0][col] = 1.0; V[0*p + k0][col] = -1.0
        W[k0*n + 0][col] = 1.0; W[k0*n + 1][col] = 1.0
        col += 1
        # M5: (a00+a01) * b_{1,k1}
        U[0][col] = 1.0; U[1][col] = 1.0
        V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = -1.0; W[k1*n + 0][col] = 1.0
        col += 1
        # M6: (a10-a00) * (b_{0,k0}+b_{0,k1})
        U[2][col] = 1.0; U[0][col] = -1.0
        V[0*p + k0][col] = 1.0; V[0*p + k1][col] = 1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        # M7: (a01-a11) * (b_{1,k0}+b_{1,k1})
        U[1][col] = 1.0; U[3][col] = -1.0
        V[1*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0
        col += 1

    def add_schoolbook(k):
        nonlocal col
        U[0][col] = 1.0
        V[0*p + k][col] = 1.0
        W[k*n + 0][col] = 1.0
        col += 1
        U[1][col] = 1.0
        V[1*p + k][col] = 1.0
        W[k*n + 0][col] = 1.0
        col += 1
        U[2][col] = 1.0
        V[0*p + k][col] = 1.0
        W[k*n + 1][col] = 1.0
        col += 1
        U[3][col] = 1.0
        V[1*p + k][col] = 1.0
        W[k*n + 1][col] = 1.0
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_schoolbook(8)
    
    return U, V, W

def _find_rank10_block(deadline):
    """
    Attempt to find a rank-10 decomposition for 2x2x3 matrix multiplication.
    Returns (U, V, W) as numpy arrays of shapes (4,10), (6,10), (6,10) or None.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # Initialize with a rank-11 structure (Strassen + 4 schoolbook) and remove one term, 
    # or just random initialization.
    # A better init: take the rank-12 schoolbook and randomize, or take rank-7 Strassen + 3 random.
    
    best_loss = float('inf')
    best_params = None
    
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        # Initialize U, V, W
        # Use a mix of Strassen-like and random
        U_init = np.random.randn(4, rank) * 0.5
        V_init = np.random.randn(6, rank) * 0.5
        W_init = np.random.randn(6, rank) * 0.5
        
        # Flatten for optimization
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def loss_func(x):
            U_b = x[:4*rank].reshape(4, rank)
            V_b = x[4*rank:4*rank+6*rank].reshape(6, rank)
            W_b = x[4*rank+6*rank:].reshape(6, rank)
            
            # Compute tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # n=2, m=2, p=3
            err = 0.0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        s = np.dot(U_b[i*2+j, :], V_b[j*3+k, :] * W_b[k*2+i, :])
                        err += (s - 1.0)**2
            return err
        
        # Use Nelder-Mead or BFGS
        # Nelder-Mead is derivative-free and robust for small problems
        try:
            res = minimize(loss_func, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_loss < 1e-9 and best_params is not None:
        U_b = best_params[:4*rank].reshape(4, rank)
        V_b = best_params[4*rank:4*rank+6*rank].reshape(6, rank)
        W_b = best_params[4*rank+6*rank:].reshape(6, rank)
        return U_b, V_b, W_b
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Start with the known rank-32 solution
    U32, V32, W32 = _build_rank32()
    
    if not _verify(U32, V32, W32, 2, 2, 9):
        # Fallback to full schoolbook rank 36
        rank_fb = 36
        U_fb = [[0.0] * rank_fb for _ in range(4)]
        V_fb = [[0.0] * rank_fb for _ in range(18)]
        W_fb = [[0.0] * rank_fb for _ in range(18)]
        c = 0
        for i in range(2):
            for j in range(2):
                for k in range(9):
                    U_fb[i*2+j][c] = 1.0
                    V_fb[j*9+k][c] = 1.0
                    W_fb[k*2+i][c] = 1.0
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    best_rank = 32
    best_U, best_V, best_W = U32, V32, W32
    
    # Try to find a rank-30 solution using three rank-10 blocks
    if time.time() < deadline - 30:
        block = _find_rank10_block(deadline - 30)
        if block is not None:
            U_b, V_b, W_b = block
            # Assemble the rank-30 solution
            rank_30 = 30
            U_30 = [[0.0] * rank_30 for _ in range(4)]
            V_30 = [[0.0] * rank_30 for _ in range(18)]
            W_30 = [[0.0] * rank_30 for _ in range(18)]
            
            for b in range(3):
                # Block b handles columns 3b, 3b+1, 3b+2
                offset = b * 10
                for i in range(4):
                    U_30[i][offset:offset+10] = U_b[i, :].tolist()
                
                for j in range(2):
                    for k in range(3):
                        v_idx = j * 9 + (3 * b + k)
                        w_idx = (3 * b + k) * 2 + i # Wait, W index is k*n+i
                        # V index in block: j*p_block + k_block = j*3 + k
                        # W index in block: k_block*n + i
                        # Map to global:
                        # V global: j*9 + (3b+k)
                        # W global: (3b+k)*2 + i
                        
                        # Fill V_30
                        V_30[v_idx][offset:offset+10] = V_b[j*3+k, :].tolist()
                        
                        # Fill W_30
                        w_g_idx = (3 * b + k) * 2 + i
                        W_30[w_g_idx][offset:offset+10] = W_b[k*2+i, :].tolist()
            
            if _verify(U_30, V_30, W_30, 2, 2, 9):
                best_U, best_V, best_W = U_30, V_30, W_30
                best_rank = 30
    
    return {"U": best_U, "V": best_V, "W": best_W}
