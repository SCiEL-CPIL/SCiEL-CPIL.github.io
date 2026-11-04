import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] where
    a = i*m + j (row of A, col of A)
    b = j*p + k (row of B, col of B)
    c = k*n + i (row of C, col of C)
    T[a,b,c] = 1 iff the (i,j) entry of A, (j,k) entry of B, and (k,i) entry of C
    participate in the same product term.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _reconstruction(U, V, W):
    """Compute the tensor reconstructed from factor matrices."""
    return np.einsum('al,bl,cl->abc', U, V, W)

def _error(U, V, W, T):
    """Frobenius norm squared of the reconstruction error."""
    T_comp = _reconstruction(U, V, W)
    diff = T_comp - T
    return np.sum(diff * diff)

def _snap_to_half_integers(U, V, W, T, tol=1e-9):
    """Try to snap to half-integers and verify exactness."""
    U_r = np.round(U * 2) / 2.0
    V_r = np.round(V * 2) / 2.0
    W_r = np.round(W * 2) / 2.0
    T_comp = _reconstruction(U_r, V_r, W_r)
    if np.max(np.abs(T_comp - T)) < tol:
        return U_r, V_r, W_r
    return None

def _try_rank(rank, T, nm, mp, pn, deadline, n_restarts=30):
    """Attempt to find a rank-`rank` decomposition via randomized local search."""
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        scale = 0.3
        U = np.random.randn(nm, rank) * scale
        V = np.random.randn(mp, rank) * scale
        W = np.random.randn(pn, rank) * scale
        
        err = _error(U, V, W, T)
        
        # Local search with adaptive step sizes
        step_sizes = [0.5, 0.25, 0.1, 0.05, 0.01]
        for step in range(200):
            if time.time() > deadline:
                break
            
            improved = False
            # Try several random perturbations
            for _ in range(20):
                mat_choice = np.random.randint(3)
                delta = np.random.choice(step_sizes) * np.random.choice([-1, 1])
                
                if mat_choice == 0:
                    idx = (np.random.randint(nm), np.random.randint(rank))
                    U_new = U.copy()
                    U_new[idx] += delta
                    err_new = _error(U_new, V, W, T)
                    if err_new < err:
                        U = U_new
                        err = err_new
                        improved = True
                elif mat_choice == 1:
                    idx = (np.random.randint(mp), np.random.randint(rank))
                    V_new = V.copy()
                    V_new[idx] += delta
                    err_new = _error(U, V_new, W, T)
                    if err_new < err:
                        V = V_new
                        err = err_new
                        improved = True
                else:
                    idx = (np.random.randint(pn), np.random.randint(rank))
                    W_new = W.copy()
                    W_new[idx] += delta
                    err_new = _error(U, V, W_new, T)
                    if err_new < err:
                        W = W_new
                        err = err_new
                        improved = True
            
            if not improved:
                break
        
        # Check for exact solution
        if err < 1e-8:
            snapped = _snap_to_half_integers(U, V, W, T)
            if snapped is not None:
                U_r, V_r, W_r = snapped
                return (U_r, V_r, W_r)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    T = _build_tensor(n, m, p)
    
    # Try ranks from the known lower bound of 19 upward
    for rank in [19, 20, 21, 22, 23, 24]:
        if time.time() > deadline:
            break
        
        result = _try_rank(rank, T, nm, mp, pn, deadline, n_restarts=20)
        if result is not None:
            U_r, V_r, W_r = result
            U_list = [[float(U_r[i][j]) for j in range(rank)] for i in range(nm)]
            V_list = [[float(V_r[i][j]) for j in range(rank)] for i in range(mp)]
            W_list = [[float(W_r[i][j]) for j in range(rank)] for i in range(pn)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook decomposition as fallback
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
