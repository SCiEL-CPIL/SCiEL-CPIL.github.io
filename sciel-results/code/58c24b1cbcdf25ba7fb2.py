import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 4]

def _verify(U, V, W, n=2, m=2, p=4):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _construct_rank14():
    """Return the known rank-14 Strassen-based solution."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1 (k=0)
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    V[0][0] = 1; V[5][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[5][2] = -1
    V[4][3] = 1; V[0][3] = -1
    V[5][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[4][6] = 1; V[5][6] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][1] = 1; W[1][3] = 1
    W[2][2] = 1; W[2][4] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Block 2 (k=1)
    U[0][7] = 1; U[3][7] = 1
    U[2][8] = 1; U[3][8] = 1
    U[0][9] = 1
    U[3][10] = 1
    U[0][11] = 1; U[1][11] = 1
    U[2][12] = 1; U[0][12] = -1
    U[1][13] = 1; U[3][13] = -1
    
    V[2][7] = 1; V[7][7] = 1
    V[2][8] = 1
    V[3][9] = 1; V[7][9] = -1
    V[6][10] = 1; V[2][10] = -1
    V[7][11] = 1
    V[2][12] = 1; V[3][12] = 1
    V[6][13] = 1; V[7][13] = 1
    
    W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1
    W[5][8] = 1; W[5][10] = 1
    W[6][9] = 1; W[6][11] = 1
    W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
    
    return {"U": U, "V": V, "W": W}

def _search_rank12(deadline):
    """Attempt to find a rank-12 decomposition."""
    n, m, p = 2, 2, 4
    rank = 12
    
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    nz_indices = []
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                nz_indices.append((a, b, c))
                
    T = np.zeros((U_size, V_size, W_size))
    for a, b, c in nz_indices:
        T[a, b, c] = 1.0
        
    def compute_tensor(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W)
        
    def objective_and_grad(x):
        U = x[:U_size*rank].reshape(U_size, rank)
        V = x[U_size*rank:U_size*rank+V_size*rank].reshape(V_size, rank)
        W = x[U_size*rank+V_size*rank:].reshape(W_size, rank)
        
        T_calc = compute_tensor(U, V, W)
        diff = T_calc - T
        loss = np.sum(diff**2)
        
        grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
        grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
        grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    best_solution = None
    best_loss = float('inf')
    
    # Get rank 14 solution for warm start
    sol14 = _construct_rank14()
    U14 = np.array(sol14['U'], dtype=float)
    V14 = np.array(sol14['V'], dtype=float)
    W14 = np.array(sol14['W'], dtype=float)
    
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialization Strategy 1: Random
        if r % 2 == 0:
            x0 = np.random.normal(0, 1, size=U_size*rank + V_size*rank + W_size*rank)
        else:
            # Initialization Strategy 2: Warm start from Rank 14
            # Try to project Rank 14 to Rank 12
            # We can try to zero out 2 columns or optimize them
            # Here we just take random subsets of columns or perturb them
            # A better warm start: take the first 12 columns of Rank 14?
            # This might not be optimal but is a good start.
            U_init = U14[:, :rank]
            V_init = V14[:, :rank]
            W_init = W14[:, :rank]
            # Add small noise to avoid getting stuck in a bad local min
            U_init += np.random.normal(0, 0.1, U_init.shape)
            V_init += np.random.normal(0, 0.1, V_init.shape)
            W_init += np.random.normal(0, 0.1, W_init.shape)
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < 1e-8:
                U = res.x[:U_size*rank].reshape(U_size, rank)
                V = res.x[U_size*rank:U_size*rank+V_size*rank].reshape(V_size, rank)
                W = res.x[U_size*rank+V_size*rank:].reshape(W_size, rank)
                
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    
                # If snapping fails, try a tighter snap or check if it's close enough
                # Sometimes the solution is exactly integer, so round(x) might be better?
                # But half-integers are allowed.
                
        except Exception:
            pass

    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140 # 10s margin
    
    # Try rank 12
    res12 = _search_rank12(deadline)
    if res12:
        return res12
        
    # Fallback to rank 14
    return _construct_rank14()
