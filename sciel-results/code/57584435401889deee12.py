import numpy as np
import time
import scipy.optimize

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Try to find a decomposition of rank 45 (3 * 15)
    # First, find a rank 15 decomposition for 2x3x3
    U0, V0, W0 = _find_2x3x3_decomposition(15, deadline)
    
    if U0 is not None:
        # Construct the full 2x3x9 decomposition
        U, V, W = _expand_2x3x3_to_2x3x9(U0, V0, W0, n, m, p)
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _find_2x3x3_decomposition(rank, deadline):
    n, m, p = 2, 3, 3
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    # Target tensor
    T = np.zeros((U_size, V_size, W_size))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Flatten target
    T_flat = T.flatten()
    
    # Objective function
    def objective(params):
        if time.time() > deadline - 5:
            return float('inf')
            
        U = params[:U_size*rank].reshape(U_size, rank)
        V = params[U_size*rank:U_size*rank + V_size*rank].reshape(V_size, rank)
        W = params[U_size*rank + V_size*rank:].reshape(W_size, rank)
        
        # Compute tensor
        # T_comp[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # We only need to check the non-zero entries of T
        # Or just compute the whole thing and check residual
        # Since it's small, compute whole thing
        T_comp = np.zeros((U_size, V_size, W_size))
        for r in range(rank):
            T_comp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
        residual = T_comp - T
        return np.sum(residual**2)
        
    # Initial guess
    np.random.seed(42)
    x0 = np.random.randn(U_size*rank + V_size*rank + W_size*rank)
    
    # Use differential evolution
    try:
        result = scipy.optimize.differential_evolution(
            objective, 
            bounds=[(-10, 10)] * (U_size*rank + V_size*rank + W_size*rank),
            maxiter=50, 
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True
        )
        
        if result.fun < 1e-6:
            params = result.x
            U = params[:U_size*rank].reshape(U_size, rank)
            V = params[U_size*rank:U_size*rank + V_size*rank].reshape(V_size, rank)
            W = params[U_size*rank + V_size*rank:].reshape(W_size, rank)
            
            # Snap to half-integers if possible?
            # The problem allows integer/half-integer.
            # If the solution is close to such values, snap them.
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            # Verify snapped
            T_comp = np.zeros((U_size, V_size, W_size))
            for r in range(rank):
                T_comp += np.einsum('i,j,k->ijk', U_snap[:, r], V_snap[:, r], W_snap[:, r])
                
            if np.max(np.abs(T_comp - T)) < 1e-6:
                return U_snap, V_snap, W_snap
            else:
                # Return float if it works, but prompt says integer/half-integer.
                # If snap fails, we might need to return float? 
                # "entries that exactly decompose".
                # If we can't snap, we can't return this?
                # Let's check if the unsnapped version works.
                if np.max(np.abs(T_comp - T)) < 1e-6:
                     # Return float? The prompt says "integer/half-integer entries".
                     # This implies the output MUST be of that type.
                     # So if we can't snap, we can't use this solution.
                     pass
                     
    except Exception as e:
        pass
        
    return None, None, None

def _expand_2x3x3_to_2x3x9(U0, V0, W0, n, m, p):
    # U0: 6x15, V0: 9x15, W0: 6x15
    # Full: U: 6x45, V: 27x45, W: 18x45
    
    U = np.zeros((n*m, 3*15))
    V = np.zeros((m*p, 3*15))
    W = np.zeros((p*n, 3*15))
    
    for b in range(3):
        # U is the same for all blocks
        U[:, b*15:(b+1)*15] = U0
        
        # V: local (j, k_local) -> global (j, 3*b + k_local)
        for j in range(m):
            for k_local in range(3):
                v_global = j * p + (3*b + k_local)
                v_local = j * 3 + k_local
                V[v_global, b*15:(b+1)*15] = V0[v_local, :]
                
        # W: local (k_local, i) -> global (3*b + k_local, i)
        for k_local in range(3):
            for i in range(n):
                w_global = (3*b + k_local) * n + i
                w_local = k_local * n + i
                W[w_global, b*15:(b+1)*15] = W0[w_local, :]
                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    T_comp = np.zeros((n*m, m*p, p*n))
    rank = U.shape[1]
    for r in range(rank):
        T_comp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
    return np.max(np.abs(T_comp - T)) < 1e-6

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
