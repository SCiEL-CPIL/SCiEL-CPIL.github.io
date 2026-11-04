"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    U_dims = n * m  # 6
    V_dims = m * p  # 12
    W_dims = p * n  # 8
    
    # Build the matrix multiplication tensor T
    T_dense = np.zeros((U_dims, V_dims, W_dims))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_dense[a, b, c] = 1.0
    
    def als_cp(rank, n_iters=500, seed=0):
        """Run ALS for CP decomposition, return (U, V, W, error)"""
        rng = random.Random(seed)
        
        # Structured initialization: start from schoolbook-like structure
        U = np.zeros((U_dims, rank))
        V = np.zeros((V_dims, rank))
        W = np.zeros((W_dims, rank))
        
        # Initialize with small random values
        for t in range(rank):
            U[:, t] = rng.gauss(0, 0.5)
            V[:, t] = rng.gauss(0, 0.5)
            W[:, t] = rng.gauss(0, 0.5)
        
        err = float('inf')
        for it in range(n_iters):
            if time.time() > deadline:
                break
            
            # Update U
            M = np.einsum('bt,ct->btc', V, W)
            M_flat = M.reshape(rank, V_dims * W_dims)
            new_U = np.zeros((U_dims, rank))
            for a in range(U_dims):
                T_a = T_dense[a].reshape(-1)
                try:
                    new_U[a] = np.linalg.lstsq(M_flat, T_a, rcond=None)[0]
                except:
                    new_U[a] = U[a]
            U = new_U
            
            # Update V
            M2 = np.einsum('at,ct->act', U, W)
            M2_flat = M2.reshape(rank, U_dims * W_dims)
            new_V = np.zeros((V_dims, rank))
            for b in range(V_dims):
                T_b = T_dense[:, b, :].reshape(-1)
                try:
                    new_V[b] = np.linalg.lstsq(M2_flat, T_b, rcond=None)[0]
                except:
                    new_V[b] = V[b]
                V = new_V
            
            # Update W
            M3 = np.einsum('at,bt->abt', U, V)
            M3_flat = M3.reshape(rank, U_dims * V_dims)
            new_W = np.zeros((W_dims, rank))
            for c in range(W_dims):
                T_c = T_dense[:, :, c].reshape(-1)
                try:
                    new_W[c] = np.linalg.lstsq(M3_flat, T_c, rcond=None)[0]
                except:
                    new_W[c] = W[c]
                W = new_W
            
            # Compute error
            T_rec = np.einsum('at,bt,ct->abc', U, V, W)
            err = np.linalg.norm(T_rec - T_dense)
            if err < 1e-10:
                break
        
        return U, V, W, err
    
    def snap_to_half_integers(U, V, W, tol=1e-8):
        """Snap to half-integers and verify"""
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        
        T_rec = np.einsum('at,bt,ct->abc', U_snap, V_snap, W_snap)
        err = np.linalg.norm(T_rec - T_dense)
        if err < tol:
            return U_snap, V_snap, W_snap
        return None
    
    def to_list(mat):
        result = []
        for row in mat:
            rrow = []
            for val in row:
                v2 = round(val * 2)
                if abs(val * 2 - v2) < 1e-9:
                    rrow.append(v2 / 2.0)
                else:
                    rrow.append(float(val))
            result.append(rrow)
        return result
    
    # Try ranks from 19 to 24
    best_result = None
    for r in range(19, n * m * p + 1):
        if time.time() > deadline:
            break
        
        # Multiple restarts with different seeds
        for seed in range(100):
            if time.time() > deadline:
                break
            try:
                U, V, W, err = als_cp(r, n_iters=300, seed=seed * 1000 + r)
                if err < 1e-8:
                    snapped = snap_to_half_integers(U, V, W)
                    if snapped is not None:
                        U_s, V_s, W_s = snapped
                        best_result = {
                            'U': to_list(U_s),
                            'V': to_list(V_s),
                            'W': to_list(W_s)
                        }
                        break
            except Exception:
                continue
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
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
