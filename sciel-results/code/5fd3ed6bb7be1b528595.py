"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Build the tensor T
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    best_rank = nm * mp * pn  # schoolbook = 42
    best_result = None
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Try ranks from 41 down to 31
    for rank in range(41, 30, -1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        for trial in range(30):
            if time.time() > deadline - 15:
                break
            
            seed = trial * 1000 + rank
            np.random.seed(seed)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # ALS iterations
            for iteration in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Fix U, V, solve for W
                W_new = np.zeros((pn, rank))
                for c in range(pn):
                    M = np.einsum('ar,br->abr', U, V)
                    M_flat = M.reshape(-1, rank)
                    T_c = T[:, :, c].flatten()
                    sol, _, _, _ = np.linalg.lstsq(M_flat, T_c, rcond=None)
                    W_new[c, :] = sol
                W = W_new
                
                # Fix W, V, solve for U
                U_new = np.zeros((nm, rank))
                for a in range(nm):
                    S = np.einsum('br,cr->bcr', V, W)
                    S_flat = S.reshape(-1, rank)
                    T_a = T[a, :, :].flatten()
                    sol, _, _, _ = np.linalg.lstsq(S_flat, T_a, rcond=None)
                    U_new[a, :] = sol
                U = U_new
                
                # Fix U, W, solve for V
                V_new = np.zeros((mp, rank))
                for b in range(mp):
                    S = np.einsum('ar,cr->acr', U, W)
                    S_flat = S.reshape(-1, rank)
                    T_b = T[:, b, :].flatten()
                    sol, _, _, _ = np.linalg.lstsq(S_flat, T_b, rcond=None)
                    V_new[b, :] = sol
                V = V_new
            
            # Check if the decomposition is valid
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.abs(T_recon - T).max()
            
            if err < 1e-8:
                if rank < best_rank:
                    best_rank = rank
                    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(nm)]
                    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(mp)]
                    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(pn)]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                break  # Found valid for this rank, try lower rank
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    rank = nm * mp * pn
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
# EVOLVE-BLOCK-END
