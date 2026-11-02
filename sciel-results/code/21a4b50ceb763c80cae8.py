"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 30
    pn = p * n  # 12
    
    deadline = time.time() + 130
    
    # Build the exact tensor T[a,b,c]
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try to find a rank-r decomposition for r from lower bound upward
    lower_bound = max(nm, mp, pn)  # 30
    
    for rank in range(lower_bound, nm * mp * pn + 1):
        if time.time() > deadline:
            break
        
        # Alternating least squares with a few restarts
        for seed in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(seed * 1000 + rank)
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            for it in range(80):
                if time.time() > deadline:
                    break
                
                # Solve for U: T_flat = U @ Vw
                Vw = np.zeros((rank, mp * pn))
                for r in range(rank):
                    Vw[r, :] = (V[:, r] * W[r, :]).flatten()
                T_flat = T.reshape(nm, mp * pn)
                U = T_flat @ np.linalg.pinv(Vw)
                
                # Solve for V
                Uw = np.zeros((rank, nm * pn))
                for r in range(rank):
                    Uw[r, :] = (U[:, r] * W[r, :]).flatten()
                T_flat2 = T.reshape(mp, nm * pn)
                V = T_flat2 @ np.linalg.pinv(Uw)
                
                # Solve for W
                UV = np.zeros((rank, nm * mp))
                for r in range(rank):
                    UV[r, :] = (U[:, r] * V[r, :]).flatten()
                T_flat3 = T.reshape(pn, nm * mp)
                W = T_flat3 @ np.linalg.pinv(UV)
                
                # Check error
                T_recon = np.zeros((nm, mp, pn))
                for r in range(rank):
                    T_recon += np.outer(U[:, r], V[:, r]) * W[r, :][:, None, None]
                err = np.linalg.norm(T_recon - T)
                if err < 1e-9:
                    # Snap to half-integers and verify
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    T_recon2 = np.zeros((nm, mp, pn))
                    for r in range(rank):
                        T_recon2 += np.outer(U_snap[:, r], V_snap[:, r]) * W_snap[r, :][:, None, None]
                    
                    if np.allclose(T_recon2, T, atol=1e-6):
                        U_list = [[float(x) for x in row] for row in U_snap]
                        V_list = [[float(x) for x in row] for row in V_snap]
                        W_list = [[float(x) for x in row] for row in W_snap]
                        return {"U": U_list, "V": V_list, "W": W_list}
                    break
    
    # Fallback: schoolbook decomposition
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
