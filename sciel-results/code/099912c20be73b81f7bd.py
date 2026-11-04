"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook
    rank_sb = n*m*p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    # Try alternating minimization for lower ranks
    np.random.seed(123)
    for target_rank in [30, 40, 50]:
        U = np.random.randn(nm, target_rank)
        V = np.random.randn(mp, target_rank)
        W = np.random.randn(pn, target_rank)
        
        for _ in range(20):
            M2 = np.einsum('bc,dc->cdb', V, W).reshape(target_rank, mp*pn)
            U = np.linalg.lstsq(M2, T.reshape(nm, mp*pn).T, rcond=None)[0].T
            M3 = np.einsum('ac,dc->cad', U, W).reshape(target_rank, nm*pn)
            V = np.linalg.lstsq(M3, T.transpose(1,0,2).reshape(mp, nm*pn).T, rcond=None)[0].T
            M4 = np.einsum('ac,bc->cab', U, V).reshape(target_rank, nm*mp)
            W = np.linalg.lstsq(M4, T.transpose(2,0,1).reshape(pn, nm*mp).T, rcond=None)[0].T
        
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        if np.linalg.norm(recon - T) < 1e-6:
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if np.linalg.norm(np.einsum('ac,bc,dc->abd', U_r, V_r, W_r) - T) < 1e-8:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
