"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def try_rank(rank, seed):
        np.random.seed(seed)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        for it in range(5000):
            # Solve for U
            Vt_pinv = np.linalg.pinv(V.T)
            U_new = np.zeros_like(U)
            for c in range(p * n):
                rhs = T[:, :, c] @ Vt_pinv
                for cp in range(rank):
                    if abs(W[c, cp]) > 1e-10:
                        U_new[:, cp] = rhs[:, cp] / W[c, cp]
                    else:
                        U_new[:, cp] = U[:, cp]
            U = U_new
            
            # Solve for V
            A_rows = [U[a, :] * W[c, :] for a in range(n * m) for c in range(p * n)]
            B_rows = [T[a, :, c] for a in range(n * m) for c in range(p * n)]
            A_mat = np.array(A_rows)
            B_mat = np.array(B_rows)
            Vt_sol = np.linalg.lstsq(A_mat, B_mat, rcond=None)[0]
            V = Vt_sol.T
            
            # Solve for W
            A_rows = [U[a, :] * V[b, :] for a in range(n * m) for b in range(m * p)]
            B_rows = [T[a, b, :] for a in range(n * m) for b in range(m * p)]
            A_mat = np.array(A_rows)
            B_mat = np.array(B_rows)
            Wt_sol = np.linalg.lstsq(A_mat, B_mat, rcond=None)[0]
            W = Wt_sol.T
            
            # Check
            T_recon = np.zeros_like(T)
            for c in range(rank):
                T_recon += np.einsum('a,b->ab', U[:, c], V[:, c]) * W[:, c]
            if np.allclose(T_recon, T, atol=1e-8):
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                T_recon2 = np.zeros_like(T)
                for c in range(rank):
                    T_recon2 += np.einsum('a,b->ab', U_r[:, c], V_r[:, c]) * W_r[:, c]
                if np.allclose(T_recon2, T, atol=1e-6):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        return None
    
    for rank in range(20, 37):
        for seed in range(10):
            result = try_rank(rank, rank * 100 + seed)
            if result is not None:
                return result
    
    # Fallback
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

