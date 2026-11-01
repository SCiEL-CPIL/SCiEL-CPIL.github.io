"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Target Tensor T[a, b, c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        # Check if U, V, W exactly decompose T
        # T_recon[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-9)

    def round_to_half_int(M):
        return np.round(M * 2) / 2

    def try_rank(R, seed=0, iters=500):
        np.random.seed(seed)
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # ALS Loop
        for _ in range(iters):
            # Solve for U
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # For fixed V, W, this is linear in U.
            # Reshape T to (n*m, m*p * p*n)
            T_flat = T.reshape(n * m, -1)
            # Construct matrix M_U where M_U[a, (b,c)*R + r] ... 
            # Actually, we can use einsum to form the linear system.
            # T[a, b, c] = U[a, r] * (V[b, r] * W[c, r])
            # Let Y[b, c, r] = V[b, r] * W[c, r]
            # T[a, b, c] = sum_r U[a, r] Y[b, c, r]
            # This is not a simple matrix multiplication for U unless we flatten b,c.
            # T[a, (b,c)] = sum_r U[a, r] Y[(b,c), r]
            Y = (V[:, :, None] * W[None, :, :]).reshape(-1, R) # Shape (m*p * p*n, R)
            # T_flat is (n*m, m*p*p*n)
            # We want U such that U @ Y.T approx T_flat
            # Least squares: U = (Y Y.T)^-1 Y T_flat.T
            # Using np.linalg.lstsq
            try:
                U, _, _, _ = np.linalg.lstsq(Y.T, T_flat.T, rcond=None)
            except:
                pass
                
            # Solve for V
            # T[a, b, c] = V[b, r] * (U[a, r] * W[c, r])
            # T[b, (a,c)] = sum_r V[b, r] Z[(a,c), r]
            Z = (U[:, :, None] * W[None, :, :]).reshape(-1, R) # Shape (n*m * p*n, R)
            T_flat_b = T.transpose(1, 0, 2).reshape(m * p, -1)
            try:
                V, _, _, _ = np.linalg.lstsq(Z.T, T_flat_b.T, rcond=None)
            except:
                pass
                
            # Solve for W
            # T[a, b, c] = W[c, r] * (U[a, r] * V[b, r])
            # T[c, (a,b)] = sum_r W[c, r] Q[(a,b), r]
            Q = (U[:, :, None] * V[None, :, :]).reshape(-1, R) # Shape (n*m * m*p, R)
            T_flat_c = T.transpose(2, 0, 1).reshape(p * n, -1)
            try:
                W, _, _, _ = np.linalg.lstsq(Q.T, T_flat_c.T, rcond=None)
            except:
                pass
            
        # Round and Verify
        U_r = round_to_half_int(U)
        V_r = round_to_half_int(V)
        W_r = round_to_half_int(W)
        
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
            
        # If rounding failed, try without rounding? 
        # The contract requires integer/half-integer. 
        # If the float solution is very close, rounding should work.
        # If not, return None.
        return None

    # Try ranks from 12 upwards
    for R in range(12, 17):
        for seed in range(20):
            res = try_rank(R, seed=seed, iters=100)
            if res:
                U, V, W = res
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # Fallback: Schoolbook Rank 16
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_fb[a, col] = 1
                V_fb[b, col] = 1
                W_fb[c, col] = 1
                col += 1
                
    U_list = [[float(x) for x in row] for row in U_fb]
    V_list = [[float(x) for x in row] for row in V_fb]
    W_list = [[float(x) for x in row] for row in W_fb]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
