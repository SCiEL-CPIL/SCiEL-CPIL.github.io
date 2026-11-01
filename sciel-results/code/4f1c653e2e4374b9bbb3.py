"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Build the target tensor
    tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook as baseline
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank via ALS
    for rank in range(schoolbook_rank - 1, 10, -1):
        if time.time() > deadline:
            break
        
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 1000 + rank)
            U_r = np.random.randn(n*m, rank)
            V_r = np.random.randn(m*p, rank)
            W_r = np.random.randn(p*n, rank)
            
            success = False
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Solve for U
                VW = (V_r[:, :, None] * W_r[None, :, :]).reshape(m*p, p*n, rank).transpose(1, 0, 2).reshape(m*p * p*n, rank)
                T_mat = tensor.reshape(n*m, m*p * p*n)
                try:
                    U_new = T_mat @ np.linalg.pinv(VW.T)
                except:
                    break
                
                # Solve for V
                UW = (U_new[:, :, None] * W_r[None, :, :]).reshape(n*m, p*n, rank).transpose(1, 0, 2).reshape(p*n * n*m, rank)
                T_mat2 = tensor.transpose(1, 0, 2).reshape(m*p, p*n * n*m)
                try:
                    V_new = T_mat2 @ np.linalg.pinv(UW.T)
                except:
                    break
                
                # Solve for W
                UV = (U_new[:, :, None] * V_new[None, :, :]).reshape(n*m, m*p, rank).transpose(1, 0, 2).reshape(m*p * n*m, rank)
                T_mat3 = tensor.transpose(2, 0, 1).reshape(p*n, m*p * n*m)
                try:
                    W_new = T_mat3 @ np.linalg.pinv(UV.T)
                except:
                    break
                
                U_r, V_r, W_r = U_new, V_new, W_new
                
                # Check residual
                T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                residual = np.linalg.norm(T_recon - tensor)
                if residual < 1e-10:
                    success = True
                    break
            
            if success:
                # Verify exactness with rational check
                T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                if np.allclose(T_recon, tensor, atol=1e-10):
                    best_rank = rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    break
    
    U_list = [[float(best_U[i, j]) for j in range(best_rank)] for i in range(n*m)]
    V_list = [[float(best_V[i, j]) for j in range(best_rank)] for i in range(m*p)]
    W_list = [[float(best_W[i, j]) for j in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
