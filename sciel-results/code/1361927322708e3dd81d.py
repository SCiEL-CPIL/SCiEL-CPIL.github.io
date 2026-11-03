import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # The tensor for matrix multiplication C = A * B
    # C_ik = sum_j A_ij B_jk
    # We want to decompose the tensor T_{ijk} = 1 (for all i,j,k) into rank R
    # T_{ijk} = sum_r U_{ijr} V_{jkr} W_{kir}
    
    # Known bounds:
    # Schoolbook: n*m*p = 100
    # We try to find a lower rank.
    
    # Strategy:
    # 1. Try to construct a decomposition using known algorithms (Strassen, Bini, etc.) if applicable.
    # 2. Use numerical optimization (ALS) to find a low-rank decomposition.
    # 3. Round to half-integers and verify.
    
    # For 4x5 * 5x5, we can try to use a combination of blocks.
    # However, a direct numerical search is more robust for finding the exact minimal rank if it's not too high.
    
    # Let's start with a rank that is slightly below schoolbook, e.g., 90, 85, 80...
    # The lower bound is max(n*m, m*p, p*n) = 25.
    # But for 4x5x5, the rank is likely around 80-90.
    
    # We will attempt to find a decomposition for ranks in a decreasing order.
    # If we find one that is exact (after rounding), we return it.
    
    # To speed up, we will use a smarter initialization and more iterations.
    
    def try_rank(R, seed=42):
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Normalize initial factors to prevent blow-up
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        V /= np.linalg.norm(V, axis=0, keepdims=True)
        W /= np.linalg.norm(W, axis=0, keepdims=True)
        
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # ALS Iterations
        # We want to minimize || T - U V W ||_F^2
        # where T is the tensor of ones (n x m x p)
        
        # Reshape for einsum
        # U: (n, m, R)
        # V: (m, p, R)
        # W: (p, n, R)
        
        # Target tensor T: (n, m, p) with all 1s
        T = np.ones((n, m, p))
        
        # Precompute T flattened for error calculation?
        # It's better to use einsum for the reconstruction.
        
        for it in range(300):
            if time.time() > deadline - 5.0:
                break
                
            # Update U
            # U_{ijr} = (sum_k T_{ijk} V_{jkr} W_{kir}) / (sum_k V_{jkr}^2 W_{kir}^2) ?? No.
            # The gradient step for ALS on CP decomposition:
            # U_{ijr} = < T_{ijk} - sum_{r'!=r} U_{ijr'} V_{jr'} W_{kr'}, V_{jr'} W_{kr'} > / < V_{jr'} W_{kr'}, V_{jr'} W_{kr'} >
            # This is expensive.
            
            # Simpler ALS update (assuming orthogonality or using the full product):
            # Actually, the standard ALS update for minimizing ||T - U V W||^2 is:
            # U_{ijr} = (sum_{k} T_{ijk} V_{jkr} W_{kir} - sum_{r'!=r} U_{ijr'} sum_{k} V_{jkr'} W_{kir'}) / (sum_{k} V_{jkr}^2 W_{kir}^2 - sum_{r'!=r} (sum_{k} V_{jkr'} W_{kir'})^2)
            # This is complex.
            
            # Let's use a simpler gradient descent or a specific ALS variant.
            # A common efficient ALS for tensor decomposition:
            # Fix V, W. Solve for U.
            # The term is sum_r U_{ijr} (V_{jkr} W_{kir}).
            # This is a linear system for each (i,j)? No, for each r.
            # Actually, for fixed r, U_{ijr} is a scalar.
            # The objective is sum_{ijk} (T_{ijk} - sum_r U_{ijr} V_{jkr} W_{kir})^2.
            # For a fixed r, the optimal U_{ijr} given other r's is:
            # Let R_{ijk} = T_{ijk} - sum_{r'!=r} U_{ijr'} V_{jr'} W_{kr'}
            # Then U_{ijr} = (sum_k R_{ijk} V_{jkr} W_{kir}) / (sum_k V_{jkr}^2 W_{kir}^2)
            
            # This requires computing the residual R, which is O(n*m*p*R).
            # With n=4, m=5, p=5, R~80, this is 100*80 = 8000 operations per update.
            # 3 factors * 300 iters * 8000 = 7.2M operations. Very fast.
            
            U_r = U.reshape(n, m, R)
            V_r = V.reshape(m, p, R)
            W_r = W.reshape(p, n, R)
            
            # Compute Reconstruction
            # Rec_{ijk} = sum_r U_{ijr} V_{jkr} W_{kir}
            Rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            
            # Residual
            Res = T - Rec
            
            # Update U
            # Num_U_{ijr} = sum_k Res_{ijk} V_{jkr} W_{kir}
            # Den_U_{ijr} = sum_k V_{jkr}^2 W_{kir}^2
            Num_U = np.einsum('ijk,jkr,kir->ijr', Res, V_r, W_r)
            Den_U = np.einsum('jkr,kir->ijr', V_r*V_r, W_r*W_r) # Note: V_r*V_r is (m,p,R), W_r*W_r is (p,n,R). Wait.
            # V_{jkr}^2 is element-wise square.
            # We need sum_k V_{jkr}^2 W_{kir}^2.
            # V_sq = V_r * V_r (m, p, R)
            # W_sq = W_r * W_r (p, n, R)
            # sum_k V_sq[j,k,r] * W_sq[k,i,r]
            V_sq = V_r * V_r
            W_sq = W_r * W_r
            Den_U = np.einsum('jkr,kir->ijr', V_sq, W_sq)
            
            Den_U[Den_U == 0] = 1e-12
            U_new = Num_U / Den_U
            U = U_new.reshape(n * m, R)
            
            # Recompute Rec and Res for V update
            U_r = U.reshape(n, m, R)
            Rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            Res = T - Rec
            
            # Update V
            # Num_V_{jkr} = sum_i Res_{ijk} U_{ijr} W_{kir}
            # Den_V_{jkr} = sum_i U_{ijr}^2 W_{kir}^2
            Num_V = np.einsum('ijk,ijr,kir->jkr', Res, U_r, W_r)
            U_sq = U_r * U_r
            Den_V = np.einsum('ijr,kir->jkr', U_sq, W_sq)
            Den_V[Den_V == 0] = 1e-12
            V_new = Num_V / Den_V
            V = V_new.reshape(m * p, R)
            
            # Recompute Rec and Res for W update
            V_r = V.reshape(m, p, R)
            Rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            Res = T - Rec
            
            # Update W
            # Num_W_{kir} = sum_j Res_{ijk} U_{ijr} V_{jkr}
            # Den_W_{kir} = sum_j U_{ijr}^2 V_{jkr}^2
            Num_W = np.einsum('ijk,ijr,jkr->kir', Res, U_r, V_r)
            V_sq = V_r * V_r
            Den_W = np.einsum('ijr,jkr->kir', U_sq, V_sq)
            Den_W[Den_W == 0] = 1e-12
            W_new = Num_W / Den_W
            W = W_new.reshape(p * n, R)
            
            # Calculate Error
            U_r = U.reshape(n, m, R)
            V_r = V.reshape(m, p, R)
            W_r = W.reshape(p, n, R)
            Rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            err = np.sum((Rec - T) ** 2)
            
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_error < 1e-8:
                break
        
        if best_error < 1e-6:
            # Try to round to half-integers
            U_int = np.round(best_U * 2) / 2
            V_int = np.round(best_V * 2) / 2
            W_int = np.round(best_W * 2) / 2
            
            # Verify
            U_r = U_int.reshape(n, m, R)
            V_r = V_int.reshape(m, p, R)
            W_r = W_int.reshape(p, n, R)
            Rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            if np.max(np.abs(Rec - T)) < 1e-9:
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
                
            # Try integer rounding too
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            U_r = U_int.reshape(n, m, R)
            V_r = V_int.reshape(m, p, R)
            W_r = W_int.reshape(p, n, R)
            Rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            if np.max(np.abs(Rec - T)) < 1e-9:
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
                
        return None

    # Try ranks
    # We know 100 works. Let's try lower.
    # Based on literature, rank of 4x5x5 might be around 80-90.
    # We will scan from 80 to 100.
    
    for R in range(80, 101):
        if time.time() > deadline - 10.0:
            break
            
        res = try_rank(R)
        if res is not None:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
            
        # Try with different seeds if time permits
        if time.time() < deadline - 20.0:
            for seed in [1, 2, 3]:
                if time.time() > deadline - 10.0:
                    break
                res = try_rank(R, seed=seed)
                if res is not None:
                    U, V, W = res
                    return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
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
