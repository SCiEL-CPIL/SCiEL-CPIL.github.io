import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_flat = T.reshape(dim_u, dim_v * dim_w)
    
    # Schoolbook fallback
    rank_schoolbook = dim_u * dim_v * dim_w / dim_w # n*m*p
    rank_schoolbook = int(n * m * p)
    
    def make_schoolbook():
        U = [[0] * rank_schoolbook for _ in range(dim_u)]
        V = [[0] * rank_schoolbook for _ in range(dim_v)]
        W = [[0] * rank_schoolbook for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
        
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Search for lower rank
    # Theoretical lower bound is roughly max(dim_u, dim_v, dim_w) but actually higher.
    # Let's try from 50 to 96.
    # If we find a valid one, return it.
    
    # To improve chances, we can use the schoolbook as a warm start?
    # No, rank is different.
    
    # Try ranks
    # We can be smart: try ranks that are multiples of something? No.
    # Just iterate.
    
    # To speed up, we can check if a rank is possible using rank of matricizations?
    # Rank(T) >= rank(T_{(dim_u, dim_v*dim_w)})
    # This is always <= dim_u.
    
    # Let's just run ALS.
    
    # We'll try a few restarts for each rank.
    # If we find a solution, we snap and verify.
    
    # Order: try lower ranks first.
    # But if the minimum is 90, starting at 50 wastes time.
    # However, if minimum is 60, starting at 90 misses it.
    # Given the dimensions, 96 is the upper bound.
    # Let's try 70, 75, 80, 85, 90, 95, 96?
    # Or just 60 to 96.
    
    # Let's estimate the rank.
    # For 4x4x4, rank 49.
    # 3x4x8 is roughly 3/4 of 4x4x8?
    # 4x4x8 rank is approx 49 * 2 = 98? No.
    # Rank is sub-multiplicative?
    # R(3,4,8) <= R(4,4,4) * R(1,1,2)?
    # R(1,1,2) = 2.
    # 49 * 2 = 98.
    # So 96 is likely close to optimal or optimal.
    # But we can try to reduce it.
    
    # Let's try ranks from 90 down to 60.
    # If we find one at 90, we return 90.
    # But maybe 80 is possible?
    # We should search upwards from a lower bound to find the MINIMUM.
    # But we don't know the lower bound.
    # Let's try 70, 71, ..., 96.
    
    # To save time, we can skip some ranks? No, we want minimal.
    
    # Let's try 80 to 96 first, as it's more likely to converge.
    # Then 70 to 80.
    
    # Actually, let's just do 60 to 96.
    # If time is running out, return schoolbook.
    
    for R in range(60, rank_schoolbook + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Try 3 restarts for each R
        for trial in range(3):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize
            U = np.random.randn(dim_u, R)
            V = np.random.randn(dim_v, R)
            W = np.random.randn(dim_w, R)
            
            # ALS
            converged = False
            for it in range(50):
                # Solve U
                # M_VW[:, r] = V[:, r] \otimes W[:, r]
                M = np.zeros((dim_v * dim_w, R))
                for r in range(R):
                    M[:, r] = np.kron(V[:, r], W[:, r])
                    
                # T_flat (dim_u, dim_v*dim_w) = U (dim_u, R) @ M.T (R, dim_v*dim_w)
                # U = T_flat @ M.T @ pinv(M @ M.T)
                # Use lstsq: M @ U.T = T_flat.T
                U_new, _, _, _ = np.linalg.lstsq(M, T_flat.T, rcond=None)
                U = U_new.T
                
                # Solve V
                # T_flat (dim_u, dim_v*dim_w)
                # We need to reshape T to (dim_v, dim_u * dim_w)
                # T_{b, a, c} = V_{b,r} * (U_{a,r} \otimes W_{c,r})
                # M_UW[:, r] = U[:, r] \otimes W[:, r]
                # T_reshape_V (dim_v, dim_u*dim_w) = V (dim_v, R) @ M_UW.T
                T_reshape_V = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
                
                M_UW = np.zeros((dim_u * dim_w, R))
                for r in range(R):
                    M_UW[:, r] = np.kron(U[:, r], W[:, r])
                    
                V_new, _, _, _ = np.linalg.lstsq(M_UW, T_reshape_V.T, rcond=None)
                V = V_new.T
                
                # Solve W
                # T_{c, a, b} = W_{c,r} * (U_{a,r} \otimes V_{b,r})
                # T_reshape_W (dim_w, dim_u * dim_v)
                T_reshape_W = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
                
                M_UV = np.zeros((dim_u * dim_v, R))
                for r in range(R):
                    M_UV[:, r] = np.kron(U[:, r], V[:, r])
                    
                W_new, _, _, _ = np.linalg.lstsq(M_UV, T_reshape_W.T, rcond=None)
                W = W_new.T
                
                # Check error
                # Reconstruct T_flat
                M_check = np.zeros((dim_v * dim_w, R))
                for r in range(R):
                    M_check[:, r] = np.kron(V[:, r], W[:, r])
                T_rec = U @ M_check.T
                err = np.linalg.norm(T_flat - T_rec)
                
                if err < 1e-8:
                    converged = True
                    break
                    
            if converged:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify snapped
                M_snap = np.zeros((dim_v * dim_w, R))
                for r in range(R):
                    M_snap[:, r] = np.kron(V_snap[:, r], W_snap[:, r])
                T_rec_snap = U_snap @ M_snap.T
                err_snap = np.linalg.norm(T_flat - T_rec_snap)
                
                if err_snap < 1e-6:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                
                # If snapping fails, check if float error is tiny
                if err < 1e-10:
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }

    return make_schoolbook()
# EVOLVE-BLOCK-END
