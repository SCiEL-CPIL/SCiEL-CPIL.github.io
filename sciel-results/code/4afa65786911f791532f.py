"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Build the multiplication tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Precompute index mappings for efficient updates
    idx_ijk = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_ijk.append((i * m + j, j * p + k, k * n + i))
                
    # Mask of 1s
    mask_1 = np.zeros(T.shape, dtype=bool)
    for (a, b, c) in idx_ijk:
        mask_1[a, b, c] = True

    def check_exact(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose T"""
        # Use float64 for precision
        U = U.astype(np.float64)
        V = V.astype(np.float64)
        W = W.astype(np.float64)
        
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        
        # Check entries that should be 1
        if not np.allclose(T_recon[mask_1], 1.0, atol=tol):
            return False
        
        # Check entries that should be 0
        if not np.allclose(T_recon[~mask_1], 0.0, atol=tol):
            return False
            
        return True

    def try_rank(r, seed=42, max_iters=500):
        if time.time() > deadline - 2:
            return None
        
        np.random.seed(seed)
        
        # Initialize with random values
        # Scaling can be important. Let's try standard normal.
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                return None
            
            # Update U
            # Minimize || T - sum_d U_d V_d W_d ||_F
            # For each a: sum_{b,c} T[a,b,c] V[b,d] W[c,d] = sum_{b,c} (sum_{d'} U[a,d'] V[b,d'] W[c,d']) V[b,d] W[c,d]
            # Let G_U[d,d'] = sum_{b,c} V[b,d] W[c,d] V[b,d'] W[c,d'] * (sum_a T[a,b,c] ? No, T is sparse)
            # Actually, the objective is sum_{a,b,c} (T[a,b,c] - sum_d U[a,d]V[b,d]W[c,d])^2
            # Derivative w.r.t U[a,d] is -2 sum_{b,c} T[a,b,c] V[b,d] W[c,d] + 2 sum_{b,c} (sum_{d'} U[a,d'] V[b,d'] W[c,d']) V[b,d] W[c,d]
            # So we solve: sum_{d'} ( sum_{b,c} V[b,d] W[c,d] V[b,d'] W[c,d'] * (sum_a T[a,b,c]?? No) ) 
            # Wait, the term sum_a T[a,b,c] is not right. T[a,b,c] is non-zero only for specific a.
            # The sum over a is inside the derivative.
            # For a fixed a, the equation is:
            # sum_{d'} U[a,d'] * ( sum_{b,c} V[b,d'] W[c,d'] * V[b,d] W[c,d] * (1 if T[a,b,c]!=0 else 0) ) = sum_{b,c} T[a,b,c] V[b,d] W[c,d]
            
            # Let's precompute M_VW[d,d',b,c] = V[b,d] W[c,d] V[b,d'] W[c,d']
            # This is too big.
            
            # Better:
            # h_U[a,d] = sum_{b,c} T[a,b,c] V[b,d] W[c,d]
            # G_U[d,d'] = sum_{b,c} V[b,d] W[c,d] V[b,d'] W[c,d'] * S[a,b,c] where S is 1 if T[a,b,c]!=0?
            # No, G_U depends on a?
            # G_U[a,d,d'] = sum_{b,c} V[b,d] W[c,d] V[b,d'] W[c,d'] * (1 if T[a,b,c]!=0 else 0)
            
            # Since T[a,b,c] is 1 only for specific (b,c) for a given a,
            # let's just loop over a? No, that's slow.
            
            # Let's use the structure.
            # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
            
            # h_U[a,d] = sum_{j,k} V[j*p+k, d] W[k*n+i, d]  (where i=a//m, j=a%m)
            # G_U[a,d,d'] = sum_{j,k} V[j*p+k, d] W[k*n+i, d] V[j*p+k, d'] W[k*n+i, d']
            
            # We can compute this efficiently using matrices.
            # Let V_j be V[j*p:(j+1)*p, :]
            # Let W_k be W[k*n:(k+1)*n, :]
            
            # For fixed a (i,j):
            # h_U[a,d] = sum_k V[j*p+k, d] W[k*n+i, d]
            # This is (V[j,:] * W[:,i])_d ? No.
            # It is sum_k V[j*p+k, d] W[k*n+i, d]
            # Let's form a matrix M_VW_a where M_VW_a[d,d'] = sum_{b,c} V[b,d] W[c,d'] (for valid b,c for a)
            # Actually, for fixed a, the set of (b,c) is fixed.
            
            # Let's just compute the gradients directly for small dimensions.
            # n*m = 9, m*p = 15, p*n = 15.
            # r is around 20-40.
            
            # Update U
            # For each a in range(n*m):
            #   h = np.zeros(r)
            #   G = np.zeros((r, r))
            #   for b in range(m*p):
            #     for c in range(p*n):
            #       if T[a,b,c] != 0:
            #         h += V[b,:] * W[c,:]
            #         G += np.outer(V[b,:] * W[c,:], V[b,:] * W[c,:])
            #   U[a,:] = solve(G, h)
            
            # This is O(n*m * m*p * p*n * r^2) which is 9 * 15 * 15 * 40^2 = 324000 per iteration.
            # 500 iterations = 162M ops. This is feasible in Python/NumPy if vectorized, but loops in Python are slow.
            
            # Let's vectorize the update for U.
            # We need h_U[a,d] and G_U[a,d,d'].
            
            # h_U[a,d] = sum_{b,c} T[a,b,c] V[b,d] W[c,d]
            # This is einsum('abc,bd,cd->ad', T, V, W)
            
            # G_U[a,d,d'] = sum_{b,c} T[a,b,c] V[b,d] W[c,d] V[b,d'] W[c,d']
            # Let P[b,c,d] = V[b,d] W[c,d]
            # G_U[a,d,d'] = sum_{b,c} T[a,b,c] P[b,c,d] P[b,c,d']
            # This is einsum('abc,bcd,bce->ade', T, P, P)
            
            P = V[:, None, :] * W[None, :, :] # (m*p, p*n, r)
            h_U = np.einsum('abc,bcd->ad', T, P)
            G_U = np.einsum('abc,bcd,bce->ade', T, P, P)
            
            # Solve G_U[a] U[a,:] = h_U[a,:] for each a
            # G_U is (n*m, r, r). h_U is (n*m, r).
            # We can use np.linalg.solve if we reshape? No, solve is for 2D.
            # We can loop over a.
            for a in range(n * m):
                try:
                    U[a, :] = np.linalg.solve(G_U[a, :, :], h_U[a, :])
                except np.linalg.LinAlgError:
                    U[a, :] = np.linalg.lstsq(G_U[a, :, :], h_U[a, :])[0]
                    
            # Update V
            # h_V[b,d] = sum_{a,c} T[a,b,c] U[a,d] W[c,d]
            # G_V[b,d,d'] = sum_{a,c} T[a,b,c] U[a,d] W[c,d] U[a,d'] W[c,d']
            
            Q = U[:, None, :] * W[None, :, :] # (n*m, p*n, r)
            h_V = np.einsum('abc,acd->bd', T, Q)
            G_V = np.einsum('abc,acd,ace->bde', T, Q, Q)
            
            for b in range(m * p):
                try:
                    V[b, :] = np.linalg.solve(G_V[b, :, :], h_V[b, :])
                except np.linalg.LinAlgError:
                    V[b, :] = np.linalg.lstsq(G_V[b, :, :], h_V[b, :])[0]
                    
            # Update W
            # h_W[c,d] = sum_{a,b} T[a,b,c] U[a,d] V[b,d]
            # G_W[c,d,d'] = sum_{a,b} T[a,b,c] U[a,d] V[b,d] U[a,d'] V[b,d']
            
            R = U[:, None, :] * V[None, :, :] # (n*m, m*p, r)
            h_W = np.einsum('abc,abd->cd', T, R)
            G_W = np.einsum('abc,abd,abe->cde', T, R, R)
            
            for c in range(p * n):
                try:
                    W[c, :] = np.linalg.solve(G_W[c, :, :], h_W[c, :])
                except np.linalg.LinAlgError:
                    W[c, :] = np.linalg.lstsq(G_W[c, :, :], h_W[c, :])[0]
            
            # Optional: check convergence or exactness periodically
            if it % 50 == 0:
                # Try rounding
                for scale in [1, 2]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    if check_exact(U_r, V_r, W_r):
                        return (r, U_r, V_r, W_r)
        
        # Final check
        for scale in [1, 2]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if check_exact(U_r, V_r, W_r):
                return (r, U_r, V_r, W_r)
                
        return None

    # Try various ranks
    # Lower bound for 3x3x3x5 is unknown, but likely < 45.
    # Let's try from 25 upwards.
    ranks_to_try = list(range(25, rank_sb))
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            break
        
        # Try multiple seeds
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            result = try_rank(r, seed)
            if result is not None:
                r_found, U_f, V_f, W_f = result
                if r_found < best_rank:
                    best_rank = r_found
                    U_list = [[float(x) for x in row] for row in U_f]
                    V_list = [[float(x) for x in row] for row in V_f]
                    W_list = [[float(x) for x in row] for row in W_f]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                break # Found a rank, try next rank
    
    return best_result
# EVOLVE-BLOCK-END
