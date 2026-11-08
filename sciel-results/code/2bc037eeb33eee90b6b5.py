import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Build the target tensor T[i*m+j, k*n+i] pattern
    # Actually, the tensor for matrix mult: T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a, b, c] = 1 if the indices are consistent, else 0
    # More precisely: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    
    # Let's build T as a 3D array: T[a, b, c]
    # a in [0, n*m), b in [0, m*p), c in [0, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # We want U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    # This is equivalent to: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r] sums to T
    
    # ALS approach: 
    # Fix V, W: U[a,r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r] / sum_{b,c} V[b,r]^2 * W[c,r]^2
    # Similar for V and W
    
    best_result = None
    
    for rank in range(28, 60):
        if time.time() > deadline - 5:
            break
        if best_result is not None:
            break
            
        # Try multiple random inits
        found = False
        max_inits = 50 if rank < 40 else 20
        for init in range(max_inits):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(init * 1000 + rank)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Normalize columns
            for r in range(rank):
                norm = np.linalg.norm(U[:,r])
                if norm > 0: U[:,r] /= norm
                norm = np.linalg.norm(V[:,r])
                if norm > 0: V[:,r] /= norm
                norm = np.linalg.norm(W[:,r])
                if norm > 0: W[:,r] /= norm
            
            # ALS iterations
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                    
                # Update U: fix V, W
                # U[:,r] should minimize ||T - sum_r' U[:,r'] V[r',:] W[r',:]^T||
                # For each r: U[:,r] = (V W^T)^{-1} * (T contracted with V[:,r] and W[:,r])
                # Actually: for fixed r, U[:,r] = argmin sum_{b,c} (T[a,b,c] - sum_r' U[a,r']V[b,r']W[c,r'])^2
                # This is a linear system. Let's use the normal equations.
                # sum_b,c V[b,r']W[c,r'] U[a,r'] = sum_b,c T[a,b,c] V[b,r] W[c,r] for each a,r
                # M = V @ W.T  (m*p x p*n -> no, V is m*p x rank, W is p*n x rank)
                # M[r',r] = sum_{b,c} V[b,r'] * W[c,r'] ... no
                # M[r',r] = sum_b V[b,r'] * (something)
                
                # Let me think again. The tensor T[a,b,c]. 
                # Residual R[a,b,c] = T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r]
                # dR/da for U[a,r]: -V[b,r] W[c,r]
                # Normal eq for U: sum_{b,c} V[b,r'] W[c,r'] U[a,r'] = sum_{b,c} T[a,b,c] V[b,r] W[c,r]
                # Let M[r',r] = sum_{b,c} V[b,r'] * W[c,r'] ... no that's not right either.
                # sum_{b,c} V[b,r'] W[c,r'] is a scalar for each (r',r), but it's sum_b V[b,r'] * sum_c W[c,r'] which is (V[:,r'] sum) * (W[:,r'] sum)
                # No! It's sum_{b,c} V[b,r'] W[c,r'] which equals (sum_b V[b,r'])(sum_c W[c,r']) only if independent, but it's actually a double sum.
                # sum_{b,c} V[b,r'] W[c,r'] = (sum_b V[b,r']) * (sum_c W[c,r'])
                # Yes! Because V[b,r'] doesn't depend on c and W[c,r'] doesn't depend on b.
                
                # So M[r',r] = (sum_b V[b,r']) * (sum_c W[c,r'])
                # RHS[a,r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r]
                
                # This is getting complicated. Let me use a different approach.
                # Reshape T into a matrix for each r.
                
                # Actually, let me use a simpler iterative approach:
                # For each r, update U[:,r] by:
                # U[:,r] = (sum_{b,c} T[a,b,c] V[b,r] W[c,r]) / (sum_{b,c} V[b,r]^2 W[c,r]^2) ... no
                # This is not quite right for ALS on tensors.
                
                # Let me use the standard tensor ALS:
                # For U: U[:,r] = (V ⊙ W)^{-1} * (T_{(1)} (V ⊙ W))[:,r]
                # where ⊙ is Khatri-Rao product and T_{(1)} is the mode-1 unfolding.
                
                # Mode-1 unfolding of T: T1[a, (b,c)] where (b,c) is flattened
                # Khatri-Rao of V and W: (V ⊙ W)[(b,c), r] = V[b,r] * W[c,r]
                
                T1 = T.reshape(n*m, m*p * p*n)  # mode-1 unfolding
                Vw = (V.reshape(m*p, 1, rank) * W.reshape(1, p*n, rank)).reshape(m*p * p*n, rank)
                
                # U = T1 @ Vw @ inv(Vw.T @ Vw)
                A = Vw.T @ Vw
                B_mat = T1 @ Vw
                try:
                    U_new = np.linalg.solve(A, B_mat.T).T
                except:
                    U_new = np.linalg.lstsq(A, B_mat.T, rcond=None)[0].T
                U = U_new
                
                # Similarly for V
                T2 = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
                Uw = (U.reshape(n*m, 1, rank) * W.reshape(1, p*n, rank)).reshape(n*m * p*n, rank)
                A = Uw.T @ Uw
                B_mat = T2 @ Uw
                try:
                    V_new = np.linalg.solve(A, B_mat.T).T
                except:
                    V_new = np.linalg.lstsq(A, B_mat.T, rcond=None)[0].T
                V = V_new
                
                # Similarly for W
                T3 = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
                Uw2 = (U.reshape(n*m, 1, rank) * V.reshape(1, m*p, rank)).reshape(n*m * m*p, rank)
                A = Uw2.T @ Uw2
                B_mat = T3 @ Uw2
                try:
                    W_new = np.linalg.solve(A, B_mat.T).T
                except:
                    W_new = np.linalg.lstsq(A, B_mat.T, rcond=None)[0].T
                W = W_new
                
                # Check convergence
                if it % 10 == 0:
                    recon = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.linalg.norm(recon - T)
                    if err < 1e-6:
                        break
            
            # Check final error
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(recon - T)
            if err < 1e-4:
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err_r = np.linalg.norm(recon_r - T)
                if err_r < 1e-6:
                    # Convert to lists
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break
        
        if found:
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
