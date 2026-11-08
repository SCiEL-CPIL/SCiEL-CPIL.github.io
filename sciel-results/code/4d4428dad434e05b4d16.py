import numpy as np
import time

def construct():
    n, m, p = 3, 4, 8
    deadline = time.time() + 100  # safety margin
    
    # Build target tensor M of shape (n*m, m*p, p*n)
    # M[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i', j==j', k==k' else 0
    # Actually: M[a, b, c] where a=i*m+j, b=j2*p+k, c=k2*n+i2
    # M[a,b,c] = 1 iff (a%m == b//p) and (b%p == c//n) and (a//m == c%n)
    # Wait: a = i*m+j, so i = a//m, j = a%m
    # b = j2*p+k, so j2 = b//p, k = b%p
    # c = k2*n+i2, so k2 = c//n, i2 = c%n
    # M[a,b,c] = 1 iff (a%m == b//p) and (b%p == c//n) and (a//m == c%n)
    
    NM = n * m  # 12
    MP = m * p  # 32
    PN = p * n  # 24
    
    # Build M as a flat array or use the structure
    # M[a, b, c] = 1 if (a%m == b//p) and (b%p == c//n) and (a//m == c%n)
    
    # For efficiency, let's build M as a 3D array
    M = np.zeros((NM, MP, PN), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a, b, c] = 1.0
    
    # Try to find rank-r decomposition
    # Use alternating minimization
    best_rank = NM * MP * PN  # schoolbook
    best_result = None
    
    # Start with schoolbook as initial solution
    def schoolbook():
        rank = NM * MP * PN
        U = np.zeros((NM, rank))
        V = np.zeros((MP, rank))
        W = np.zeros((PN, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W
    
    # Try decreasing ranks
    for target_rank in range(90, 50, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + target_rank * 100 + trial)
            U = np.random.randn(NM, target_rank) * 0.1
            V = np.random.randn(MP, target_rank) * 0.1
            W = np.zeros((PN, target_rank))
            
            # Alternating minimization
            for iteration in range(500):
                if time.time() > deadline:
                    break
                
                # Update W: for each c', solve linear system
                # sum_d U[a,d]*V[b,d]*W[c',d] = M[a,b,c']
                # This is: for fixed c', we have a vector of length NM*MP on LHS
                # Let's form the system differently
                
                # Reshape: for each d, define X_d = U[:,d] * V[:,d] (element-wise, shape NM*MP)
                # Then: sum_d X_d[a,b] * W[c',d] = M[a,b,c'] for all a,b,c'
                
                # This is a linear system: for each c', W[c',:] is a vector of length target_rank
                # The equations are: for all (a,b), sum_d W[c',d] * (U[a,d]*V[b,d]) = M[a,b,c']
                
                # Let's stack: for all c', we have M[:, :, c'] = sum_d W[c',d] * (U[:,d] * V[:,d])
                # This is: M_flat = sum_d W_flat_d * (UV)_d where M_flat has shape (NM*MP*PN,)
                
                # Actually, let's solve for W given U,V:
                # For each c' in range(PN):
                #   We need: sum_d W[c',d] * (U[:,d] * V[:,d]) = M[:, :, c']
                #   This is a linear system in W[c',:]
                #   Let A_c' be the NM*MP x target_rank matrix with A_c'[a*MP+b, d] = U[a,d]*V[b,d]
                #   Then A_c' @ W[c',:] = M[:, :, c'].flatten()
                
                # But this is a separate system for each c', which is expensive.
                # Alternative: solve all at once.
                
                # Let me think of it as: the tensor M can be written as:
                # M[a,b,c'] = sum_d U[a,d] * V[b,d] * W[c',d]
                # This is a CP decomposition with fixed U,V, solve for W.
                
                # Using the mode-3 unfolding: M_3 has shape (NM*MP, PN)
                # M_3[:, c'] = sum_d (U[:,d] * V[:,d])_flattened * W[c',d]
                # So M_3 = (U⊙V) @ W^T where U⊙V is NM*MP x target_rank with (U⊙V)[a*MP+b, d] = U[a,d]*V[b,d]
                
                UV = (U[:, :, None] * V[None, :, :]).reshape(NM * MP, target_rank)
                M3 = M.reshape(NM * MP, PN)
                # Solve: M3 = UV @ W^T  =>  W^T = UV^{-1} M3 (in least squares sense)
                W_new, residuals, rank_u, sv = np.linalg.lstsq(UV, M3, rcond=None)
                W = W_new.T  # shape (PN, target_rank)
                
                # Update U: fix V, W
                # M[a,b,c'] = sum_d U[a,d] * V[b,d] * W[c',d]
                # For fixed a: sum_d U[a,d] * (V[b,d] * W[c',d]) = M[a,b,c'] for all b,c'
                # Mode-1 unfolding: M_1 has shape (NM, MP*PN)
                # M_1[a, b*PN+c'] = sum_d U[a,d] * (V[b,d] * W[c',d])
                # So M_1 = U @ (V⊙W)^T where (V⊙W)[b*PN+c', d] = V[b,d]*W[c',d]
                
                VW = (V[:, :, None] * W[None, :, :]).reshape(MP * PN, target_rank)
                M1 = M.reshape(NM, MP * PN)
                U_new, residuals, rank_u, sv = np.linalg.lstsq(VW, M1, rcond=None)
                U = U_new.T
                
                # Update V: fix U, W
                # M_2[b, a*PN+c'] = sum_d V[b,d] * (U[a,d] * W[c',d])
                UW = (U[:, :, None] * W[None, :, :]).reshape(NM * PN, target_rank)
                M2 = M.reshape(MP, NM * PN)
                V_new, residuals, rank_u, sv = np.linalg.lstsq(UW, M2, rcond=None)
                V = V_new.T
                
                # Check convergence
                # Compute reconstruction error
                recon = np.einsum('ad,bd,cd->abc', U, V, W)
                err = np.linalg.norm(recon - M) / np.linalg.norm(M)
                if err < 1e-10:
                    break
            
            if time.time() > deadline:
                break
            
            # Check if we got a good solution
            recon = np.einsum('ad,bd,cd->abc', U, V, W)
            err = np.linalg.norm(recon - M) / np.linalg.norm(M)
            if err < 1e-8:
                # Try to snap to half-integers
                U_h = np.round(U * 2) / 2
                V_h = np.round(V * 2) / 2
                W_h = np.round(W * 2) / 2
                recon_h = np.einsum('ad,bd,cd->abc', U_h, V_h, W_h)
                err_h = np.linalg.norm(recon_h - M) / np.linalg.norm(M)
                if err_h < 1e-8:
                    # Found a valid half-integer solution!
                    best_rank = target_rank
                    best_result = (U_h, V_h, W_h)
                    break
                else:
                    # Try without snapping - use the real solution but convert to fractions
                    # For now, accept if very close and try to find rational approximation
                    pass
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U_h, V_h, W_h = best_result
        rank = best_rank
        # Convert to lists of integers/halves
        U_list = [[int(x * 2) / 2 if abs(x * 2 - round(x * 2)) < 1e-9 else float(x) for x in row] for row in U_h]
        V_list = [[int(x * 2) / 2 if abs(x * 2 - round(x * 2)) < 1e-9 else float(x) for x in row] for row in V_h]
        W_list = [[int(x * 2) / 2 if abs(x * 2 - round(x * 2)) < 1e-9 else float(x) for x in row] for row in W_h]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        U, V, W = schoolbook()
        rank = NM * MP * PN
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
