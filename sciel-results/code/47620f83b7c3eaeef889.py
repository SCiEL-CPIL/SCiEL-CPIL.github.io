"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor T as a 3D array
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    deadline = time.time() + 100
    
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    for rank in range(18, 37):
        if time.time() > deadline:
            break
        
        for trial in range(100):
            if time.time() > deadline:
                break
            
            np.random.seed(trial * 13 + rank * 7)
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Alternating minimization
            for it in range(500):
                if time.time() > deadline:
                    break
                
                # Solve for U: for each a, U[a,:] should satisfy
                # sum_r U[a,r] * (V[:,r] outer W[:,r])[b,c] = T[a,b,c]
                # Let M_U[a] = T[a,:,:].reshape(-1)
                # Let G[a] = matrix where G[a, r, bc] = V[b,r]*W[c,r]
                # Then U[a,:] = argmin ||G[a] @ U[a,:] - M_U[a]||^2
                # G[a] is (mp*pn) x rank
                
                # Build G_U: shape (nm, rank, mp*pn)
                # G_U[a, r, b*pn+c] = V[b,r]*W[c,r]
                VW = np.einsum('br,cr->bcr', V, W)  # (mp, pn, rank)
                VW_flat = VW.transpose(2, 0, 1).reshape(rank, mp * pn)  # (rank, mp*pn)
                
                T_U = T.reshape(nm, -1)  # (nm, mp*pn)
                
                # Solve U = argmin ||VW_flat.T @ U.T - T_U.T||^2
                # i.e., for each a: VW_flat.T @ U[a,:] - T_U[a,:] = 0
                # U[a,:] = pinv(VW_flat.T) @ T_U[a,:]
                # Actually, U.T = (VW_flat.T)^+ @ T_U.T
                # U = pinv(VW_flat) @ T_U.T ... let me be careful
                # We want: sum_r U[a,r] * VW_flat[r, bc] = T_U[a, bc]
                # This is: VW_flat.T @ U[a,:] = T_U[a,:]
                # So U[a,:] = pinv(VW_flat.T) @ T_U[a,:]
                # Or U = pinv(VW_flat.T) @ T_U.T ... no
                # U is (nm, rank), T_U is (nm, mp*pn), VW_flat is (rank, mp*pn)
                # We want VW_flat.T @ U.T = T_U.T
                # VW_flat.T is (mp*pn, rank), U.T is (rank, nm), T_U.T is (mp*pn, nm)
                # So U.T = pinv(VW_flat.T) @ T_U.T
                # U = pinv(VW_flat.T) @ T_U.T ... T is (nm, rank)
                
                VW_flat_T = VW_flat.T  # (mp*pn, rank)
                U_new = np.linalg.pinv(VW_flat_T) @ T_U.T  # (rank, nm) -> wait no
                # pinv(VW_flat_T) is (rank, mp*pn), T_U.T is (mp*pn, nm)
                # Result is (rank, nm), but U should be (nm, rank)
                # So U_new = (pinv(VW_flat_T) @ T_U.T).T
                
                U_new = (np.linalg.pinv(VW_flat_T) @ T_U.T).T  # (nm, rank)
                U = U_new
                
                # Similarly for V
                UW = np.einsum('ar,cr->acr', U, W)  # (nm, pn, rank)
                UW_flat = UW.transpose(2, 0, 1).reshape(rank, nm * pn)  # (rank, nm*pn)
                T_V = T.transpose(1, 0, 2).reshape(mp, -1)  # (mp, nm*pn)
                UW_flat_T = UW_flat.T  # (nm*pn, rank)
                V_new = (np.linalg.pinv(UW_flat_T) @ T_V.T).T  # (mp, rank)
                V = V_new
                
                # Similarly for W
                UV = np.einsum('ar,br->abr', U, V)  # (nm, mp, rank)
                UV_flat = UV.transpose(2, 0, 1).reshape(rank, nm * mp)  # (rank, nm*mp)
                T_W = T.transpose(2, 0, 1).reshape(pn, -1)  # (pn, nm*mp)
                UV_flat_T = UV_flat.T  # (nm*mp, rank)
                W_new = (np.linalg.pinv(UV_flat_T) @ T_W.T).T  # (pn, rank)
                W = W_new
                
                # Check convergence
                if it % 50 == 0:
                    R = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.max(np.abs(R - T))
                    if err < 1e-10:
                        break
            
            # Check if we found a good solution
            R = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(R - T))
            if err < 1e-8:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                R_check = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                if np.max(np.abs(R_check - T)) < 1e-10:
                    best_rank = rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    break
        
        if best_rank is not None:
            break
    
    if best_rank is not None:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
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
