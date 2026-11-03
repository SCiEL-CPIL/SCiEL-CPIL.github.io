"""Tensor decomposition search for 2x3x10 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0  # safety margin before 150s kill

def _time_left():
    return DEADLINE - time.time()

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    # Build the matrix multiplication tensor T[i,j,k] where
    # (A @ B)[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] = 1 for the matmul tensor
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # The tensor T has entries T[i,j,k] = 1 (it's the all-ones tensor for matmul)
    # Actually the matmul tensor is: T[i,j,k] = 1 for all i,j,k
    # because (AB)[i,k] = sum_j A[i,j] B[j,k], so the tensor is all-ones.
    
    # Shape of T: (n, m, p) with T[i,j,k] = 1
    T = np.ones((n, m, p))
    
    # We want to find rank-r decomposition:
    # T[i,j,k] = sum_{r=1}^{R} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U has shape (n*m, R), V has shape (m*p, R), W has shape (p*n, R)
    
    # Use RRSVD: flatten T into matrices and find low-rank approximations
    
    best_rank = n * m * p  # schoolbook
    best_decomp = None
    
    # Try ranks from small to large
    # Known: rank of 2x3x10 matmul tensor is at most 60
    # Lower bound: max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # So rank is at least 30
    
    # Strategy: For each candidate rank R, try to find a decomposition
    # using random initialization + RRSVD
    
    for R in range(30, n * m * p + 1):
        if time.time() > DEADLINE:
            break
        
        # Try multiple random initializations
        n_trials = 50 if R < 40 else 20
        found = False
        
        for trial in range(n_trials):
            if time.time() > DEADLINE:
                break
            
            # Random initialization
            np.random.seed(trial * 1000 + R)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # RRSVD iteration: alternate between fixing two factors and
            # solving for the third via SVD
            for it in range(50):
                if time.time() > DEADLINE:
                    break
                
                # Fix U and V, solve for W
                # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # For fixed U, V: reshape to solve for W
                # W[k,i] = (1/R) * sum_r T[i,j,k] * U[i,j,r] * V[j,k,r] / (sum_r' ...)
                # Actually, use the standard RRSVD:
                # Flatten: for each (k,i) pair, W[k,i] should satisfy:
                # sum_{j,r} U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k] for all i,j,k
                # This is: for each r, the outer product U_r * V_r * W_r should sum to T
                
                # RRSVD step 1: Fix V, W. Solve for U.
                # T[i,j,k] = sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r])
                # Let M_r[j,k] = V[j,k,r] * W[k,i,r]... this doesn't work directly
                # because W depends on i.
                
                # Better approach: Use the matrix unfolding.
                # Mode-1 unfolding: U_mat[i, (j,k,r)] = sum... no.
                
                # Let me use a different formulation.
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                # Fix V and W. For each r, define B_r[i,j,k] = V[j*p+k,r]*W[k*n+i,r]
                # Then T[i,j,k] = sum_r U[i*m+j,r] * B_r[i,j,k]
                # This is: for each (j,k), T[:,j,k] = sum_r U[:,j,r] * B_r[:,j,k]
                # i.e., T_jk = U_jk @ diag(B_r) ... not quite.
                
                # Actually: T[i,j,k] = sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r])
                # For fixed j,k: T[:,j,k] = sum_r U[:,j,r] * C_r where C_r = V[j*p+k,r]*W[k*n+i,r]
                # But C_r depends on i through W! So this doesn't separate cleanly.
                
                # Let me use a different approach: gradient-based or direct solve.
                # For fixed V, W: the equation is linear in U.
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For each (i,j,k): sum_r U[i*m+j,r] * (V[j*p+k,r]*W[k*n+i,r]) = T[i,j,k]
                # This is a system: for each (i,j), we have p equations (one per k)
                # sum_r U[i*m+j,r] * D_{i,j,k,r} = T[i,j,k] where D = V[j*p+k,r]*W[k*n+i,r]
                # So for each (i,j): D_{i,j} @ U[i*m+j, :] = T[i,j,:]
                # where D_{i,j} has shape (p, R)
                
                # Solve for U given V, W:
                U_new = np.zeros((n * m, R))
                for i in range(n):
                    for j in range(m):
                        # D has shape (p, R): D[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        D = np.outer(V[j*p:(j+1)*p].T, W[:p*n].reshape(p, n)[:, i].T) if False else \
                            np.array([[V[j*p+k, r] * W[k*n+i, r] for r in range(R)] for k in range(p)])
                        # D @ u = T[i,j,:] where u = U[i*m+j, :]
                        # D is (p, R), we want u of shape (R,)
                        # p equations, R unknowns. If R > p, underdetermined.
                        # Use least squares / minimum norm solution.
                        try:
                            if R <= p:
                                u, _, _, _ = np.linalg.lstsq(D, T[i, j, :], rcond=None)
                            else:
                                # Minimum norm: u = D^T (D D^T)^{-1} T
                                DDT = D @ D.T
                                u = D.T @ np.linalg.solve(DDT + 1e-10 * np.eye(p), T[i, j, :])
                            U_new[i * m + j, :] = u
                        except np.linalg.LinAlgError:
                            U_new[i * m + j, :] = np.linalg.pinv(D) @ T[i, j, :]
                U = U_new
                
                # Fix U and W. Solve for V.
                V_new = np.zeros((m * p, R))
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        # For fixed j,k: T[:,j,k] = sum_r (U[:,j,r] * W[k*n+?, r]) * V[j*p+k,r]
                        # U[:,j,r] = U[i*m+j, r] for each i
                        # W[k*n+i, r] for each i
                        # So: T[i,j,k] = sum_r U[i*m+j,r] * W[k*n+i,r] * V[j*p+k,r]
                        # For each i: T[i,j,k] = sum_r (U[i*m+j,r]*W[k*n+i,r]) * V[j*p+k,r]
                        # This gives n equations for V[j*p+k, :] (R unknowns)
                        D = np.array([[U[i*m+j, r] * W[k*n+i, r] for r in range(R)] for i in range(n)])
                        # D @ v = T[:,j,k] where v = V[j*p+k, :]
                        try:
                            if R <= n:
                                v, _, _, _ = np.linalg.lstsq(D, T[:, j, k], rcond=None)
                            else:
                                DDT = D @ D.T
                                v = D.T @ np.linalg.solve(DDT + 1e-10 * np.eye(n), T[:, j, k])
                            V_new[j * p + k, :] = v
                        except np.linalg.LinAlgError:
                            V_new[j * p + k, :] = np.linalg.pinv(D) @ T[:, j, k]
                V = V_new
                
                # Fix U and V. Solve for W.
                W_new = np.zeros((p * n, R))
                for k in range(p):
                    for i in range(n):
                        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        # For fixed k,i: T[i,:,k] = sum_r (U[i*m+?,r] * V[?*p+k,r]) * W[k*n+i,r]
                        # U[i*m+j, r] for each j, V[j*p+k, r] for each j
                        # So: T[i,j,k] = sum_r (U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r]
                        # m equations for W[k*n+i, :] (R unknowns)
                        D = np.array([[U[i*m+j, r] * V[j*p+k, r] for r in range(R)] for j in range(m)])
                        # D @ w = T[i,:,k] where w = W[k*n+i, :]
                        try:
                            if R <= m:
                                w, _, _, _ = np.linalg.lstsq(D, T[i, :, k], rcond=None)
                            else:
                                DDT = D @ D.T
                                w = D.T @ np.linalg.solve(DDT + 1e-10 * np.eye(m), T[i, :, k])
                            W_new[k * n + i, :] = w
                        except np.linalg.LinAlgError:
                            W_new[k * n + i, :] = np.linalg.pinv(D) @ T[i, :, k]
                W = W_new
            
            # Verify: compute the reconstructed tensor and check exactness
            recon = np.zeros((n, m, p))
            for r in range(R):
                Ur = U[:, r].reshape(n, m)
                Vr = V[:, r].reshape(m, p)
                Wr = W[:, r].reshape(p, n)
                recon += Ur * Vr * Wr[np.newaxis, np.newaxis, :]  # Wr is (p,n), need broadcast
                # Actually: recon[i,j,k] += U[i,j,r]*V[j,k,r]*W[k,i,r]
                # Wr[k,i] so we need Wr.T[i,k]
            # Redo properly
            recon = np.zeros((n, m, p))
            for r in range(R):
                Ur = U[:, r].reshape(n, m)
                Vr = V[:, r].reshape(m, p)
                Wr = W[:, r].reshape(p, n)
                # Wr has indices [k,i], so Wr.T has indices [i,k]
                WrT = Wr.T  # shape (n, p), WrT[i,k] = W[k*n+i, r]
                # recon[i,j,k] += Ur[i,j] * Vr[j,k] * WrT[i,k]
                recon += Ur[:, np.newaxis, :] * Vr[np.newaxis, :, :] * WrT[:, np.newaxis, :]
            
            err = np.max(np.abs(recon - T))
            if err < 1e-8:
                # Found a valid decomposition! Try to round to rationals
                found = True
                # Check if entries are close to integers or half-integers
                all_close = True
                for mat in [U, V, W]:
                    rounded = np.round(mat * 2) / 2.0
                    if np.max(np.abs(mat - rounded)) > 1e-6:
                        all_close = False
                        break
                if all_close:
                    # Convert to exact fractions
                    U_frac = [[Fraction(round(U[i, r] * 2)) / 2 for r in range(R)] for i in range(n * m)]
                    V_frac = [[Fraction(round(V[i, r] * 2)) / 2 for r in range(R)] for i in range(m * p)]
                    W_frac = [[Fraction(round(W[i, r] * 2)) / 2 for r in range(R)] for i in range(p * n)]
                    # Final verification with exact arithmetic
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = Fraction(0)
                                for r in range(R):
                                    s += U_frac[i * m + j][r] * V_frac[j * p + k][r] * W_frac[k * n + i][r]
                                if s != 1:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if ok:
                        best_rank = R
                        best_decomp = {"U": U_frac, "V": V_frac, "W": W_frac}
                        break
                else:
                    # Not close to half-integers, but still valid. Keep as fallback.
                    if best_decomp is None or R < best_rank:
                        best_rank = R
                        # Store as floats for now, will try to rationalize later
                        best_decomp = {"U": U.copy(), "V": V.copy(), "W": W.copy(), "float": True}
        
        if found and not (isinstance(best_decomp.get("U"), list) and best_decomp.get("U") and isinstance(best_decomp["U"][0], list) and isinstance(best_decomp["U"][0][0], Fraction)):
            # We found a valid decomposition at rank R, try to find better
            if best_decomp and best_decomp.get("float"):
                # Try to rationalize the float solution
                pass
        
        if best_decomp and not best_decomp.get("float", False):
            # Found exact rational solution, can stop searching for lower ranks
            # But we should check if we can do better
            break
    
    # If we didn't find an exact rational solution, use the best we have
    if best_decomp is None or best_decomp.get("float", False):
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
    
    return best_decomp

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))
    
    best_rank = n * m * p
    best_decomp = None
    
    for R in range(30, n * m * p + 1):
        if time.time() > DEADLINE:
            break
        
        n_trials = 30 if R <= 35 else 15
        found_exact = False
        
        for trial in range(n_trials):
            if time.time() > DEADLINE:
                break
            
            np.random.seed(trial * 7919 + R * 31)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            for it in range(40):
                if time.time() > DEADLINE:
                    break
                
                U_new = np.zeros((n * m, R))
                for i in range(n):
                    for j in range(m):
                        D = np.array([[V[j*p+k, r] * W[k*n+i, r] for r in range(R)] for k in range(p)])
                        try:
                            if R <= p:
                                u, _, _, _ = np.linalg.lstsq(D, T[i, j, :], rcond=None)
                            else:
                                DDT = D @ D.T
                                u = D.T @ np.linalg.solve(DDT + 1e-12 * np.eye(p), T[i, j, :])
                            U_new[i * m + j, :] = u
                        except:
                            U_new[i * m + j, :] = np.linalg.pinv(D) @ T[i, j, :]
                U = U_new
                
                V_new = np.zeros((m * p, R))
                for j in range(m):
                    for k in range(p):
                        D = np.array([[U[i*m+j, r] * W[k*n+i, r] for r in range(R)] for i in range(n)])
                        try:
                            if R <= n:
                                v, _, _, _ = np.linalg.lstsq(D, T[:, j, k], rcond=None)
                            else:
                                DDT = D @ D.T
                                v = D.T @ np.linalg.solve(DDT + 1e-12 * np.eye(n), T[:, j, k])
                            V_new[j * p + k, :] = v
                        except:
                            V_new[j * p + k, :] = np.linalg.pinv(D) @ T[:, j, k]
                V = V_new
                
                W_new = np.zeros((p * n, R))
                for k in range(p):
                    for i in range(n):
                        D = np.array([[U[i*m+j, r] * V[j*p+k, r] for r in range(R)] for j in range(m)])
                        try:
                            if R <= m:
                                w, _, _, _ = np.linalg.lstsq(D, T[i, :, k], rcond=None)
                            else:
                                DDT = D @ D.T
                                w = D.T @ np.linalg.solve(DDT + 1e-12 * np.eye(m), T[i, :, k])
                            W_new[k * n + i, :] = w
                        except:
                            W_new[k * n + i, :] = np.linalg.pinv(D) @ T[i, :, k]
                W = W_new
            
            recon = np.zeros((n, m, p))
            for r in range(R):
                Ur = U[:, r].reshape(n, m)
                Vr = V[:, r].reshape(m, p)
                Wr = W[:, r].reshape(p, n)
                WrT = Wr.T
                recon += Ur[:, np.newaxis, :] * Vr[np.newaxis, :, :] * WrT[:, np.newaxis, :]
            
            err = np.max(np.abs(recon - T))
            if err < 1e-9:
                # Check for rational (half-integer) entries
                all_close = True
                for mat in [U, V, W]:
                    rounded = np.round(mat * 2) / 2.0
                    if np.max(np.abs(mat - rounded)) > 1e-5:
                        all_close = False
                        break
                if all_close:
                    U_frac = [[Fraction(int(round(U[i, r] * 2))) / 2 for r in range(R)] for i in range(n * m)]
                    V_frac = [[Fraction(int(round(V[i, r] * 2))) / 2 for r in range(R)] for i in range(m * p)]
                    W_frac = [[Fraction(int(round(W[i, r] * 2))) / 2 for r in range(R)] for i in range(p * n)]
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = Fraction(0)
                                for r in range(R):
                                    s += U_frac[i * m + j][r] * V_frac[j * p + k][r] * W_frac[k * n + i][r]
                                if s != 1:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if ok:
                        best_rank = R
                        best_decomp = {"U": U_frac, "V": V_frac, "W": W_frac}
                        found_exact = True
                        break
        
        if found_exact:
            break
    
    if best_decomp is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return best_decomp
# EVOLVE-BLOCK-END
