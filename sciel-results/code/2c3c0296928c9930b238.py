import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    nm, mp, pn = n * m, m * p, p * n  # 6, 30, 20
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Precompute flattened views of T for fast ALS
    # T_flat_u: shape (nm, mp*pn), row a is T[a, :, :].flatten()
    T_flat_u = T.reshape(nm, mp * pn).copy()
    # T_flat_v: shape (nm*pn, mp), col b is T[:, b, :].flatten()
    T_flat_v = np.zeros((nm * pn, mp))
    for b in range(mp):
        T_flat_v[:, b] = T[:, b, :].flatten()
    # T_flat_w: shape (nm*mp, pn), col c is T[:, :, c].flatten()
    T_flat_w = np.zeros((nm * mp, pn))
    for c in range(pn):
        T_flat_w[:, c] = T[:, :, c].flatten()
    
    def kron_vw(V, W, r):
        """Build B_U matrix: (mp*pn) x r where B[:,r] = kron(V[:,r], W[:,r])"""
        B = np.zeros((mp * pn, r))
        for idx in range(r):
            B[:, idx] = np.kron(V[:, idx], W[:, idx])
        return B
    
    def kron_uw(U, W, r):
        """Build B_V matrix: (nm*pn) x r where B[:,r] = kron(U[:,r], W[:,r])"""
        B = np.zeros((nm * pn, r))
        for idx in range(r):
            B[:, idx] = np.kron(U[:, idx], W[:, idx])
        return B
    
    def kron_uv(U, V, r):
        """Build B_W matrix: (nm*mp) x r where B[:,r] = kron(U[:,r], V[:,r])"""
        B = np.zeros((nm * mp, r))
        for idx in range(r):
            B[:, idx] = np.kron(U[:, idx], V[:, idx])
        return B
    
    # Build the rank-55 solution
    def build_rank55():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        c = 0
        # Part A: j=0 (first column of A times first row of B)
        for i in range(n):
            for k in range(p):
                U[i*m+0, c] = 1
                V[0*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
        
        # Part B: A1*B1 + A2*B2 using Strassen for 2x2 blocks
        for b in range(5):
            k0 = 2*b
            k1 = 2*b+1
            
            u11, u12 = 0*m+1, 0*m+2
            u21, u22 = 1*m+1, 1*m+2
            
            v11, v12 = 1*p+k0, 1*p+k1
            v21, v22 = 2*p+k0, 2*p+k1
            
            w00, w01 = k0*n+0, k1*n+0
            w10, w11 = k0*n+1, k1*n+1
            
            # M1 = (a11+a22)(b11+b22)
            U[u11,c]+=1; U[u22,c]+=1; V[v11,c]+=1; V[v22,c]+=1; W[w00,c]+=1; W[w11,c]+=1; c+=1
            # M2 = (a21+a22)b11
            U[u21,c]+=1; U[u22,c]+=1; V[v11,c]+=1; W[w10,c]+=1; W[w11,c]+=-1; c+=1
            # M3 = a11(b12-b22)
            U[u11,c]+=1; V[v12,c]+=1; V[v22,c]+=-1; W[w01,c]+=1; W[w11,c]+=1; c+=1
            # M4 = a22(b21-b11)
            U[u22,c]+=1; V[v21,c]+=1; V[v11,c]+=-1; W[w00,c]+=1; W[w10,c]+=1; c+=1
            # M5 = (a11+a12)b22
            U[u11,c]+=1; U[u12,c]+=1; V[v22,c]+=1; W[w00,c]+=-1; W[w01,c]+=1; c+=1
            # M6 = (a21-a11)(b11+b12)
            U[u21,c]+=1; U[u11,c]+=-1; V[v11,c]+=1; V[v12,c]+=1; W[w11,c]+=1; c+=1
            # M7 = (a12-a22)(b21+b22)
            U[u12,c]+=1; U[u22,c]+=-1; V[v21,c]+=1; V[v22,c]+=1; W[w00,c]+=1; c+=1
        
        return U, V, W, rank
    
    # Verify a decomposition - check ALL entries
    def verify(U, V, W, tol=1e-6):
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T_recon[i*m+j, j*p+k, k*n+i] - 1.0) > tol:
                        return False
        # Check zero entries
        zero_mask = (T == 0)
        max_err = np.max(np.abs(T_recon[zero_mask]))
        return max_err < tol
    
    # Compute error on target entries only (fast)
    def target_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        return err
    
    # Compute full error including zero entries
    def full_error(U, V, W):
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.sum((T_recon - T)**2)
    
    # Fast vectorized ALS
    def als_optimize(U_init, V_init, W_init, target_rank, max_iters=500, tol=1e-12):
        U = U_init[:, :target_rank].copy()
        V = V_init[:, :target_rank].copy()
        W = W_init[:, :target_rank].copy()
        
        prev_err = float('inf')
        stall_count = 0
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U: solve B_U.T @ U.T = T_flat_u.T
            B_U = kron_vw(V, W, target_rank)
            try:
                U_T, _, _, _ = np.linalg.lstsq(B_U.T, T_flat_u.T, rcond=None)
                U = U_T.T
            except:
                pass
            
            # Update V: solve B_V.T @ V.T = T_flat_v
            B_V = kron_uw(U, W, target_rank)
            try:
                V_T, _, _, _ = np.linalg.lstsq(B_V.T, T_flat_v, rcond=None)
                V = V_T.T
            except:
                pass
            
            # Update W: solve B_W.T @ W.T = T_flat_w
            B_W = kron_uv(U, V, target_rank)
            try:
                W_T, _, _, _ = np.linalg.lstsq(B_W.T, T_flat_w, rcond=None)
                W = W_T.T
            except:
                pass
            
            # Compute error on target entries
            if iteration % 3 == 0 or iteration == max_iters - 1:
                err = target_error(U, V, W)
                if err < tol:
                    break
                if abs(prev_err - err) < 1e-15:
                    stall_count += 1
                    if stall_count > 20:
                        break
                else:
                    stall_count = 0
                prev_err = err
        
        return U, V, W
    
    # Exact rational polish: round to nearest 0.5 and verify
    def polish_exact(U, V, W):
        """Round to nearest half-integer and check exactness"""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, tol=1e-9):
            return U_r, V_r, W_r
        # Try rounding to nearest integer
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if verify(U_i, V_i, W_i, tol=1e-9):
            return U_i, V_i, W_i
        # Try mixed: round U to int, V to half, W to half
        U_i2 = np.round(U)
        V_h2 = np.round(V * 2) / 2
        W_h2 = np.round(W * 2) / 2
        if verify(U_i2, V_h2, W_h2, tol=1e-9):
            return U_i2, V_h2, W_h2
        # Try: U half, V int, W half
        U_h3 = np.round(U * 2) / 2
        V_i3 = np.round(V)
        W_h3 = np.round(W * 2) / 2
        if verify(U_h3, V_i3, W_h3, tol=1e-9):
            return U_h3, V_i3, W_h3
        # Try: U half, V half, W int
        U_h4 = np.round(U * 2) / 2
        V_h4 = np.round(V * 2) / 2
        W_i4 = np.round(W)
        if verify(U_h4, V_h4, W_i4, tol=1e-9):
            return U_h4, V_h4, W_i4
        # Try all integers
        U_a = np.round(U)
        V_a = np.round(V)
        W_a = np.round(W)
        if verify(U_a, V_a, W_a, tol=1e-9):
            return U_a, V_a, W_a
        return None
    
    # Coordinate descent polish: fix individual entries to nearest 0.5
    def coord_descent_polish(U, V, W, max_rounds=50):
        """Try to fix the decomposition by adjusting individual entries"""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for rd in range(max_rounds):
            if time.time() > deadline - 8:
                break
            improved = False
            
            # Try rounding each entry of U
            for a in range(nm):
                for r_idx in range(U.shape[1]):
                    old = U[a, r_idx]
                    for candidate in [np.floor(old*2)/2, np.ceil(old*2)/2, np.round(old)]:
                        if candidate == old:
                            continue
                        U[a, r_idx] = candidate
                        if verify(U, V, W, tol=1e-9):
                            return U, V, W
                        U[a, r_idx] = old
            
            # Try rounding each entry of V
            for b in range(mp):
                for r_idx in range(V.shape[1]):
                    old = V[b, r_idx]
                    for candidate in [np.floor(old*2)/2, np.ceil(old*2)/2, np.round(old)]:
                        if candidate == old:
                            continue
                        V[b, r_idx] = candidate
                        if verify(U, V, W, tol=1e-9):
                            return U, V, W
                        V[b, r_idx] = old
            
            # Try rounding each entry of W
            for c in range(pn):
                for r_idx in range(W.shape[1]):
                    old = W[c, r_idx]
                    for candidate in [np.floor(old*2)/2, np.ceil(old*2)/2, np.round(old)]:
                        if candidate == old:
                            continue
                        W[c, r_idx] = candidate
                        if verify(U, V, W, tol=1e-9):
                            return U, V, W
                        W[c, r_idx] = old
        
        return None
    
    # Get initial rank-55 solution
    U55, V55, W55, rank55 = build_rank55()
    if not verify(U55, V55, W55, tol=1e-9):
        # Fallback: return schoolbook rank 60
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W, best_rank = U55, V55, W55, rank55
    
    # Try to find lower-rank solutions using fast ALS
    for target_rank in range(rank55 - 1, 40, -1):
        if time.time() > deadline - 20:
            break
        
        # Multiple random restarts
        for restart in range(8):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(12345 + restart * 7 + target_rank * 13)
            
            if restart == 0:
                # Truncate the rank-55 solution and add small noise
                U_init = U55[:, :target_rank].copy() + 0.05 * np.random.randn(nm, target_rank)
                V_init = V55[:, :target_rank].copy() + 0.05 * np.random.randn(mp, target_rank)
                W_init = W55[:, :target_rank].copy() + 0.05 * np.random.randn(pn, target_rank)
            elif restart == 1:
                # Random initialization
                U_init = np.random.randn(nm, target_rank) * 0.5
                V_init = np.random.randn(mp, target_rank) * 0.5
                W_init = np.random.randn(pn, target_rank) * 0.5
            elif restart == 2:
                # Use best found so far
                U_init = best_U[:, :target_rank].copy() + 0.02 * np.random.randn(nm, target_rank)
                V_init = best_V[:, :target_rank].copy() + 0.02 * np.random.randn(mp, target_rank)
                W_init = best_W[:, :target_rank].copy() + 0.02 * np.random.randn(pn, target_rank)
            elif restart == 3:
                # Different random
                U_init = np.random.randn(nm, target_rank)
                V_init = np.random.randn(mp, target_rank)
                W_init = np.random.randn(pn, target_rank)
            elif restart == 4:
                # Truncate with larger noise
                U_init = U55[:, :target_rank].copy() + 0.15 * np.random.randn(nm, target_rank)
                V_init = V55[:, :target_rank].copy() + 0.15 * np.random.randn(mp, target_rank)
                W_init = W55[:, :target_rank].copy() + 0.15 * np.random.randn(pn, target_rank)
            elif restart == 5:
                # Sparse random
                U_init = np.random.choice([-1, 0, 1], size=(nm, target_rank)).astype(float)
                V_init = np.random.choice([-1, 0, 1], size=(mp, target_rank)).astype(float)
                W_init = np.random.choice([-1, 0, 1], size=(pn, target_rank)).astype(float)
            elif restart == 6:
                # Structured: copy from rank-55 but zero out some columns
                U_init = U55[:, :target_rank].copy()
                V_init = V55[:, :target_rank].copy()
                W_init = W55[:, :target_rank].copy()
                # Add perturbation to last few columns
                for col in range(max(0, target_rank-3), target_rank):
                    U_init[:, col] += 0.1 * np.random.randn(nm)
                    V_init[:, col] += 0.1 * np.random.randn(mp)
                    W_init[:, col] += 0.1 * np.random.randn(pn)
            else:
                # Very small noise
                U_init = U55[:, :target_rank].copy() + 0.01 * np.random.randn(nm, target_rank)
                V_init = V55[:, :target_rank].copy() + 0.01 * np.random.randn(mp, target_rank)
                W_init = W55[:, :target_rank].copy() + 0.01 * np.random.randn(pn, target_rank)
            
            U_opt, V_opt, W_opt = als_optimize(U_init, V_init, W_init, target_rank, max_iters=400)
            
            # Try exact polish
            polished = polish_exact(U_opt, V_opt, W_opt)
            if polished is not None:
                best_U, best_V, best_W, best_rank = polished
                break
            
            # Try coordinate descent polish
            cd_result = coord_descent_polish(U_opt, V_opt, W_opt, max_rounds=30)
            if cd_result is not None:
                best_U, best_V, best_W, best_rank = cd_result
                break
            
            # Also check with looser tolerance
            if verify(U_opt, V_opt, W_opt, tol=1e-6):
                polished = polish_exact(U_opt, V_opt, W_opt)
                if polished is not None:
                    best_U, best_V, best_W, best_rank = polished
                    break
                else:
                    best_U, best_V, best_W, best_rank = U_opt, V_opt, W_opt, target_rank
                    break
        
        if best_rank < rank55:
            continue
    
    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-6):
        best_U, best_V, best_W, best_rank = U55, V55, W55, rank55
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
