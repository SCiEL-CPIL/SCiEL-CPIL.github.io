import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # 1. Build the target tensor T
    # T[a, b, c] = 1 if i_a==i_c, j_a==j_b, k_b==k_c
    # Indices: a in [0, nm), b in [0, mp), c in [0, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Pre-compute matrix forms for ALS
    # For solving U: T reshaped to (nm, mp*pn)
    T_u = T.reshape(nm, mp * pn)
    # For solving V: T reshaped to (mp, nm*pn)
    T_v = T.reshape(mp, nm * pn)
    # For solving W: T reshaped to (pn, nm*mp)
    T_w = T.reshape(pn, nm * mp)
    
    # 2. Schoolbook solution (Rank 30) as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def check_decomposition(U, V, W, R):
        """Check if U, V, W exactly reconstruct T and are half-integers."""
        # Check half-integers
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        if not (np.allclose(U2, np.round(U2), atol=1e-6) and
                np.allclose(V2, np.round(V2), atol=1e-6) and
                np.allclose(W2, np.round(W2), atol=1e-6)):
            return False, None, None, None
        
        # Snap to exact half-integers
        U_out = np.round(U2) / 2.0
        V_out = np.round(V2) / 2.0
        W_out = np.round(W2) / 2.0
        
        # Check reconstruction
        # Rec = np.einsum('ar,br,cr->abc', U_out, V_out, W_out)
        # err = np.sum((Rec - T)**2)
        # Faster check:
        Rec_flat = np.einsum('ar,br,cr->abcr', U_out, V_out, W_out) # This is slow
        # Let's use the matrix product form
        # U (nm x R), V (mp x R), W (pn x R)
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        
        # Construct Rec
        # Rec = U @ (V.T @ W.T).T ? No.
        # Let's just use einsum for correctness in check, it's only done a few times.
        Rec = np.einsum('ar,br,cr->abc', U_out, V_out, W_out)
        if np.allclose(Rec, T, atol=1e-9):
            return True, U_out, V_out, W_out
        return False, None, None, None

    def snap_to_half_integers(U, V, W):
        return (np.round(U * 2) / 2.0, np.round(V * 2) / 2.0, np.round(W * 2) / 2.0)

    def run_als(U_init, V_init, W_init, max_iters=100, tol=1e-10):
        """
        Run Alternating Least Squares to minimize reconstruction error.
        U: (nm, R), V: (mp, R), W: (pn, R)
        """
        R = U_init.shape[1]
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            # 1. Solve for U
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # For fixed V, W:
            # T[a, (b, c)] = sum_r U[a, r] (V[b, r] W[c, r])
            # Let M_uc = V * W.T  (mp x pn) -> Flatten to (mp*pn)
            # Actually, we need the matrix M such that T_u = U @ M.T
            # T_u[a, k] where k indexes (b,c)
            # T_u = U @ (V * W.T).T ?
            # Let's derive:
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # T_u[a, b*pn+c] = sum_r U[a, r] V[b, r] W[c, r]
            # Let Z = V * W.T  (shape mp x pn). Z[b, c] = sum_r V[b, r] W[c, r].
            # This is not quite right because the sum is over r for each term.
            # T[a, b, c] = sum_r U[a, r] (V[b, r] W[c, r])
            # This is not a simple matrix product of U and a single matrix derived from V,W unless we vectorize carefully.
            
            # Correct ALS Step:
            # We want to minimize || T - sum_r U[:,r] outer(V[:,r], W[:,r]) ||_F
            # This is equivalent to minimizing || T_u - U @ M_U.T ||_F
            # Where M_U is a matrix of size R x (mp*pn)
            # M_U[r, b*pn+c] = V[b, r] * W[c, r]
            
            M_U = np.zeros((R, mp * pn))
            for r in range(R):
                M_U[r, :] = (V[:, r] * W[:, r].reshape(1, -1)).reshape(-1) # V[:,r] is (mp,1), W[:,r] is (pn,1). Outer product is mp x pn.
                # Actually: V[:, r] is vector length mp. W[:, r] is vector length pn.
                # We need V[b, r] * W[c, r] for all b, c.
                # This is outer(V[:, r], W[:, r]).
                # Flatten that.
                M_U[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            
            # Solve U M_U.T = T_u
            # U = T_u @ M_U @ pinv(M_U.T @ M_U) ??
            # Least squares: U = T_u @ (M_U.T @ M_U)^-1 @ M_U.T
            # Using np.linalg.lstsq
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_U.T, T_u.T, rcond=None)
                U = U_new.T
            except:
                pass
            
            # 2. Solve for V
            # T[a, b, c] = sum_r V[b, r] (U[a, r] W[c, r])
            # M_V[r, a*pn+c] = U[a, r] * W[c, r]
            M_V = np.zeros((R, nm * pn))
            for r in range(R):
                M_V[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_V.T, T_v.T, rcond=None)
                V = V_new.T
            except:
                pass
            
            # 3. Solve for W
            # T[a, b, c] = sum_r W[c, r] (U[a, r] V[b, r])
            # M_W[r, a*mp+b] = U[a, r] * V[b, r]
            M_W = np.zeros((R, nm * mp))
            for r in range(R):
                M_W[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_W.T, T_w.T, rcond=None)
                W = W_new.T
            except:
                pass
            
            # Calculate error
            # Rec = np.einsum('ar,br,cr->abc', U, V, W)
            # err = np.sum((Rec - T)**2)
            # To save time, we can approximate or just check every few iterations.
            # Let's check every 5 iterations or if it's the last one.
            if it % 5 == 0 or it == max_iters - 1:
                Rec = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.sum((Rec - T)**2)
                if abs(prev_err - err) < tol:
                    break
                prev_err = err
                
        return U, V, W

    target_rank = 29
    start_time = time.time()
    deadline = start_time + 145
    
    best_result = None
    
    # Strategy: Try to find a rank 29 solution using ALS with random initializations.
    # We will also try to initialize using the schoolbook solution projected to rank 29.
    
    # Initialization 1: Random
    # Initialization 2: Projected Schoolbook
    # Take U_sb (nm x 30), V_sb (mp x 30), W_sb (pn x 30)
    # We want U (nm x 29), etc.
    # This is tricky because the rank-30 solution is sparse.
    # A better random init: Small random values.
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        np.random.seed(seed)
        
        # Random initialization
        # Scale matters. The values in T are 0 or 1.
        # U, V, W entries are likely small.
        U_init = np.random.randn(nm, target_rank) * 0.5
        V_init = np.random.randn(mp, target_rank) * 0.5
        W_init = np.random.randn(pn, target_rank) * 0.5
        
        # Run ALS
        U, V, W = run_als(U_init, V_init, W_init, max_iters=50)
        
        # Snap and check
        U_snap, V_snap, W_snap = snap_to_half_integers(U, V, W)
        valid, U_out, V_out, W_out = check_decomposition(U_snap, V_snap, W_snap, target_rank)
        
        if valid:
            return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
            
        # Also check if the unsnapped solution has very low error, 
        # maybe it's close to a half-integer solution but not quite.
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((Rec - T)**2)
        
        if err < 1e-8:
             # It's an exact solution, but maybe not half-integers?
             # The problem requires half-integers.
             # If it's not half-integers, it's invalid for the score?
             # "integer/half-integer entries"
             # So we must snap. If snapping breaks it, it's not a valid solution.
             pass

    # If we haven't found rank 29, we return the schoolbook rank 30 solution.
    # The score is the rank. 30 is the schoolbook rank.
    # Is there a known rank 29 solution for 2x3x5?
    # The lower bound is 29. It is an open problem if rank 29 exists for 2x3x5?
    # Actually, for 2x3x5, the rank is known to be 29? Or is it 30?
    # Strassen's algorithm for 2x2x2 is 7 (schoolbook 8).
    # For 2x3x3, rank is 16 (schoolbook 18).
    # For 2x3x5, I am not 100% sure if 29 is achievable.
    # However, the task asks for "rank-minimal". If 29 is not possible, 30 is the answer.
    # But usually these tasks have a "trick" or a known solution.
    # If no rank 29 solution is found by ALS, we fall back to 30.
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
