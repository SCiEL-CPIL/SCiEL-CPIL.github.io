import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Precompute the tensor T_slices
    # T_slices[w, u, v] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    T_slices = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_slices[w, u, v] = 1

    # Store the schoolbook solution as a fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    start_time = time.time()
    time_limit = 110

    # We will try ranks from schoolbook down to a lower bound
    # A heuristic lower bound is max(n*p, m*p, n*m) * something?
    # Let's just try down to 70 or 60.
    min_rank_to_try = 70
    
    for R in range(rank_schoolbook, min_rank_to_try - 1, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random initializations for this rank
        found = False
        num_attempts = 10
        for attempt in range(num_attempts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V
            if attempt == 0:
                # Warm start with schoolbook subset
                U_curr = U_sb[:, :R].copy()
                V_curr = V_sb[:, :R].copy()
            else:
                # Random initialization
                U_curr = np.random.randn(n * m, R)
                V_curr = np.random.randn(m * p, R)
                # Normalize columns to help convergence
                norms = np.linalg.norm(U_curr, axis=0)
                norms[norms < 1e-8] = 1.0
                U_curr /= norms
                norms = np.linalg.norm(V_curr, axis=0)
                norms[norms < 1e-8] = 1.0
                V_curr /= norms
            
            W_curr = np.zeros((p * n, R))
            
            # ALS Optimization
            # We iterate a few times
            max_iters = 20
            for it in range(max_iters):
                # Update W
                # For each w, T_w = sum_r W_wr (U_r V_r^T)
                # vec(T_w) = A_w W_w where A_w has columns vec(U_r V_r^T)
                # Actually, the matrix A is the same for all w if we vectorize the outer product?
                # vec(u v^T) = v \otimes u.
                # So A is a matrix of size (n*m * m*p) x R.
                # A[:, r] = np.outer(V_curr[:, r], U_curr[:, r]).flatten()
                # Note: vec(u v^T) = v \otimes u.
                # T_w is n*m x m*p.
                # T_w[u, v] = sum_r W_wr U_ur V_vr.
                # This is a linear system for W_w.
                
                # Construct A
                # A[i, r] corresponds to the (i-th element of vec(T))
                # vec(T) is flattened row-major.
                # T[u, v] is at index u * (m*p) + v.
                # A[idx, r] = U_curr[u, r] * V_curr[v, r]
                
                # We can build A once per iteration step (when U, V change)
                A = np.zeros((n * m * m * p, R))
                # Vectorize U and V contributions
                # For each r, A[:, r] = np.kron(V_curr[:, r], U_curr[:, r])
                # This is efficient enough for small R.
                for r in range(R):
                    A[:, r] = np.kron(V_curr[:, r], U_curr[:, r])
                
                # Solve for W
                # T_slices is (p*n, n*m, m*p)
                # We can flatten the last two dims
                T_flat = T_slices.reshape(p * n, -1)
                # T_flat[w] = A @ W_curr[w]
                # This is a batched linear system.
                # We can use lstsq for each w, or solve the whole system if we stack?
                # Since A is the same for all w, we can compute the pseudo-inverse of A?
                # No, A is (N x R) where N = n*m*m*p. R is small.
                # We want to minimize || T_flat[w] - A W_w ||^2 for all w.
                # This is equivalent to minimizing || T_flat - W_curr A^T ||_F^2?
                # Let's check: (W_curr A^T)[w, idx] = sum_r W_curr[w, r] A[idx, r].
                # Yes.
                # So we want to find W_curr that minimizes || T_flat - W_curr A^T ||_F.
                # This is a standard least squares problem: W_curr = T_flat @ A_pinv
                # Where A_pinv = (A^T A)^{-1} A^T.
                
                try:
                    # Compute pseudo inverse of A^T? Or just solve W A^T = T_flat
                    # W_curr = T_flat @ A.T @ inv(A @ A.T)
                    # Let's use lstsq on the whole thing?
                    # np.linalg.lstsq(A.T, T_flat.T, rcond=None) gives W_curr.T?
                    # Let's use the formula: W_curr = T_flat @ np.linalg.pinv(A)
                    # pinv(A) is R x N.
                    A_pinv = np.linalg.pinv(A)
                    W_curr = T_flat @ A_pinv
                except np.linalg.LinAlgError:
                    W_curr = np.zeros((p * n, R))
                    break

                # Update U
                # Fix V, W. Minimize || T - sum_r W_wr U_r V_r^T ||_F
                # For each u, T[:, u, :] = sum_r U_ur (V_r W[:, r]^T)
                # Let B_u be a matrix of size (m*p * p*n) x R?
                # T[:, u, :] is (p*n, m*p).
                # T[:, u, :] = sum_r U_ur (V_r W[:, r]^T)
                # vec(T[:, u, :]) = sum_r U_ur vec(V_r W[:, r]^T)
                # vec(V_r W[:, r]^T) = W[:, r] \otimes V_r
                # So we form a matrix B of size (m*p * p*n) x R
                # B[:, r] = np.kron(W_curr[:, r], V_curr[:, r])
                # Then T_flat_u = B @ U_curr[:, u]
                # We can stack all u into a big system?
                # T_slices is (p*n, n*m, m*p).
                # Let's transpose to (n*m, p*n, m*p) to group by u.
                T_u = T_slices.transpose(1, 0, 2) # (n*m, p*n, m*p)
                T_u_flat = T_u.reshape(n * m, -1) # (n*m, p*n * m*p)
                
                B = np.zeros((p * n * m * p, R))
                for r in range(R):
                    B[:, r] = np.kron(W_curr[:, r], V_curr[:, r])
                
                B_pinv = np.linalg.pinv(B)
                U_curr = T_u_flat @ B_pinv

                # Update V
                # Fix U, W. Minimize || T - sum_r W_wr U_r V_r^T ||_F
                # For each v, T[:, :, v] = sum_r V_vr (U_r W[:, r]^T)
                # vec(T[:, :, v]) = sum_r V_vr vec(U_r W[:, r]^T)
                # vec(U_r W[:, r]^T) = W[:, r] \otimes U_r
                # Matrix C of size (n*m * p*n) x R
                # C[:, r] = np.kron(W_curr[:, r], U_curr[:, r])
                # T_v_flat = C @ V_curr[:, v]
                
                T_v = T_slices.transpose(0, 2, 1) # (p*n, m*p, n*m)
                # We want to group by v (index in m*p)
                # T_slices is (w, u, v).
                # T[:, :, v] is (p*n, n*m).
                # Let's transpose to (m*p, p*n, n*m)
                T_v = T_slices.transpose(2, 0, 1)
                T_v_flat = T_v.reshape(m * p, -1)
                
                C = np.zeros((n * m * p * n, R))
                for r in range(R):
                    C[:, r] = np.kron(W_curr[:, r], U_curr[:, r])
                
                C_pinv = np.linalg.pinv(C)
                V_curr = T_v_flat @ C_pinv

            # After ALS, check residual
            # Reconstruct T
            # T_recon[w, u, v] = sum_r W[w, r] U[u, r] V[v, r]
            # We can compute this efficiently.
            # T_recon = np.einsum('wr,ur,vr->wuv', W_curr, U_curr, V_curr)
            # This might be memory heavy if p*n is large, but here 18*15*30 = 8100, R~90.
            # 18*15*30*90 ~ 730k ops. Very fast.
            
            T_recon = np.einsum('wr,ur,vr->wuv', W_curr, U_curr, V_curr)
            diff = np.abs(T_recon - T_slices)
            max_diff = np.max(diff)
            
            if max_diff < 1e-6:
                # Try to round to half-integers
                # The ALS solution might be floating point.
                # We need to find half-integers U', V', W' such that T' = T.
                # If the solution is close to integers/half-integers, rounding might work.
                U_rounded = np.round(U_curr * 2) / 2
                V_rounded = np.round(V_curr * 2) / 2
                W_rounded = np.round(W_curr * 2) / 2
                
                T_recon_rounded = np.einsum('wr,ur,vr->wuv', W_rounded, U_rounded, V_rounded)
                diff_rounded = np.abs(T_recon_rounded - T_slices)
                if np.max(diff_rounded) < 1e-9:
                    best_rank = R
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    found = True
                    break

        if found:
            # We found a solution for this R.
            # Can we go lower?
            # Given the time constraint, we might want to continue to see if we can go lower.
            # But if we found 90, we should try 89.
            # If we found 89, try 88.
            # The loop naturally continues to R-1.
            # We break the inner loop, but the outer loop continues.
            # However, if we found a solution, we update best_rank.
            # We should NOT break the outer loop unless we hit the minimum rank we want to try.
            # But wait, if we found a solution for R, it's the best so far.
            # We continue to R-1.
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
