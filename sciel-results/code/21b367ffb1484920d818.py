import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    # Build the target tensor T[i, j, k] = 1 for all (i,j,k)
    T = np.ones((n, m, p))
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill

    def compute_tensor(U, V, W, rank):
        """Compute the tensor from decomposition."""
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        return np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    def compute_error(U, V, W, rank):
        """Compute the error of the tensor decomposition."""
        result = compute_tensor(U, V, W, rank)
        return np.linalg.norm(result - T)
    
    def verify_exact(U, V, W, rank):
        """Verify exact decomposition with tight tolerance."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    max_err = max(max_err, abs(val - 1.0))
        return max_err

    def polish(U, V, W, rank):
        """Gauss-Newton polish step."""
        # Flatten
        x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Precompute indices for Jacobian construction
        # Residuals are indexed by (i, j, k)
        # Parameters:
        # U: (n, m, rank) -> idx u(i,j,r)
        # V: (m, p, rank) -> idx v(j,k,r)
        # W: (p, n, rank) -> idx w(k,i,r)
        
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        for _ in range(50):
            if time.time() - start_time > deadline:
                break
            
            # Compute residual
            result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            res = result - T
            res_flat = res.flatten()
            
            if np.linalg.norm(res_flat) < 1e-12:
                break
            
            # Compute Jacobian J (18 x 21*rank)
            # This is small, so we can just build it
            J = np.zeros((n*m*p, (n*m + m*p + p*n) * rank))
            
            # Derivatives w.r.t U
            # d r(i,j,k) / d U(i',j',r) = delta(i,i') delta(j,j') V(j,k,r) W(k,i,r)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r_idx = i*m*p + j*p + k
                        for r in range(rank):
                            # dU
                            col_u = i*m + j
                            J[r_idx, col_u * rank + r] = V_r[j, k, r] * W_r[k, i, r]
                            
                            # dV
                            col_v = (i*m + j) * rank + (j*p + k) * rank # Wait, V is separate block
                            # Block sizes: U_size = n*m*rank, V_size = m*p*rank, W_size = p*n*rank
                            u_size = n * m * rank
                            v_size = m * p * rank
                            
                            # Index in J for V(j,k,r)
                            col_v = u_size + (j*p + k) * rank + r
                            J[r_idx, col_v] = U_r[i, j, r] * W_r[k, i, r]
                            
                            # Index in J for W(k,i,r)
                            col_w = u_size + v_size + (k*n + i) * rank + r
                            J[r_idx, col_w] = U_r[i, j, r] * V_r[j, k, r]
            
            # Solve (J^T J + lambda I) dx = -J^T res
            # Use Levenberg-Marquardt style damping
            JTJ = J.T @ J
            JTr = J.T @ res_flat
            
            # Damping parameter
            lam = 1e-6
            for _ in range(10):
                try:
                    dx = np.linalg.solve(JTJ + lam * np.eye(JTJ.shape[0]), -JTr)
                    break
                except np.linalg.LinAlgError:
                    lam *= 10
            
            x_new = x + dx
            
            # Unpack
            u_size = n * m * rank
            v_size = m * p * rank
            w_size = p * n * rank
            
            U_new = x_new[:u_size].reshape(n, m, rank)
            V_new = x_new[u_size:u_size+v_size].reshape(m, p, rank)
            W_new = x_new[u_size+v_size:].reshape(p, n, rank)
            
            # Check improvement
            res_new = np.einsum('ijr,jkr,kir->ijk', U_new, V_new, W_new) - T
            if np.linalg.norm(res_new) < np.linalg.norm(res_flat):
                U, V, W = U_new, V_new, W_new
                U_r, V_r, W_r = U.reshape(n, m, rank), V.reshape(m, p, rank), W.reshape(p, n, rank)
                lam = max(lam * 0.5, 1e-10)
            else:
                lam *= 10
                if lam > 1e10:
                    break
            
        return U, V, W

    def try_rank(target_rank, max_restarts=50):
        """Try to find a decomposition of given rank."""
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                return None, None, None, None
            
            rng = np.random.RandomState(restart * 7919 + target_rank * 104729 + 12345)
            
            # Initialize
            U = rng.randn(n * m, target_rank)
            V = rng.randn(m * p, target_rank)
            W = rng.randn(p * n, target_rank)
            
            # Scale to avoid overflow/underflow initially
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Vectorized ALS
            for iter_ in range(100):
                if time.time() - start_time > deadline:
                    return None, None, None, None
                
                U_r = U.reshape(n, m, target_rank)
                V_r = V.reshape(m, p, target_rank)
                W_r = W.reshape(p, n, target_rank)
                
                # Update U
                # For each i, j: Solve B u = t
                # B[k, r] = V_r[j, k, r] * W_r[k, i, r]
                # t[k] = T[i, j, k]
                for i in range(n):
                    for j in range(m):
                        B = V_r[j, :, :] * W_r[:, i, :] # (p, R)
                        t = T[i, j, :] # (p,)
                        # Solve B u = t
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(B, t, rcond=None)[0]
                        except:
                            pass
                
                # Update V
                # For each j, k: Solve A v = t
                # A[i, r] = U_r[i, j, r] * W_r[k, i, r]
                # t[i] = T[i, j, k]
                for j in range(m):
                    for k in range(p):
                        A = U_r[:, j, :] * W_r[k, :, :] # (n, R)
                        t = T[:, j, k] # (n,)
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(A, t, rcond=None)[0]
                        except:
                            pass
                
                # Update W
                # For each k, i: Solve C w = t
                # C[j, r] = U_r[i, j, r] * V_r[j, k, r]
                # t[j] = T[i, j, k]
                for k in range(p):
                    for i in range(n):
                        C = U_r[i, :, :] * V_r[:, k, :] # (m, R)
                        t = T[i, :, k] # (m,)
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(C, t, rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                err = compute_error(U, V, W, target_rank)
                if err < 1e-10:
                    break
            
            # Polish
            U, V, W = polish(U, V, W, target_rank)
            
            # Verify
            max_err = verify_exact(U, V, W, target_rank)
            if max_err < 1e-6:
                return U, V, W, max_err
        
        return None, None, None, None
    
    # Try ranks from 11 upward (Landsberg bound for 2x3x3 is 11)
    for target_rank in range(11, R_school + 1):
        if time.time() - start_time > deadline:
            break
        
        max_restarts = 100 if target_rank <= 12 else 50
        U, V, W, err = try_rank(target_rank, max_restarts)
        
        if U is not None and err < 1e-6:
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break
    
    # Final verification
    if best_rank < R_school:
        final_err = verify_exact(best_U, best_V, best_W, best_rank)
        if final_err > 1e-4:
            best_U, best_V, best_W = U_school, V_school, W_school
            best_rank = R_school
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
