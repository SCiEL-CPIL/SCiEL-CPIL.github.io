import time
import numpy as np
from scipy.optimize import lsq_linear

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    n, m, p = 2, 5, 5
    
    # Schoolbook construction
    def build_schoolbook(rank):
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Fast validation using random matrices
    def is_valid(U, V, W, tol=1e-8, seed=42):
        np.random.seed(seed)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        R = U.shape[1]
        # S_r = L_r(A) * M_r(B)
        # L_r(A) = sum_{i,j} U_{ij,r} A_{ij} = np.dot(U[:,r], A.flatten())
        # M_r(B) = sum_{j,k} V_{jk,r} B_{jk} = np.dot(V[:,r], B.flatten())
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        L = U @ A_flat  # shape (R,)
        M = V @ B_flat  # shape (R,)
        S = L * M       # shape (R,)
        
        # C_rec[i, k] = sum_r W_{k*n+i, r} S_r
        # W is (p*n, R). We need W[k*n+i, :] for each i,k.
        # Let's reshape W to (p, n, R) and index W[k, i, :]
        W_reshaped = W.T.reshape(p, n, R) # W.T is (R, p*n). Reshape to (R, p, n) then transpose?
        # W[idx, r] where idx = k*n + i.
        # W_reshaped[r, k, i] = W[k*n+i, r].
        # So we want sum_r S[r] * W_reshaped[r, k, i].
        
        C_rec = np.zeros((n, p))
        for k in range(p):
            for i in range(n):
                C_rec[i, k] = np.dot(S, W_reshaped[:, k, i])
        
        return np.allclose(C, C_rec, atol=tol)

    # ALS Step
    def als_step(U, V, W, R):
        # Solve for U
        # Equations: sum_r W_{ki,r} U_{ab,r} V_{cd,r} = delta_{a,i} delta_{d,k} delta_{b,c}
        # For fixed V, W, this is linear in U.
        # Variables: U_{ab, r}.
        # We can group by (a, b).
        # For a specific (a, b), the variables are U_{a, b, r} for r=0..R-1.
        # The equations involving U_{a, b, r} are those where the target is non-zero.
        # Target non-zero if a=i, d=k, b=c.
        # So for fixed a, b:
        #   i must be a.
        #   c must be b.
        #   k is free (0..p-1).
        #   d must be k.
        # So for each k, we have an equation:
        #   sum_r W_{k, a, r} U_{a, b, r} V_{b, k, r} = 1  (if we normalize target to 1)
        #   Wait, the target is 1 for valid (i,k,a,b,c,d).
        #   Here i=a, c=b, d=k.
        #   So sum_r W_{k n + a, r} U_{a m + b, r} V_{b p + k, r} = 1.
        
        U_new = np.zeros_like(U)
        for a in range(n):
            for b in range(m):
                # Vector of unknowns: u_r = U[a*m+b, r]
                # We have p equations (for each k)
                # Matrix A_eq of shape (p, R)
                # A_eq[k, r] = W[k*n+a, r] * V[b*p+k, r]
                A_eq = np.zeros((p, R))
                for k in range(p):
                    A_eq[k, :] = W[k*n+a, :] * V[b*p+k, :]
                b_vec = np.ones(p)
                # Solve A_eq @ u = b_vec
                u_sol, residuals, rank, sv = np.linalg.lstsq(A_eq, b_vec, rcond=None)
                U_new[a*m+b, :] = u_sol
        
        # Solve for V
        V_new = np.zeros_like(V)
        # Equations: sum_r W_{ki,r} U_{ab,r} V_{cd,r} = 1
        # For fixed c, d (V indices):
        #   Target non-zero if c=b, d=k, a=i.
        #   So for fixed c, d:
        #     k must be d.
        #     b must be c.
        #     i is free (0..n-1).
        #     a must be i.
        #   Equation for each i:
        #   sum_r W_{d n + i, r} U_{i m + c, r} V_{c p + d, r} = 1.
        
        for c in range(m):
            for d in range(p):
                A_eq = np.zeros((n, R))
                for i in range(n):
                    A_eq[i, :] = W[d*n+i, :] * U[i*m+c, :]
                b_vec = np.ones(n)
                v_sol, _, _, _ = np.linalg.lstsq(A_eq, b_vec, rcond=None)
                V_new[c*p+d, :] = v_sol
        
        # Solve for W
        W_new = np.zeros_like(W)
        # Equations: sum_r W_{ki,r} U_{ab,r} V_{cd,r} = 1
        # For fixed k, i (W indices):
        #   Target non-zero if d=k, a=i, b=c.
        #   So for fixed k, i:
        #     d must be k.
        #     a must be i.
        #     b=c is free (0..m-1).
        #   Equation for each b:
        #   sum_r W_{k n + i, r} U_{i m + b, r} V_{b p + k, r} = 1.
        
        for k in range(p):
            for i in range(n):
                A_eq = np.zeros((m, R))
                for b in range(m):
                    A_eq[b, :] = U[i*m+b, :] * V[b*p+k, :]
                b_vec = np.ones(m)
                w_sol, _, _, _ = np.linalg.lstsq(A_eq, b_vec, rcond=None)
                W_new[k*n+i, :] = w_sol
        
        return U_new, V_new, W_new

    # Try to find a solution of rank R
    def try_rank(R, max_iter=50, n_restarts=5):
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                return None
            
            # Initialize
            # Random init scaled
            scale = 1.0 / np.sqrt(R)
            U = np.random.randn(n*m, R) * scale
            V = np.random.randn(m*p, R) * scale
            W = np.random.randn(p*n, R) * scale
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    return None
                
                U, V, W = als_step(U, V, W, R)
                
                # Check validity
                if is_valid(U, V, W, tol=1e-6):
                    return U, V, W
        
        return None

    # Start with schoolbook
    best_U, best_V, best_W = build_schoolbook(n*m*p)
    best_rank = n*m*p
    
    # Try lower ranks
    # Start from 49 downwards
    for R in range(best_rank - 1, max(best_rank - 15, 40), -1):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(R)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = R
            break # If found, stop. We want minimal rank, so first found (lowest R) is best?
            # Wait, we iterate R downwards. So first found is the lowest R we tried.
            # But maybe we can go lower?
            # The loop continues to lower R if we don't break.
            # But we break on success.
            # Should we break?
            # If we find rank 49, we should check if 48 is possible.
            # But finding 49 is already an improvement.
            # Given time constraints, breaking on first success is safe.
            # However, if 49 is easy and 48 is hard, we might want to try 48.
            # But 49 is a significant jump from 50.
            # Let's not break, but prioritize lower ranks.
            # Actually, if we find 49, the score is 50/49.
            # If we find 48, score is 50/48.
            # We should try to go as low as possible.
            # So do NOT break.
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
