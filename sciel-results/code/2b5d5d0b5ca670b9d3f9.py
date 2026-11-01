import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def verify(U, V, W, n=2, m=3, p=10, trials=10):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        a_vals = U.T @ A.flatten()
        b_vals = V.T @ B.flatten()
        vec = W.T @ (a_vals * b_vals)
        C_dec = vec.reshape(p, n).T
        if not np.allclose(C, C_dec, atol=1e-8):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 10s margin
    
    # Schoolbook baseline
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Try to find lower rank decompositions
    # Use alternating least squares with random restarts
    for target_rank in range(20, rank0):
        if time.time() > deadline - 5:
            break
        
        found = False
        max_trials = 100 if target_rank < 40 else 30
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U_t = np.random.randn(n*m, target_rank)
            V_t = np.random.randn(m*p, target_rank)
            W_t = np.random.randn(p*n, target_rank)
            
            # Alternating minimization
            for it in range(100):
                if time.time() > deadline - 3:
                    break
                
                # Generate random test points
                num_pts = 15
                A_list = np.random.randn(num_pts, n, m)
                B_list = np.random.randn(num_pts, m, p)
                C_list = np.einsum('pim,pjk->pik', A_list, B_list)
                
                # Solve for U (fix V, W)
                # For each (pt, i, k): sum_r U[:,r]@A_flat * (V[:,r]@B_flat) * W[k*n+i,r] = C[pt,i,k]
                # Linear in U.
                num_eqs = num_pts * n * p
                num_unknowns = n*m * target_rank
                
                A_flat_list = A_list.reshape(num_pts, n*m)
                B_flat_list = B_list.reshape(num_pts, m*p)
                
                # Precompute V @ B_flat for each r, pt
                # V_t shape (m*p, target_rank), B_flat_list shape (num_pts, m*p)
                # v_b[pt, r] = V_t[:,r] @ B_flat_list[pt,:]
                v_b = B_flat_list @ V_t  # shape (num_pts, target_rank)
                # w_vals[pt, i, k, r] = W_t[k*n+i, r]
                # W_t shape (p*n, target_rank)
                w_reshaped = W_t.reshape(p, n, target_rank)  # w_reshaped[k, i, r]
                
                # Build equation matrix for U
                # Row (pt, i, k): for unknown (idx, r): A_flat_list[pt, idx] * v_b[pt, r] * w_reshaped[k, i, r]
                # This is a big matrix. Let's build it efficiently.
                # Shape: (num_eqs, num_unknowns)
                
                # For each r, the contribution to the row is:
                # A_flat_list[pt, idx] * v_b[pt, r] * w_reshaped[k, i, r]
                # = (A_flat_list[pt, idx]) * (v_b[pt, r] * w_reshaped[k, i, r])
                
                # We can write this as: for each r, a matrix M_r of shape (num_eqs, n*m)
                # where M_r[pt*n*p + i*p + k, idx] = A_flat_list[pt, idx] * v_b[pt, r] * w_reshaped[k, i, r]
                
                # This is expensive to build explicitly. Let me use a different approach.
                # Instead, solve the linear system using the structure.
                
                # Actually, for small dimensions, let's just build the matrix.
                # num_eqs = 15 * 2 * 10 = 300
                # num_unknowns = 6 * target_rank (at most 6*59 = 354)
                # So the matrix is at most 300 x 354, which is fine.
                
                eqs_U = np.zeros((num_eqs, num_unknowns))
                rhs_U = np.zeros(num_eqs)
                
                for pt in range(num_pts):
                    for i in range(n):
                        for k in range(p):
                            eq_idx = pt * n * p + i * p + k
                            rhs_U[eq_idx] = C_list[pt, i, k]
                            for r in range(target_rank):
                                vb = v_b[pt, r]
                                wv = w_reshaped[k, i, r]
                                if vb == 0 or wv == 0:
                                    continue
                                # Coefficient for U[idx, r] is A_flat_list[pt, idx] * vb * wv
                                eqs_U[eq_idx, r*n*m:(r+1)*n*m] = A_flat_list[pt, :] * (vb * wv)
                
                # Solve
                U_flat, _, _, _ = np.linalg.lstsq(eqs_U, rhs_U, rcond=None)
                U_t = U_flat.reshape(n*m, target_rank)
                
                # Solve for V (fix U, W)
                u_a = A_flat_list @ U_t  # shape (num_pts, target_rank)
                eqs_V = np.zeros((num_eqs, m*p * target_rank))
                rhs_V = np.zeros(num_eqs)
                
                for pt in range(num_pts):
                    for i in range(n):
                        for k in range(p):
                            eq_idx = pt * n * p + i * p + k
                            rhs_V[eq_idx] = C_list[pt, i, k]
                            for r in range(target_rank):
                                ua = u_a[pt, r]
                                wv = w_reshaped[k, i, r]
                                if ua == 0 or wv == 0:
                                    continue
                                eqs_V[eq_idx, r*m*p:(r+1)*m*p] = B_flat_list[pt, :] * (ua * wv)
                
                V_flat, _, _, _ = np.linalg.lstsq(eqs_V, rhs_V, rcond=None)
                V_t = V_flat.reshape(m*p, target_rank)
                
                # Solve for W (fix U, V)
                v_b2 = B_flat_list @ V_t  # shape (num_pts, target_rank)
                eqs_W = np.zeros((num_eqs, p*n * target_rank))
                rhs_W = np.zeros(num_eqs)
                
                for pt in range(num_pts):
                    for i in range(n):
                        for k in range(p):
                            eq_idx = pt * n * p + i * p + k
                            rhs_W[eq_idx] = C_list[pt, i, k]
                            for r in range(target_rank):
                                ua = u_a[pt, r]
                                vb = v_b2[pt, r]
                                if ua == 0 or vb == 0:
                                    continue
                                eqs_W[eq_idx, (k*n+i)*target_rank + r] = ua * vb
                
                W_flat, _, _, _ = np.linalg.lstsq(eqs_W, rhs_W, rcond=None)
                W_t = W_flat.reshape(p*n, target_rank)
                
                # Periodic check
                if it % 30 == 0 and it > 0:
                    if verify(U_t.tolist(), V_t.tolist(), W_t.tolist(), trials=3):
                        found = True
                        break
            
            if found:
                if verify(U_t.tolist(), V_t.tolist(), W_t.tolist(), trials=20):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
                    break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
