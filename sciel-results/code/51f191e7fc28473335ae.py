import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    deadline = time.time() + 110  # safety margin

    def verify(U_mat, V_mat, W_mat):
        """Verify exact decomposition. Returns True if valid."""
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            for ip in range(n):
                                s = 0
                                for c in range(W_mat.shape[1]):
                                    s += U_mat[i*m+j, c] * V_mat[jp*p+k, c] * W_mat[kp*n+ip, c]
                                expected = 1 if (j == jp and k == kp and i == ip) else 0
                                if s != expected:
                                    return False
        return True

    def try_rank(r, max_trials=500):
        nonlocal best_rank, best_U, best_V, best_W
        if r >= best_rank:
            return False
        
        # Precompute structure for linear system
        # Unknowns: W[k'*n + i', c] -> index (k'*n + i') * r + c
        num_unknowns = p * n * r
        
        # Equations: (i, j, jp, k, kp, ip)
        # Note: In verify, indices are i, j (U); jp, k (V); kp, ip (W).
        # Condition: i=ip, j=jp, k=kp.
        # So for each eq (i, j, jp, k, kp, ip), RHS is 1 if i=ip, j=jp, k=kp else 0.
        # LHS: sum_c U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c]
        
        rows = []
        cols = []
        vals = []
        b_vec = []
        
        # We need to map (i, j, jp, k, kp, ip) to a row index
        # And (kp, ip, c) to a col index
        # Note: U and V depend on the trial, so we can't precompute vals.
        # But we can precompute the row/col indices for each (i, j, jp, k, kp, ip, c) tuple?
        # No, the coefficient is U[i*m+j, c] * V[jp*p+k, c].
        # We can precompute the list of (row, col, u_idx, v_idx) for each eq and c.
        
        eq_data = []
        row_idx = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                rhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                b_vec.append(rhs)
                                # For each c, we have a term
                                for c in range(r):
                                    # col index for W[kp*n+ip, c]
                                    col_idx = (kp * n + ip) * r + c
                                    # u_idx, v_idx for coefficient
                                    u_idx = i * m + j
                                    v_idx = jp * p + k
                                    eq_data.append((row_idx, col_idx, u_idx, v_idx))
                                row_idx += 1
                                
        num_eq = row_idx
        b_arr = np.array(b_vec)
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return False
            
            # Generate random U and V with small integers
            U = np.random.randint(-2, 3, size=(n*m, r))
            V = np.random.randint(-2, 3, size=(m*p, r))
            
            # Build sparse matrix A
            # A[row, col] = sum of U[u_idx, c] * V[v_idx, c] for all eq_data with same row, col
            # Since each (row, col) in eq_data corresponds to a unique c (because col encodes c),
            # and row encodes the eq, each entry in A is just U[u_idx, c] * V[v_idx, c].
            
            # We can build A efficiently
            A_rows = [d[0] for d in eq_data]
            A_cols = [d[1] for d in eq_data]
            # Coefficients
            A_vals = []
            for d in eq_data:
                _, _, u_idx, v_idx = d
                # We need c. c is not explicitly in d, but col = (kp*n+ip)*r + c.
                # We can extract c from col? Or just keep it in d.
                # Let's re-extract c from col_idx
                c = d[1] % r
                coeff = U[u_idx, c] * V[v_idx, c]
                A_vals.append(coeff)
                
            A_mat = sp.csr_matrix((A_vals, (A_rows, A_cols)), shape=(num_eq, num_unknowns))
            
            # Solve least squares
            try:
                x = spla.lsqr(A_mat, b_arr, atol=1e-10, btol=1e-10, itermax=10000)[0]
                residual = A_mat @ x - b_arr
                max_res = np.max(np.abs(residual))
                
                if max_res < 1e-6:
                    W_flat = x
                    W_mat = W_flat.reshape(p * n, r)
                    
                    # Check if W is close to integer
                    W_rounded = np.round(W_mat)
                    if np.max(np.abs(W_mat - W_rounded)) < 1e-4:
                        W_int = W_rounded.astype(int)
                        U_int = U.astype(int)
                        V_int = V.astype(int)
                        
                        # Verify exactly
                        if verify(U_int, V_int, W_int):
                            best_rank = r
                            best_U = U_int
                            best_V = V_int
                            best_W = W_int
                            return True
            except Exception:
                pass
        
        return False

    # Try ranks from 70 upward
    # We can start lower, but 70 is a reasonable guess for 3x4x7
    for r in range(70, best_rank + 1):
        if time.time() > deadline:
            break
        if try_rank(r, max_trials=100):
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
