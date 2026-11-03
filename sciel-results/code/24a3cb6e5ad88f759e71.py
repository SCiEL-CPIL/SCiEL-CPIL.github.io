import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 6]
DEADLINE_SECONDS = 130  # Leave 20s margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    def time_left():
        return DEADLINE_SECONDS - (time.time() - start_time)

    def check_exact_fast(U, V, W, tol=1e-9):
        """Vectorized exact check using einsum."""
        R = U.shape[1]
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return np.all(np.abs(T - 1.0) <= tol)

    def round_to_half(U, V, W):
        """Round to nearest half-integer."""
        return np.round(U * 2) / 2.0, np.round(V * 2) / 2.0, np.round(W * 2) / 2.0

    def try_round_and_check(U, V, W, tol=1e-9):
        """Round to half-integers and check."""
        Ur, Vr, Wr = round_to_half(U, V, W)
        if check_exact_fast(Ur, Vr, Wr, tol):
            return Ur, Vr, Wr
        return None

    def schoolbook():
        """Return the schoolbook decomposition (rank 72)."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def strassen_2x2x2():
        """
        Strassen decomposition for 2x2x2 matrix multiplication.
        Returns U (4x7), V (4x7), W (4x7) where:
        U index: a*2 + d (a in 0..1, d in 0..1)
        V index: d*2 + c (d in 0..1, c in 0..1)
        W index: c*2 + a (c in 0..1, a in 0..1)
        """
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U_s[0*2+0, 0] = 1; U_s[1*2+1, 0] = 1
        V_s[0*2+0, 0] = 1; V_s[1*2+1, 0] = 1
        W_s[0*2+0, 0] = 1; W_s[1*2+1, 0] = 1
        
        # M2 = (A10+A11)B00
        U_s[1*2+0, 1] = 1; U_s[1*2+1, 1] = 1
        V_s[0*2+0, 1] = 1
        W_s[1*2+0, 1] = 1
        
        # M3 = A00(B01-B11)
        U_s[0*2+0, 2] = 1
        V_s[0*2+1, 2] = 1; V_s[1*2+1, 2] = -1
        W_s[0*2+1, 2] = 1
        
        # M4 = A11(B10-B00)
        U_s[1*2+1, 3] = 1
        V_s[1*2+0, 3] = 1; V_s[0*2+0, 3] = -1
        W_s[0*2+0, 3] = 1
        W_s[1*2+0, 3] = 1
        
        # M5 = (A00+A10)B11
        U_s[0*2+0, 4] = 1; U_s[1*2+0, 4] = 1
        V_s[1*2+1, 4] = 1
        W_s[0*2+0, 4] = -1
        W_s[1*2+1, 4] = 1
        
        # M6 = (A10-A00)(B00+B10)
        U_s[1*2+0, 5] = 1; U_s[0*2+0, 5] = -1
        V_s[0*2+0, 5] = 1; V_s[1*2+0, 5] = 1
        W_s[1*2+1, 5] = 1
        
        # M7 = (A01-A11)(B01+B11)
        U_s[0*2+1, 6] = 1; U_s[1*2+1, 6] = -1
        V_s[0*2+1, 6] = 1; V_s[1*2+1, 6] = 1
        W_s[0*2+0, 6] = 1
        
        return U_s, V_s, W_s

    def schoolbook_1x2x2():
        """
        Schoolbook decomposition for 1x2x2 matrix multiplication.
        A is 1x2, B is 2x2, C is 1x2.
        U: (1*2, 4) = (2, 4), row index = d (0,1)
        V: (2*2, 4) = (4, 4), row index = d*2 + c
        W: (2*1, 4) = (2, 4), row index = c (0,1)
        """
        U_s = np.zeros((2, 4))
        V_s = np.zeros((4, 4))
        W_s = np.zeros((2, 4))
        
        col = 0
        for d in range(2):
            for c in range(2):
                U_s[d, col] = 1
                V_s[d*2 + c, col] = 1
                W_s[c, col] = 1
                col += 1
        
        return U_s, V_s, W_s

    def build_rank_66():
        """
        Builds a rank 66 decomposition.
        Splits A rows [2, 1], B cols [2, 2, 2].
        For each (A_block, B_block):
          If A_block size 2:
            Split mid [2, 2].
            2x2x2 (Strassen) + 2x2x2 (Strassen) = 14.
          If A_block size 1:
            Split mid [2, 2].
            1x2x2 (Schoolbook) + 1x2x2 (Schoolbook) = 8.
        Total: 3 * 14 + 3 * 8 = 66.
        """
        U_list = []
        V_list = []
        W_list = []
        
        U_s, V_s, W_s = strassen_2x2x2()
        U_1, V_1, W_1 = schoolbook_1x2x2()
        
        # A row blocks: [0, 1] (size 2), [2] (size 1)
        # B col blocks: [0, 1], [2, 3], [4, 5]
        
        a_blocks = [
            {"rows": [0, 1], "size": 2},
            {"rows": [2], "size": 1}
        ]
        
        b_col_blocks = [
            {"cols": [0, 1]},
            {"cols": [2, 3]},
            {"cols": [4, 5]}
        ]
        
        for a_block in a_blocks:
            a_rows = a_block["rows"]
            a_size = a_block["size"]
            
            for b_col_block in b_col_blocks:
                b_cols = b_col_block["cols"]
                
                # Mid dimension (size 4) split into [0, 1] and [2, 3]
                mid_blocks = [
                    {"rows": [0, 1]},
                    {"rows": [2, 3]}
                ]
                
                for mid_block in mid_blocks:
                    b_rows = mid_block["rows"]
                    
                    U_full = np.zeros((n*m, 7 if a_size == 2 else 4))
                    V_full = np.zeros((m*p, 7 if a_size == 2 else 4))
                    W_full = np.zeros((p*n, 7 if a_size == 2 else 4))
                    
                    if a_size == 2:
                        # 2x2x2 Strassen
                        for a_idx, a_row in enumerate(a_rows):
                            for d_idx, b_row in enumerate(b_rows):
                                for c_idx, b_col in enumerate(b_cols):
                                    U_full[a_row * m + b_row, :] += U_s[a_idx * 2 + d_idx, :]
                                    V_full[b_row * p + b_col, :] += V_s[d_idx * 2 + c_idx, :]
                                    W_full[b_col * n + a_row, :] += W_s[c_idx * 2 + a_idx, :]
                    else:
                        # 1x2x2 Schoolbook
                        a_row = a_rows[0]
                        for d_idx, b_row in enumerate(b_rows):
                            for c_idx, b_col in enumerate(b_cols):
                                U_full[a_row * m + b_row, :] += U_1[d_idx, :]
                                V_full[b_row * p + b_col, :] += V_1[d_idx * 2 + c_idx, :]
                                W_full[b_col * n + a_row, :] += W_1[c_idx, :]
                    
                    U_list.append(U_full)
                    V_list.append(V_full)
                    W_list.append(W_full)
        
        U = np.hstack(U_list)
        V = np.hstack(V_list)
        W = np.hstack(W_list)
        
        return U, V, W

    def nullspace_reduce(U, V, W, tol=1e-10):
        """
        Try to reduce rank by finding linear dependencies among the rank-1 components.
        If the tensor flattening matrix M (shape n*m*p x R) has rank < R,
        we can find a nullspace vector c and eliminate one component.
        """
        R = U.shape[1]
        if R <= 1:
            return U, V, W, R
        
        # Build the flattening matrix M where M[:, r] is the vectorized tensor component
        # M[i*m*p + j*p + k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        
        # M shape: (n*m*p, R)
        M = np.einsum('ijr,jkr,kir->ijk r', U_t, V_t, W_t)
        M = M.reshape(n*m*p, R)
        
        # Find nullspace of M
        try:
            from scipy.linalg import null_space
            N = null_space(M, tol=tol)
        except Exception:
            # Fallback: use SVD
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                # Nullspace vectors are columns of Vt corresponding to zero singular values
                rank = np.sum(s > tol * max(s)) if len(s) > 0 else 0
                if rank < R:
                    N = Vt[rank:].T
                else:
                    return U, V, W, R
            except Exception:
                return U, V, W, R
        
        if N.shape[1] == 0:
            return U, V, W, R
        
        # Try to eliminate components using nullspace vectors
        # For each nullspace vector c, we have sum_r c_r * T_r = 0
        # This means component r can be expressed in terms of others
        # We'll greedily try to eliminate the component with largest |c_r|
        
        best_U, best_V, best_W, best_rank = U, V, W, R
        
        for c in N.T:
            # Find the index with largest absolute value
            idx = np.argmax(np.abs(c))
            if abs(c[idx]) < tol:
                continue
            
            # Try to eliminate component idx
            # New rank: R-1
            # We need to rescale: for each r != idx, adjust by c_r / c_idx
            # Actually, the nullspace relation is: sum_r c_r * T_r = 0
            # So T_idx = -sum_{r!=idx} (c_r/c_idx) * T_r
            # We can remove T_idx and add its contribution to the others
            
            new_R = R - 1
            keep = [r for r in range(R) if r != idx]
            
            U_new = U[:, keep].copy()
            V_new = V[:, keep].copy()
            W_new = W[:, keep].copy()
            
            # Add the contribution of component idx to the kept components
            for pos, r in enumerate(keep):
                factor = -c[r] / c[idx]
                U_new[:, pos] += factor * U[:, idx]
                V_new[:, pos] += factor * V[:, idx]
                W_new[:, pos] += factor * W[:, idx]
            
            # Check if this is valid
            if check_exact_fast(U_new, V_new, W_new, tol=1e-9):
                # Try to round to half-integers
                Ur, Vr, Wr = round_to_half(U_new, V_new, W_new)
                if check_exact_fast(Ur, Vr, Wr, tol=1e-9):
                    if new_R < best_rank:
                        best_U, best_V, best_W, best_rank = Ur, Vr, Wr, new_R
                else:
                    if new_R < best_rank:
                        best_U, best_V, best_W, best_rank = U_new, V_new, W_new, new_R
        
        return best_U, best_V, best_W, best_rank

    def april_refine(U, V, W, max_iter=2000, lr_init=0.05, tol=1e-10):
        """Run APRIL refinement until convergence or max_iter."""
        R = U.shape[1]
        U_t = U.reshape(n, m, R).copy()
        V_t = V.reshape(m, p, R).copy()
        W_t = W.reshape(p, n, R).copy()
        
        lr = lr_init
        best_loss = np.inf
        best_state = (U_t.copy(), V_t.copy(), W_t.copy())
        
        for it in range(max_iter):
            if time_left() < 20:
                break
            
            # Compute T
            T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            loss = np.sum((T - 1.0)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_state = (U_t.copy(), V_t.copy(), W_t.copy())
            
            if loss < tol:
                break
            
            # Adaptive learning rate
            if it % 100 == 0 and it > 0:
                lr *= 0.95
            
            # APRIL step
            dU = np.einsum('jkr,kir,ijk->ijr', V_t, W_t, T - 1.0)
            U_t -= lr * dU
            
            T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            
            dV = np.einsum('ijr,kir,ijk->jkr', U_t, W_t, T - 1.0)
            V_t -= lr * dV
            
            T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            
            dW = np.einsum('ijr,jkr,ijk->kir', U_t, V_t, T - 1.0)
            W_t -= lr * dW
        
        U_best, V_best, W_best = best_state
        return U_best.reshape(n*m, R), V_best.reshape(m*p, R), W_best.reshape(p*n, R)

    # 1. Try the deterministic block decomposition (Rank 66), then try nullspace reduction
    best_result = None
    best_rank = n * m * p  # 72
    
    if time_left() > 10:
        U, V, W = build_rank_66()
        if check_exact_fast(U, V, W, tol=1e-9):
            # Try nullspace reduction
            U, V, W, R = nullspace_reduce(U, V, W)
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
                print(f"Block decomposition reduced to rank {best_rank} via nullspace")
            else:
                best_rank = U.shape[1]
                best_result = (U, V, W)
                print(f"Block decomposition found with rank {best_rank}")
    
    # 2. Try to improve further with random restarts + APRIL
    # Ranks to try: 64, 63, 62...
    # We only try if we have time and if the block decomposition worked (or even if it didn't, but we want lower)
    
    target_ranks = [64, 63, 62, 60, 58]
    
    np.random.seed(42)
    
    for R in target_ranks:
        if time_left() < 30:
            break
        if R >= best_rank:
            continue
            
        for restart in range(2):
            if time_left() < 30:
                break
            
            # Random initialization
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            # Run APRIL refinement
            U, V, W = april_refine(U, V, W, max_iter=500, lr_init=0.05)
            
            # Try rounding to half-integers
            result = try_round_and_check(U, V, W, tol=1e-9)
            if result is not None:
                Ur, Vr, Wr = result
                if check_exact_fast(Ur, Vr, Wr, tol=1e-9):
                    if R < best_rank:
                        best_rank = R
                        best_result = (Ur, Vr, Wr)
                        print(f"Found rank {R} decomposition via search!")
                        break
        
        if best_result is not None and best_rank <= R:
            break

    # Fallback to schoolbook if nothing found (should not happen if block works)
    if best_result is None:
        U, V, W = schoolbook()
        best_result = (U, V, W)
        best_rank = U.shape[1]

    Ur, Vr, Wr = best_result
    return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    print(f"Rank: {U.shape[1]}")
    print(f"Valid: {check_exact_fast(U, V, W)}")
