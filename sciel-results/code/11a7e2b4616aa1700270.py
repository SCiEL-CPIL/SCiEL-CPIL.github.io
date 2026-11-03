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

    def build_block_decomposition():
        """
        Build a block-structured decomposition of rank 66.
        A (3x4), B (4x6).
        Split A rows into [2, 1].
        Split B cols into [2, 2, 2].
        Split B rows (middle dim) into [2, 2].
        
        For each combination of A-row-block (2 or 1) and B-col-block (2):
          If A-row-block is 2: use Strassen 2x2x2 (rank 7) for each B-row-subblock (2).
          If A-row-block is 1: use Schoolbook 1x2x2 (rank 4) for each B-row-subblock (2).
        
        Total rank = 3 (col blocks) * (2 (mid blocks) * 7 (Strassen for A[0:2]) + 2 (mid blocks) * 4 (Schoolbook for A[2:3]))
                   = 3 * (14 + 8) = 3 * 22 = 66.
        """
        U = np.zeros((n*m, 66))
        V = np.zeros((m*p, 66))
        W = np.zeros((p*n, 66))
        
        U_s, V_s, W_s = strassen_2x2x2()
        U_1, V_1, W_1 = schoolbook_1x2x2()
        
        col_offset = 0
        
        for c_block in range(3):
            # B cols: [c_block*2, c_block*2+1]
            b_cols = [c_block*2, c_block*2+1]
            
            # A[0:2] part (Strassen)
            for mid_block in range(2):
                # B rows (middle): [mid_block*2, mid_block*2+1]
                b_rows = [mid_block*2, mid_block*2+1]
                # A rows: [0, 1]
                a_rows = [0, 1]
                # A cols: [mid_block*2, mid_block*2+1]
                a_cols = [mid_block*2, mid_block*2+1]
                
                # Map local Strassen indices to global
                # Local U index: a*2 + d. Global U row: a_rows[a]*m + a_cols[d]
                # Local V index: d*2 + c. Global V row: b_rows[d]*p + b_cols[c]
                # Local W index: c*2 + a. Global W row: b_cols[c]*n + a_rows[a]
                
                for a in range(2):
                    for d in range(2):
                        u_row = a_rows[a]*m + a_cols[d]
                        U[u_row, col_offset:col_offset+7] += U_s[a*2+d, :]
                        
                for d in range(2):
                    for c in range(2):
                        v_row = b_rows[d]*p + b_cols[c]
                        V[v_row, col_offset:col_offset+7] += V_s[d*2+c, :]
                        
                for c in range(2):
                    for a in range(2):
                        w_row = b_cols[c]*n + a_rows[a]
                        W[w_row, col_offset:col_offset+7] += W_s[c*2+a, :]
                        
                col_offset += 7
            
            # A[2:3] part (Schoolbook)
            for mid_block in range(2):
                # B rows (middle): [mid_block*2, mid_block*2+1]
                b_rows = [mid_block*2, mid_block*2+1]
                # A row: [2]
                a_row = 2
                # A cols: [mid_block*2, mid_block*2+1]
                a_cols = [mid_block*2, mid_block*2+1]
                
                # Map local Schoolbook indices to global
                # Local U index: d. Global U row: a_row*m + a_cols[d]
                # Local V index: d*2 + c. Global V row: b_rows[d]*p + b_cols[c]
                # Local W index: c. Global W row: b_cols[c]*n + a_row
                
                for d in range(2):
                    u_row = a_row*m + a_cols[d]
                    U[u_row, col_offset:col_offset+4] += U_1[d, :]
                    
                for d in range(2):
                    for c in range(2):
                        v_row = b_rows[d]*p + b_cols[c]
                        V[v_row, col_offset:col_offset+4] += V_1[d*2+c, :]
                        
                for c in range(2):
                    w_row = b_cols[c]*n + a_row
                    W[w_row, col_offset:col_offset+4] += W_1[c, :]
                    
                col_offset += 4
                
        return U, V, W

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

    # 1. Try the deterministic block decomposition (Rank 66)
    best_result = None
    best_rank = n * m * p  # 72
    
    if time_left() > 10:
        U, V, W = build_block_decomposition()
        if check_exact_fast(U, V, W, tol=1e-9):
            best_rank = U.shape[1]
            best_result = (U, V, W)
            print(f"Block decomposition found with rank {best_rank}")
    
    # Fallback to schoolbook if nothing found
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
