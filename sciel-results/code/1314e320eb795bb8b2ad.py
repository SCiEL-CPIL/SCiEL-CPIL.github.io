import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5s margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    deadline = time.time() + TIME_LIMIT - 5  # safety margin
    
    U_rows = n * m  # 16
    V_rows = m * p  # 24
    W_rows = p * n  # 24
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(np.array(u_vec, dtype=np.float64))
        V_cols.append(np.array(v_vec, dtype=np.float64))
        W_cols.append(np.array(w_vec, dtype=np.float64))
    
    def strassen_2x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 7 rank-1 terms for Strassen 2x2*2x2."""
        def ai(i,j): return (a_r+i)*m + (a_c+j)
        def bi(i,j): return (b_r+i)*p + (b_c+j)
        def ci(i,j): return (c_c+j)*n + (c_r+i)
        
        def term(a_entries, b_entries, c_entries):
            u = [0.0]*U_rows; v = [0.0]*V_rows; w = [0.0]*W_rows
            for idx, val in a_entries:
                u[idx] = val
            for idx, val in b_entries:
                v[idx] = val
            for idx, val in c_entries:
                w[idx] = val
            add_term(u, v, w)
        
        # M1 = (A11+A22)*(B11+B22)
        term([(ai(0,0),1.0),(ai(1,1),1.0)], [(bi(0,0),1.0),(bi(1,1),1.0)], [(ci(0,0),1.0),(ci(1,1),1.0)])
        # M2 = (A21+A22)*B11
        term([(ai(1,0),1.0),(ai(1,1),1.0)], [(bi(0,0),1.0)], [(ci(1,0),1.0),(ci(1,1),-1.0)])
        # M3 = A11*(B12-B22)
        term([(ai(0,0),1.0)], [(bi(0,1),1.0),(bi(1,1),-1.0)], [(ci(0,1),1.0),(ci(1,1),1.0)])
        # M4 = A22*(B21-B11)
        term([(ai(1,1),1.0)], [(bi(1,0),1.0),(bi(0,0),-1.0)], [(ci(0,0),1.0),(ci(1,0),1.0)])
        # M5 = (A11+A12)*B22
        term([(ai(0,0),1.0),(ai(0,1),1.0)], [(bi(1,1),1.0)], [(ci(0,0),-1.0),(ci(0,1),1.0)])
        # M6 = (A21-A11)*(B11+B12)
        term([(ai(1,0),1.0),(ai(0,0),-1.0)], [(bi(0,0),1.0),(bi(0,1),1.0)], [(ci(1,1),1.0)])
        # M7 = (A12-A22)*(B21+B22)
        term([(ai(0,1),1.0),(ai(1,1),-1.0)], [(bi(1,0),1.0),(bi(1,1),1.0)], [(ci(0,0),1.0)])
    
    # Build the decomposition by tiling
    for block in range(3):
        b_col_offset = block * 2
        c_col_offset = block * 2
        
        strassen_2x2_2x2(0, 0, 0, b_col_offset, 0, c_col_offset)
        strassen_2x2_2x2(0, 2, 2, b_col_offset, 0, c_col_offset)
        strassen_2x2_2x2(2, 0, 0, b_col_offset, 2, c_col_offset)
        strassen_2x2_2x2(2, 2, 2, b_col_offset, 2, c_col_offset)
    
    rank = len(U_cols)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    # Build target tensor
    T = np.zeros((U_rows, V_rows, W_rows), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def check(U, V, W, atol=1e-9):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)
    
    def err_of(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T - recon) ** 2)
    
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = rank
    best_err = err_of(best_U, best_V, best_W)
    
    # Try to find linear dependencies by checking if removing a column
    # and adjusting others can still reconstruct the tensor
    # Use a greedy approach: for each column, try to express it as a
    # linear combination of others
    
    for trial in range(100):
        if time.time() > deadline:
            break
        
        improved = False
        for r in range(best_rank):
            if time.time() > deadline:
                break
            
            mask = [i for i in range(best_rank) if i != r]
            if len(mask) < 3:
                break
            
            # Try to find c such that the tensor contribution of col r
            # is approximated by sum of others
            # We need: outer(U[:,r], V[:,r], W[:,r]) ≈ sum_s c_s * outer(U[:,s], V[:,s], W[:,s])
            # This is equivalent to: for all (a,b,c), U[a,r]*V[b,r]*W[c,r] = sum_s c_s * U[a,s]*V[b,s]*W[c,s]
            # We can solve this by flattening and using least squares
            
            # Create design matrix: each row corresponds to (a,b,c) triple
            # But that's 16*24*24 = 9216 rows, which is fine
            
            # Actually, let's use a simpler approach: just try removing and check
            U_trial = best_U[:, mask]
            V_trial = best_V[:, mask]
            W_trial = best_W[:, mask]
            
            err_trial = err_of(U_trial, V_trial, W_trial)
            
            if err_trial < 1e-12:
                best_U = U_trial
                best_V = V_trial
                best_W = W_trial
                best_rank -= 1
                best_err = err_trial
                improved = True
                break
        
        if not improved:
            break
    
    # Try a more sophisticated approach: use the fact that Strassen terms
    # across different blocks may have linear dependencies
    # For each pair of blocks, check if we can merge terms
    
    # Actually, let's try a different approach: use randomized search to find
    # better decompositions
    
    rng = np.random.RandomState(42)
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Start from current best, try removing random subsets
        U_cur = best_U.copy()
        V_cur = best_V.copy()
        W_cur = best_W.copy()
        rank_cur = best_rank
        err_cur = best_err
        
        # Try removing one random column at a time
        for step in range(rank_cur):
            if time.time() > deadline:
                break
            if rank_cur <= 1:
                break
            
            r = rng.randint(rank_cur)
            mask = [i for i in range(rank_cur) if i != r]
            
            U_trial = U_cur[:, mask]
            V_trial = V_cur[:, mask]
            W_trial = W_cur[:, mask]
            
            err_trial = err_of(U_trial, V_trial, W_trial)
            
            if err_trial < 1e-12:
                U_cur = U_trial
                V_cur = V_trial
                W_cur = W_trial
                rank_cur -= 1
                err_trial = 0.0
        
        if rank_cur < best_rank:
            best_U = U_cur
            best_V = V_cur
            best_W = W_cur
            best_rank = rank_cur
            best_err = err_cur
    
    # Verify correctness
    if check(best_U, best_V, best_W):
        def to_list(mat):
            return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    
    # Fallback to original
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
