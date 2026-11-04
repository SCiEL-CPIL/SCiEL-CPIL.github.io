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
    
    # Build the rank-84 decomposition
    for block in range(3):
        b_col_offset = block * 2
        c_col_offset = block * 2
        
        strassen_2x2_2x2(0, 0, 0, b_col_offset, 0, c_col_offset)
        strassen_2x2_2x2(0, 2, 2, b_col_offset, 0, c_col_offset)
        strassen_2x2_2x2(2, 0, 0, b_col_offset, 2, c_col_offset)
        strassen_2x2_2x2(2, 2, 2, b_col_offset, 2, c_col_offset)
    
    rank = len(U_cols)
    
    if time.time() > deadline:
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        def to_list(mat):
            return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
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
    
    best_U = np.column_stack(U_cols)
    best_V = np.column_stack(V_cols)
    best_W = np.column_stack(W_cols)
    best_rank = rank
    best_err = err_of(best_U, best_V, best_W)
    
    # Structural approach: coordinate descent with column elimination
    # For each column r, try to find a linear combination of other columns
    # that can replace it, then remove it and adjust the remaining columns.
    
    # Key insight: if we have a decomposition with rank R, and column r's
    # contribution can be expressed as a combination of other columns' contributions
    # in the tensor sense, we can eliminate it.
    
    # More practical: try all pairs of columns to see if their difference
    # or some combination can be absorbed.
    
    # First, try simple elimination: for each column, check if removing it
    # and adjusting others via least squares reduces error to zero.
    
    def try_eliminate_column(U, V, W, r, T, atol=1e-10):
        """Try to eliminate column r by adjusting other columns."""
        R = U.shape[1]
        if R <= 1:
            return None, None, None, False
        
        mask = [i for i in range(R) if i != r]
        U_others = U[:, mask].copy()
        V_others = V[:, mask].copy()
        W_others = W[:, mask].copy()
        
        # The column r contributes: outer(U[:,r], V[:,r], W[:,r])
        # We want to find adjustments delta_U, delta_V, delta_W to other columns
        # such that the total tensor matches.
        
        # This is a bilinear problem. Instead, use a simpler approach:
        # Check if the tensor contribution of column r is already captured
        # by the other columns (i.e., removing it gives zero error).
        
        U_trial = U[:, mask]
        V_trial = V[:, mask]
        W_trial = W[:, mask]
        
        recon = np.einsum('ar,br,cr->abc', U_trial, V_trial, W_trial)
        err = np.sum((T - recon) ** 2)
        
        if err < atol:
            return U_trial, V_trial, W_trial, True
        
        # Try a more sophisticated approach: use the fact that for matrix multiplication
        # tensors, there are known symmetries. Try swapping or combining columns.
        # For now, just return failure.
        return None, None, None, False
    
    # Iterative elimination pass
    for trial in range(100):
        if time.time() > deadline:
            break
        
        improved = False
        for r in range(best_rank):
            if time.time() > deadline:
                break
            
            U_new, V_new, W_new, success = try_eliminate_column(
                best_U, best_V, best_W, r, T
            )
            
            if success:
                best_U = U_new
                best_V = V_new
                best_W = W_new
                best_rank -= 1
                improved = True
                break
        
        if not improved:
            break
    
    # Now try a different structural approach: exploit the rectangular nature.
    # For 4x4 * 4x6, we can think of it as 4x4 * 4x2 done 3 times.
    # Or better: use a mixed strategy where some blocks use Strassen and others use schoolbook.
    
    # Actually, let's try a completely different decomposition strategy:
    # Use the fact that 4 = 2+2 and 6 = 2+2+2, but with cross-block terms.
    
    # Alternative: try to find a decomposition with rank < 84 by using
    # the known optimal rank for 4x4*4x4 (which is 49) and extending to 4x6.
    
    # For 4x4*4x4, the best known is rank 49 (Larsen 2008).
    # For 4x4*4x6, we can potentially do better than 84 by using
    # a more clever block structure.
    
    # Let's try: split B into three 4x2 blocks. Use a 4x4*4x2 algorithm
    # (rank 28?) three times, but share some terms.
    
    # Actually, the current approach already does this. Let's try to
    # find linear dependencies more aggressively.
    
    # Use SVD-like approach: for each pair of columns, check if one can
    # be expressed in terms of the other plus small adjustments.
    
    rng = np.random.RandomState(42)
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Randomly perturb the columns slightly and try elimination
        U_pert = best_U.copy()
        V_pert = best_V.copy()
        W_pert = best_W.copy()
        
        # Add small random noise to encourage different elimination paths
        for r in range(best_rank):
            if rng.random() < 0.3:
                U_pert[:, r] += 0.001 * rng.randn(U_rows)
                V_pert[:, r] += 0.001 * rng.randn(V_rows)
                W_pert[:, r] += 0.001 * rng.randn(W_rows)
        
        # Try elimination on the perturbed version
        improved = False
        for r in range(best_rank):
            if time.time() > deadline:
                break
            
            mask = [i for i in range(best_rank) if i != r]
            U_trial = U_pert[:, mask]
            V_trial = V_pert[:, mask]
            W_trial = W_pert[:, mask]
            
            recon = np.einsum('ar,br,cr->abc', U_trial, V_trial, W_trial)
            err = np.sum((T - recon) ** 2)
            
            if err < 1e-10:
                # Found an elimination
                best_U = U_trial
                best_V = V_trial
                best_W = W_trial
                best_rank -= 1
                improved = True
                break
        
        if not improved:
            # Try a different approach: merge two columns
            r1 = rng.randint(best_rank)
            r2 = rng.randint(best_rank)
            if r1 == r2:
                continue
            
            # Try to replace columns r1 and r2 with a single column
            # such that the tensor contribution is preserved.
            # This is generally not possible, but try anyway.
            mask = [i for i in range(best_rank) if i != r1 and i != r2]
            U_trial = np.column_stack([best_U[:, mask], np.zeros(U_rows)])
            V_trial = np.column_stack([best_V[:, mask], np.zeros(V_rows)])
            W_trial = np.column_stack([best_W[:, mask], np.zeros(W_rows)])
            
            # The new column should satisfy:
            # outer(u_new, v_new, w_new) = outer(u_r1, v_r1, w_r1) + outer(u_r2, v_r2, w_r2)
            # This is generally not rank-1, so this approach fails.
            # Skip this.
            pass
    
    # Final verification
    if check(best_U, best_V, best_W):
        def to_list(mat):
            return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    
    # Fallback to original
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    return {"U": to_list(np.column_stack(U_cols)), "V": to_list(np.column_stack(V_cols)), "W": to_list(np.column_stack(W_cols))}
