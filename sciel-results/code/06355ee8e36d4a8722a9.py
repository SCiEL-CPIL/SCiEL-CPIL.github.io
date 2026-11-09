import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    
    # Generate schoolbook decomposition as fallback
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = get_schoolbook()
    
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    # Verify schoolbook decomposition
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if not np.isclose(s, 1.0):
                        return False
        return True

    if not verify(U_sb, V_sb, W_sb):
        # Should not happen, but fallback
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    # Attempt to find rank 44 decomposition
    # Strategy: Try to find U, V, W of rank 44 such that the tensor equation holds.
    # We'll use a heuristic: fix W to be mostly schoolbook-like and solve for U, V.
    
    # For rank 44, we have 44 terms. The tensor equation is:
    # sum_{c=1}^{44} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    
    # This is a system of 45 equations (one per (i,j,k) triple) with 44*9 + 44*15 + 44*15 = 396 + 660 + 660 = 1716 unknowns.
    # But it's nonlinear (products of variables from different matrices).
    
    # Alternative: fix two matrices and solve for the third.
    # If we fix U and V, then for each (k,i), we need:
    # sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c]) = 1 for all j? No, that's not right.
    
    # Actually, the equation is:
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # If we fix U and V, define A_{(i,j,k), c} = U[i*m+j, c] * V[j*p+k, c]
    # Then we need: sum_c A_{(i,j,k), c} * W[k*n+i, c] = 1
    
    # This is linear in W! For each (k,i), we have one equation per j (since j varies).
    # Wait, no: for fixed (k,i), j ranges over 0..m-1, so we have m equations for the same W row.
    # But A depends on j, so it's: for each j, sum_c A_{(i,j,k), c} * W[k*n+i, c] = 1
    
    # So for each (k,i), we have a system of m equations in 44 unknowns (W[k*n+i, :]).
    # Total: p*n = 15 such systems, each with 3 equations and 44 unknowns.
    # This is underdetermined, so we can find solutions if consistent.
    
    # Let's try: fix U and V to be schoolbook-like (but with 44 columns), and solve for W.
    
    # Construct U and V with 44 columns. Start with schoolbook 45 columns, remove one column,
    # and see if we can adjust.
    
    # Try removing the last column (i=2, j=2, k=4) from schoolbook.
    # Then U has 44 columns, V has 44 columns, and we solve for W with 44 columns.
    
    U_44 = [row[:44] for row in U_sb]
    V_44 = [row[:44] for row in V_sb]
    
    # Now solve for W_44 (15 rows, 44 columns)
    # For each (k,i), we have a system:
    # For j in range(m): sum_{c=0}^{43} U_44[i*m+j][c] * V_44[j*p+k][c] * W_44[k*n+i][c] = 1
    
    W_44 = [[0] * 44 for _ in range(p * n)]
    
    # Build the linear system for each (k,i)
    # Variables: W_44[k*n+i][0], ..., W_44[k*n+i][43]
    # Equations: for each j in 0..m-1:
    #   sum_c (U_44[i*m+j][c] * V_44[j*p+k][c]) * W_44[k*n+i][c] = 1
    
    # This is a linear system A * w = b, where A is m x 44, w is 44x1, b is m x 1.
    # Since m=3 < 44, the system is underdetermined. We can find a solution using least squares or by setting free variables to 0.
    
    # Use numpy to solve each system
    import numpy as np
    
    for k in range(p):
        for i in range(n):
            # Build A (m x 44) and b (m x 1)
            A = np.zeros((m, 44))
            b = np.ones(m)
            for j in range(m):
                for c in range(44):
                    A[j, c] = U_44[i*m+j][c] * V_44[j*p+k][c]
            
            # Solve A * w = b using least squares (minimum norm solution)
            try:
                w, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                W_44[k*n+i] = w.tolist()
            except:
                W_44[k*n+i] = [0] * 44
    
    # Verify the rank-44 candidate
    if verify(U_44, V_44, W_44):
        # Check if entries are integer or half-integer
        def is_int_or_half(x):
            return np.isclose(x * 2, round(x * 2))
        
        valid_entries = True
        for mat in [U_44, V_44, W_44]:
            for row in mat:
                for val in row:
                    if not is_int_or_half(val):
                        valid_entries = False
                        break
                if not valid_entries:
                    break
            if not valid_entries:
                break
        
        if valid_entries:
            return {"U": U_44, "V": V_44, "W": W_44}
    
    # If rank 44 didn't work or entries weren't clean, try a few more removals
    for removed_col in range(45):
        if time.time() - start_time > timeout:
            break
        
        U_try = [row[:removed_col] + row[removed_col+1:] for row in U_sb]
        V_try = [row[:removed_col] + row[removed_col+1:] for row in V_sb]
        
        W_try = [[0] * 44 for _ in range(p * n)]
        
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, 44))
                b = np.ones(m)
                for j in range(m):
                    for c in range(44):
                        A[j, c] = U_try[i*m+j][c] * V_try[j*p+k][c]
                
                try:
                    w, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_try[k*n+i] = w.tolist()
                except:
                    W_try[k*n+i] = [0] * 44
        
        if verify(U_try, V_try, W_try):
            def is_int_or_half(x):
                return np.isclose(x * 2, round(x * 2))
            
            valid_entries = True
            for mat in [U_try, V_try, W_try]:
                for row in mat:
                    for val in row:
                        if not is_int_or_half(val):
                            valid_entries = False
                            break
                    if not valid_entries:
                        break
                if not valid_entries:
                    break
            
            if valid_entries:
                return {"U": U_try, "V": V_try, "W": W_try}
    
    # Fallback: return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
