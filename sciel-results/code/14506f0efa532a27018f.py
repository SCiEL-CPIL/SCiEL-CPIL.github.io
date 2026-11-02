"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    # Default rank-11 solution
    default_rank = 11
    U_def = [[0] * default_rank for _ in range(n * m)]
    V_def = [[0] * default_rank for _ in range(m * p)]
    W_def = [[0] * default_rank for _ in range(p * n)]
    
    U_def[0][0] = 1; U_def[3][0] = 1
    V_def[0][0] = 1; V_def[4][0] = 1
    U_def[2][1] = 1; U_def[3][1] = 1
    V_def[0][1] = 1
    U_def[0][2] = 1
    V_def[1][2] = 1; V_def[4][2] = -1
    U_def[3][3] = 1
    V_def[3][3] = 1; V_def[0][3] = -1
    U_def[0][4] = 1; U_def[1][4] = 1
    V_def[4][4] = 1
    U_def[2][5] = 1; U_def[0][5] = -1
    V_def[0][5] = 1; V_def[1][5] = 1
    U_def[1][6] = 1; U_def[3][6] = -1
    V_def[3][6] = 1; V_def[4][6] = 1
    W_def[0][0] = 1; W_def[0][3] = 1; W_def[0][4] = -1; W_def[0][6] = 1
    W_def[2][2] = 1; W_def[2][4] = 1
    W_def[1][1] = 1; W_def[1][3] = 1
    W_def[3][0] = 1; W_def[3][1] = -1; W_def[3][2] = 1; W_def[3][5] = 1
    U_def[0][7] = 1
    V_def[2][7] = 1
    U_def[1][8] = 1
    V_def[5][8] = 1
    U_def[2][9] = 1
    V_def[2][9] = 1
    U_def[3][10] = 1
    V_def[5][10] = 1
    W_def[4][7] = 1; W_def[4][8] = 1
    W_def[5][9] = 1; W_def[5][10] = 1
    
    def verify(U, V, W, rank):
        """Check if the decomposition exactly reproduces matrix multiplication."""
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    expected = 1
                    actual = 0
                    for r in range(rank):
                        actual += U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                    if actual != expected:
                        return False
        return True
    
    # Verify default
    if not verify(U_def, V_def, W_def, default_rank):
        return {"U": U_def, "V": V_def, "W": W_def}
    
    # Strategy: Try to find rank-10 by solving the tensor equations over GF(53)
    # The tensor equations are: for all (i,j,k), sum_r U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r] = 1
    # This is a system of trilinear equations. We can fix the support pattern and solve.
    
    # For rank 10, we need to find U, V, W with 10 columns such that the equations hold.
    # Key insight: we can use the structure of the problem. The matrix multiplication tensor
    # for 2x2 x 2x3 has a specific structure that might allow rank 10.
    
    # Let's try a direct algebraic approach:
    # Fix the support pattern (which entries are non-zero) and solve for the values.
    
    # For rank 10, try all possible support patterns derived from the rank-11 solution
    # by removing one column and adjusting.
    
    # Alternative: Use the fact that we can scale columns. If we have a rank-r solution,
    # we can try to merge two columns that have proportional U and V vectors.
    
    # Check if any two columns in the rank-11 solution can be merged
    for c1 in range(default_rank):
        if time.time() > deadline:
            break
        for c2 in range(c1 + 1, default_rank):
            if time.time() > deadline:
                break
            
            # Check if U[:, c1] is proportional to U[:, c2]
            # and V[:, c1] is proportional to V[:, c2]
            U1 = [U_def[i][c1] for i in range(n*m)]
            U2 = [U_def[i][c2] for i in range(n*m)]
            V1 = [V_def[i][c1] for i in range(m*p)]
            V2 = [V_def[i][c2] for i in range(m*p)]
            
            # Check proportionality
            def are_proportional(a, b):
                # Find first non-zero in a
                scale = None
                for i in range(len(a)):
                    if a[i] != 0:
                        if b[i] == 0:
                            return None
                        scale = b[i] / a[i]
                        break
                if scale is None:
                    # a is all zeros, check if b is also all zeros
                    return None if any(x != 0 for x in b) else 0
                # Check if b[i] == scale * a[i] for all i
                for i in range(len(a)):
                    if abs(b[i] - scale * a[i]) > 1e-10:
                        return None
                return scale
            
            U_scale = are_proportional(U1, U2)
            V_scale = are_proportional(V1, V2)
            
            if U_scale is not None and V_scale is not None:
                # Columns can potentially be merged
                # The merged column would have U = U1 + alpha*U2, V = V1 + beta*V2
                # where alpha*beta = 1 for the scaling to work
                # Actually, we need to be more careful.
                
                # If U2 = s*U1 and V2 = t*V1, then:
                # U2*V2 = s*t*U1*V1
                # So the contribution of column 2 is s*t times the contribution of column 1
                # We can merge by: U_merged = U1 + U2/s (if s != 0), V_merged = V1
                # Then U_merged * V_merged = (U1 + U2/s) * V1 = U1*V1 + U2*V1/s
                # But we need U2*V2 = U2*(t*V1) = t*U2*V1
                # So we need 1/s = t, i.e., s*t = 1
                
                if abs(U_scale * V_scale - 1) < 1e-10:
                    # Perfect merge! Create rank-10 solution
                    rank_10 = default_rank - 1
                    U_10 = [[0] * rank_10 for _ in range(n*m)]
                    V_10 = [[0] * rank_10 for _ in range(m*p)]
                    W_10 = [[0] * rank_10 for _ in range(p*n)]
                    
                    # Map old columns to new columns
                    col_map = {}
                    new_idx = 0
                    for c in range(default_rank):
                        if c == c2:
                            continue
                        col_map[c] = new_idx
                        new_idx += 1
                    
                    # Copy columns
                    for i in range(n*m):
                        for c in range(default_rank):
                            if c != c2:
                                U_10[i][col_map[c]] = U_def[i][c]
                    for i in range(m*p):
                        for c in range(default_rank):
                            if c != c2:
                                V_10[i][col_map[c]] = V_def[i][c]
                    for i in range(p*n):
                        for c in range(default_rank):
                            if c != c2:
                                W_10[i][col_map[c]] = W_def[i][c]
                    
                    # Merge column c2 into column c1
                    # U_merged = U1 + U2/s, V_merged = V1, W_merged = W1 + (t/s)*W2
                    # Wait, let me redo this. If U2 = s*U1, V2 = t*V1, and s*t=1:
                    # Contribution of col1: U1*V1*W1
                    # Contribution of col2: U2*V2*W2 = s*t*U1*V1*W2 = U1*V1*W2
                    # So total: U1*V1*(W1+W2)
                    # We can just add W2 to W1 and remove column 2!
                    
                    target_idx = col_map[c1]
                    for i in range(p*n):
                        W_10[i][target_idx] += W_def[i][c2]
                    
                    # Verify
                    if verify(U_10, V_10, W_10, rank_10):
                        return {"U": U_10, "V": V_10, "W": W_10}
    
    # Try a more systematic approach: search for rank-10 by modifying the support pattern
    # Use finite field GF(53) to solve the trilinear system
    
    P = 53  # Prime for GF(53)
    
    # For rank 10, we have 10 columns. The equations are trilinear.
    # Strategy: fix the support pattern, then solve.
    
    # Let's try: take the rank-11 solution, remove one column, and try to solve
    # for adjusted values in the remaining columns.
    
    # For each column to remove, set up the system and try to solve.
    
    for col_remove in range(default_rank):
        if time.time() > deadline:
            break
        
        remaining = [c for c in range(default_rank) if c != col_remove]
        new_rank = len(remaining)
        
        # Build the system: for each (i,j,k), sum over remaining r of 
        # U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r] = 1
        # This is nonlinear. But we can fix two matrices and solve for the third.
        
        # Fix U and V from the default (zeroing out col_remove), solve for W.
        # The equation becomes: for each (j,n+i) index, sum_r U[i*m+k][r]*V[k*p+j][r]*W[j*n+i][r] = 1
        # For fixed (j,n+i), this is linear in W[j*n+i][r]... no, it's still trilinear.
        
        # Actually, for fixed (i,j,k), the equation is:
        # sum_r (U[i*m+k][r] * V[k*p+j][r]) * W[j*n+i][r] = 1
        # This is linear in W[j*n+i][r] if we fix U and V!
        
        # So: fix U and V (from default, with col_remove zeroed), solve for W.
        # For each (j, i) pair (i.e., each row of W), we have m equations (one for each k)
        # and new_rank unknowns. This is underdetermined if new_rank > m.
        
        # Let's try: for each (j,i), we have m equations in new_rank unknowns.
        # We can use least squares or find a particular solution.
        
        # Actually, let's be more careful. For fixed (j,i), the equations are:
        # For k=0: sum_r U[i*m+0][r]*V[0*p+j][r]*W[j*n+i][r] = 1
        # For k=1: sum_r U[i*m+1][r]*V[1*p+j][r]*W[j*n+i][r] = 1
        
        # So for each (j,i), we have a 2 x new_rank system.
        
        # Let's set up and solve this system over the reals, then check if integer solution exists.
        
        U_fixed = [[U_def[i][r] if r != col_remove else 0 for r in range(default_rank)] for i in range(n*m)]
        V_fixed = [[V_def[i][r] if r != col_remove else 0 for r in range(default_rank)] for i in range(m*p)]
        
        # Remove col_remove from U_fixed and V_fixed
        U_rem = [[U_fixed[i][r] for r in remaining] for i in range(n*m)]
        V_rem = [[V_fixed[i][r] for r in remaining] for i in range(m*p)]
        
        # For each (j,i), solve for W[j*n+i][r]
        W_solution = [[0] * new_rank for _ in range(p*n)]
        solvable = True
        
        for j in range(p):
            for i in range(n):
                # Build 2 x new_rank system
                A_mat = np.zeros((m, new_rank))
                b_vec = np.ones(m)
                
                for k in range(m):
                    for r_idx, r in enumerate(remaining):
                        A_mat[k][r_idx] = U_rem[i*m+k][r_idx] * V_rem[k*p+j][r_idx]
                
                # Solve A_mat @ W_col = b_vec
                # This is 2 x 10 system, underdetermined. Find minimum norm solution.
                try:
                    W_col, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    # Check if solution is exact
                    if np.allclose(A_mat @ W_col, b_vec):
                        W_solution[j*n+i] = W_col.tolist()
                    else:
                        solvable = False
                        break
                except:
                    solvable = False
                    break
            
            if not solvable:
                break
        
        if solvable:
            # Check if W_solution has nice values (integers or half-integers)
            # Convert to fractions and check
            W_check = [[0] * new_rank for _ in range(p*n)]
            valid = True
            for i in range(p*n):
                for r in range(new_rank):
                    val = W_solution[i][r]
                    # Check if close to integer or half-integer
                    nearest_int = round(val)
                    nearest_half = round(val * 2) / 2
                    if abs(val - nearest_int) < 1e-6:
                        W_check[i][r] = int(nearest_int)
                    elif abs(val * 2 - round(val * 2)) < 1e-6:
                        W_check[i][r] = round(val * 2) / 2
                    else:
                        # Not nice, but let's still try
                        W_check[i][r] = val
                        valid = False
            
            # Build full solution and verify
            U_full = [[0] * new_rank for _ in range(n*m)]
            V_full = [[0] * new_rank for _ in range(m*p)]
            for i in range(n*m):
                for r_idx, r in enumerate(remaining):
                    U_full[i][r_idx] = U_def[i][r]
            for i in range(m*p):
                for r_idx, r in enumerate(remaining):
                    V_full[i][r_idx] = V_def[i][r]
            
            if verify(U_full, V_full, W_check, new_rank):
                return {"U": U_full, "V": V_full, "W": W_check}
    
    # If rank-10 search fails, return rank-11
    return {"U": U_def, "V": V_def, "W": W_def}
# EVOLVE-BLOCK-END

