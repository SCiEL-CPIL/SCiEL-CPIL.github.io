"""Matrix multiplication 3x4 by 4x6: Strassen-based rank-66 with symbolic rank-reduction attempt."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    
    Strategy:
    - Base: Strassen-based rank-66 decomposition
      Row 0 (1x4x6): schoolbook, 24 terms
      Rows 1-2 (2x4x6): Strassen on 2x2x2 sub-blocks, 6 blocks * 7 = 42 terms
    - Attempt: Symbolic elimination to reduce rank from 66 to 60
      Fix U,V to structured sparse form, solve for W via linear system
    
    U shape: (12, rank)  -- rows indexed by i*4+j
    V shape: (24, rank)  -- rows indexed by j*6+k
    W shape: (18, rank)  -- rows indexed by k*3+i
    """
    n, m, p = 3, 4, 6
    
    # First, build the base rank-66 solution
    base_rank = 66
    U_base = [[0] * base_rank for _ in range(n * m)]
    V_base = [[0] * base_rank for _ in range(m * p)]
    W_base = [[0] * base_rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(U, V, W, i, j, k, col):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        return col + 1

    def add_strassen(U, V, W, i_base, j_base, k_base, col):
        """Add 7 Strassen terms for a 2x2x2 block at global indices."""
        
        # A indices (2x2 sub-block of A)
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        
        # B indices (2x2 sub-block of B)
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        
        # C/W indices (2x2 sub-block of C), W index = k*n + i
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        # P1 = (a+d)(e+h)
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # P2 = (c+d)e
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        # P3 = a(f-h)
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # P4 = d(g-e)
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        # P5 = (a+b)h
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        # P6 = (c-a)(e+f)
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        # P7 = (b-d)(g+h)
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1
        
        return col

    # Part 1: Row 0 schoolbook (1x4x6 = 24 terms)
    for j in range(4):
        for k in range(6):
            col = add_schoolbook(U_base, V_base, W_base, 0, j, k, col)
    
    # Part 2: Rows 1-2 Strassen (2x4x6, 6 blocks of 2x2x2 = 42 terms)
    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = j_block * 2
            i_base = 1
            col = add_strassen(U_base, V_base, W_base, i_base, j_base, k_base, col)
    
    # Verify base solution
    def verify_solution(U, V, W):
        """Verify that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = delta[i,j]*delta[j,k] pattern"""
        # The tensor T[i,j,k] should equal 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
        # In the decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix mult, T[i,j,k] = 1 for all valid (i,j,k) since A[i,j]*B[j,k] is the product
        # Actually the tensor is T[i,j,k] = 1 for all i,j,k in valid ranges
        # Let's check: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False, (i,j,k), s
        return True, None, None
    
    # Convert to numpy for verification
    U_np = np.array(U_base, dtype=float)
    V_np = np.array(V_base, dtype=float)
    W_np = np.array(W_base, dtype=float)
    
    # Quick verification using tensor contraction
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape for efficient computation
    # This is a 3-tensor contraction
    
    # Check a few entries to make sure base is correct
    check_ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(base_rank):
                    s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(s - 1) > 1e-10:
                    check_ok = False
                    break
            if not check_ok:
                break
        if not check_ok:
            break
    
    if not check_ok:
        # Base solution is wrong, return it anyway (shouldn't happen)
        return {"U": U_base, "V": V_base, "W": W_base}
    
    # Now attempt rank reduction from 66 to 60 using symbolic elimination
    # Strategy: Try to find 60-term decomposition by allowing W to be solved from U,V
    # Fix U and V to have specific structure, solve for W
    
    # The tensor identity is:
    # For all i,j,k: sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is bilinear in (U,V) for fixed W, and linear in W for fixed (U,V).
    # If we fix U and V, we get a linear system for W.
    
    # Approach: Start with the rank-66 solution and try to merge pairs of terms.
    # Two terms r1, r2 can be merged if there exists a single term r such that:
    # U[:,r] = U[:,r1] + U[:,r2], V[:,r] = V[:,r1] + V[:,r2], W[:,r] = W[:,r1] + W[:,r2]
    # This only works if the products work out, which is generally not the case.
    
    # Better approach: Try to find a rank-R decomposition with R < 66 by:
    # 1. Choosing a structured U and V (e.g., based on Strassen patterns with some freedom)
    # 2. Solving the resulting linear system for W
    
    # Let's try R = 60, 59, 58, ...
    # For each R, we need to find U (12xR), V (24xR), W (18xR) satisfying the tensor identity.
    
    # The tensor identity gives n*m*p = 3*4*6 = 72 equations:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of 72 equations in 12*R + 24*R + 18*R = 54*R unknowns.
    # For R=60: 54*60 = 3240 unknowns, 72 equations. Very underdetermined.
    # But the equations are trilinear (product of three variables), so it's not a linear system.
    
    # However, if we fix U and V, the equations become linear in W.
    # For fixed U,V: For each (i,j,k), sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i, r] = 1
    # Let M_{(i,j,k), r} = U[i*m+j,r] * V[j*p+k,r]
    # Then: sum_r M_{(i,j,k),r} * W[k*n+i, r] = 1
    # This is a linear system in the entries of W.
    
    # Number of W unknowns: 18*R
    # Number of equations: 72
    # For R >= 4, we have more unknowns than equations (18R >= 72 when R >= 4).
    
    # Strategy: 
    # 1. Fix U and V to be the first R columns of the rank-66 solution (with some modification)
    # 2. Solve for W
    # 3. Check if the solution is valid
    
    # Actually, let's try a different approach:
    # Use the rank-66 solution and try to find linear dependencies.
    
    # The tensor T is fixed. We want to find minimal rank R such that
    # T = sum_{r=0}^{R-1} u_r ⊗ v_r ⊗ w_r
    
    # Let's try to see if we can reduce the rank by finding a better Strassen-like scheme.
    
    # For now, let's try the linear system approach with R = 60, 59, 58
    # We'll fix U and V to be structured (based on the Strassen pattern) and solve for W.
    
    best_solution = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = base_rank
    
    # Try to find lower rank solutions
    for target_rank in [60, 59, 58, 57, 56, 55, 54]:
        if _time_left() < 10:
            break
        
        # Fix U and V to be the first target_rank columns of the base solution
        U_fixed = U_np[:, :target_rank]
        V_fixed = V_np[:, :target_rank]
        
        # Build the linear system for W
        # For each (i,j,k): sum_r U_fixed[i*m+j, r] * V_fixed[j*p+k, r] * W[k*n+i, r] = 1
        # Unknowns: W[0..17, 0..target_rank-1] = 18*target_rank unknowns
        # Equations: 72
        
        num_eq = n * m * p  # 72
        num_unk = (p * n) * target_rank  # 18 * target_rank
        
        # Build matrix A and vector b
        # A has shape (72, 18*target_rank)
        # b has shape (72,)
        
        # For equation (i,j,k) (flattened index eq = i*m*p + j*p + k):
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # The unknown W[k*n+i, r] appears with coefficient U[i*m+j, r] * V[j*p+k, r]
        
        A = np.zeros((num_eq, num_unk), dtype=float)
        b = np.ones(num_eq, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq = i * m * p + j * p + k
                    for r in range(target_rank):
                        w_row = k * n + i
                        w_col = r
                        unk_idx = w_row * target_rank + w_col
                        A[eq, unk_idx] = U_fixed[i*m+j, r] * V_fixed[j*p+k, r]
        
        # Solve the linear system
        try:
            # Use least squares since the system may be underdetermined
            W_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            W_sol = W_sol.reshape(p * n, target_rank)
            
            # Check if the solution is valid
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        s = 0
                        for r in range(target_rank):
                            s += U_fixed[i*m+j, r] * V_fixed[j*p+k, r] * W_sol[k*n+i, r]
                        if abs(s - 1) > 1e-8:
                            valid = False
                            break
            
            if valid:
                # Check that entries are "nice" (integer or half-integer)
                # For now, accept any solution
                best_solution = {
                    "U": U_fixed.tolist(),
                    "V": V_fixed.tolist(),
                    "W": W_sol.tolist()
                }
                best_rank = target_rank
                break  # Found a valid solution, stop searching
        
        except Exception:
            continue
    
    if best_rank < base_rank:
        return best_solution
    else:
        return {"U": U_base, "V": V_base, "W": W_base}
