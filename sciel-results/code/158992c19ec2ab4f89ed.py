import numpy as np
import time
import sys

def construct():
    """
    Search for a rank-minimal exact tensor decomposition of 3x4 @ 4x8.
    
    Strategy: Global algebraic search.
    1. Set up the bilinear constraint system: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    2. Use randomized initialization with Newton/least-squares refinement
    3. Try to reduce rank below the known 88 (Strassen-based)
    4. Verify with exact rational arithmetic (fractions) to ensure correctness
    """
    n, m, p = 3, 4, 8
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # The matmul tensor: T[i,k,j] = 1 for all i in [0,n), k in [0,p), j in [0,m)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    nU = n * m  # 12
    nV = m * p  # 32
    nW = p * n  # 24
    n_constraints = n * m * p  # 96
    
    # Build the constraint Jacobian structure
    # For each constraint (i, k, j) and each rank slot c:
    #   d/dc of (U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c])
    #   = V[j*p+k,c] * W[k*n+i,c]  (for U[i*m+j,c])
    #   = U[i*m+j,c] * W[k*n+i,c]  (for V[j*p+k,c])
    #   = U[i*m+j,c] * V[j*p+k,c]  (for W[k*n+i,c])
    
    # Precompute indices for each constraint
    constraints = []
    for i in range(n):
        for k in range(p):
            for j in range(m):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
    
    def verify_exact(U, V, W):
        """Verify using exact rational arithmetic."""
        from fractions import Fraction
        # Convert to fractions for exactness
        Uf = [[Fraction(int(round(x*2)), 2) for x in row] for row in U]
        Vf = [[Fraction(int(round(x*2)), 2) for x in row] for row in V]
        Wf = [[Fraction(int(round(x*2)), 2) for x in row] for row in W]
        
        rank = len(U[0])
        for (i, k, j) in constraints:
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            s = Fraction(0)
            for c in range(rank):
                s += Uf[u_idx][c] * Vf[v_idx][c] * Wf[w_idx][c]
            if s != 1:
                return False
        return True
    
    def verify_numeric(U, V, W, tol=1e-9):
        """Quick numeric verification."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        for (i, k, j) in constraints:
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            s = np.sum(U_arr[u_idx, :] * V_arr[v_idx, :] * W_arr[w_idx, :])
            if abs(s - 1.0) > tol:
                return False
        return True
    
    def newton_step(U, V, W, constraints, max_iter=50, tol=1e-12):
        """
        Perform Newton iteration on the bilinear system.
        Variables: U (nU x rank), V (nV x rank), W (nW x rank)
        Equations: for each constraint c_idx: sum_c U[u,c]*V[v,c]*W[w,c] - 1 = 0
        """
        rank = U.shape[1]
        nvars = rank * (nU + nV + nW)
        neqs = len(constraints)
        
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for it in range(max_iter):
            # Compute residuals
            resid = np.zeros(neqs)
            for ci, (u_idx, v_idx, w_idx) in enumerate(constraints):
                resid[ci] = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :]) - 1.0
            
            if np.max(np.abs(resid)) < tol:
                break
            
            # Build Jacobian (sparse structure)
            # J[eq, var] where var = c*(nU+nV+nW) + offset
            # For equation (i,k,j) at rank slot c:
            #   d/dc: V[v,c]*W[w,c] for U[u,c]
            #         U[u,c]*W[w,c] for V[v,c]
            #         U[u,c]*V[v,c] for W[w,c]
            
            # Use Gauss-Newton style: J^T J delta = -J^T r
            # But for speed, use a simpler approach: coordinate descent on rank slots
            
            # Actually, let's use a direct Newton with the full Jacobian
            # This is expensive but we have limited constraints (96) and rank (~80-90)
            
            # For efficiency, use the fact that each equation only involves
            # rank-many triples. Let's use a vectorized approach.
            
            # Build J implicitly
            # J has shape (neqs, nvars)
            # For eq ci, var corresponding to (c, 'U', u_idx):
            #   J[ci, c*(nU+nV+nW) + u_idx] = V[v_idx, c] * W[w_idx, c]
            # For var corresponding to (c, 'V', v_idx):
            #   J[ci, c*(nU+nV+nW) + nU + v_idx] = U[u_idx, c] * W[w_idx, c]
            # For var corresponding to (c, 'W', w_idx):
            #   J[ci, c*(nU+nV+nW) + nU + nV + w_idx] = U[u_idx, c] * V[v_idx, c]
            
            # This is too large for dense. Use iterative solver.
            # Instead, use a simpler gradient-based approach with line search.
            
            # Gradient-based with adaptive step
            lr = 0.01
            for _ in range(3):
                # Compute gradient
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for ci, (u_idx, v_idx, w_idx) in enumerate(constraints):
                    r = resid[ci]
                    for c in range(rank):
                        u_val = U[u_idx, c]
                        v_val = V[v_idx, c]
                        w_val = W[w_idx, c]
                        grad_U[u_idx, c] += r * v_val * w_val
                        grad_V[v_idx, c] += r * u_val * w_val
                        grad_W[w_idx, c] += r * u_val * v_val
                
                # Normalize gradients
                gnorm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if gnorm < 1e-15:
                    break
                
                # Line search
                alpha = lr
                for _ls in range(20):
                    U_new = U - alpha * grad_U
                    V_new = V - alpha * grad_V
                    W_new = W - alpha * grad_W
                    
                    # Check if residual decreased
                    new_resid_max = 0
                    for ci, (u_idx, v_idx, w_idx) in enumerate(constraints):
                        s = np.sum(U_new[u_idx, :] * V_new[v_idx, :] * W_new[w_idx, :]) - 1.0
                        new_resid_max = max(new_resid_max, abs(s))
                    
                    if new_resid_max < np.max(np.abs(resid)):
                        U, V, W = U_new, V_new, W_new
                        break
                    alpha *= 0.5
                
                # Recompute residual
                for ci, (u_idx, v_idx, w_idx) in enumerate(constraints):
                    resid[ci] = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :]) - 1.0
        
        return U, V, W
    
    def try_rank(rank, seed=42):
        """Try to find a decomposition with given rank."""
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(nU, rank) * 0.1
        V = rng.randn(nV, rank) * 0.1
        W = rng.randn(nW, rank) * 0.1
        
        # Run Newton
        U, V, W = newton_step(U, V, W, constraints, max_iter=100)
        
        # Check if converged
        if verify_numeric(U, V, W, tol=1e-8):
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify exact
            if verify_exact(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return U_r, V_r, W_r
            else:
                # Try with integers
                U_i = np.round(U).astype(int)
                V_i = np.round(V).astype(int)
                W_i = np.round(W).astype(int)
                if verify_exact(U_i.tolist(), V_i.tolist(), W_i.tolist()):
                    return U_i.astype(float), V_i.astype(float), W_i.astype(float)
        
        return None
    
    # First, establish a baseline with the known Strassen decomposition (rank 88)
    # Then try to find something better
    
    # Known lower bound: max(n*m, m*p, p*n, n*p) = max(12, 32, 24, 24) = 32
    # But matmul rank is at least max(n, m, p)^2 / something...
    # Actually, for n x m @ m x p, rank >= max(n*m, m*p, p*n) is not right.
    # The tensor rank of matmul is at least max(n, m, p) for square, but for rectangular...
    # A trivial lower bound is max(n, p) since C has n*p entries and each rank-1 term
    # contributes at most n*p to C, but we need exactly 1 for each (i,k).
    # Actually, each rank-1 term contributes a product that's 1 for at most one (i,k,j) triple
    # in the simplest case, so rank >= n*m*p = 96? No, that's not right either.
    # 
    # The correct lower bound: the matmul tensor T[i,k,j] has rank at least
    # max over all (i,k) of (number of j's) = m, and similarly for other slices.
    # More precisely, rank >= max(n, m, p) is a weak bound.
    # For 3x4x8, the best known is likely around 50-80.
    
    # Let's try various ranks
    best_result = None
    
    # Try ranks from a reasonable range
    # Start from lower bound and go up
    # Lower bound: at least max(n, m, p) = 8, but more realistically >= n*p = 24
    # and >= m*p = 32
    
    ranks_to_try = [32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        for seed in range(10):
            if time.time() > deadline:
                break
            
            result = try_rank(rank, seed=seed)
            if result is not None:
                U, V, W = result
                best_result = (U, V, W, rank)
                break
        
        if best_result is not None:
            break
    
    # If no better result found, use the known Strassen decomposition
    if best_result is None:
        # Fall back to the hardcoded Strassen-based rank-88 solution
        U, V, W = _strassen_88()
        best_result = (U, V, W, 88)
    
    U, V, W, rank = best_result
    
    # Final verification
    if not verify_numeric(U, V, W, tol=1e-6):
        # Emergency fallback: schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((nU, rank_sb))
        V_sb = np.zeros((nV, rank_sb))
        W_sb = np.zeros((nW, rank_sb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, c] = 1
                    V_sb[j * p + k, c] = 1
                    W_sb[k * n + i, c] = 1
                    c += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _strassen_88():
    """Hardcoded Strassen-based rank-88 decomposition (fallback)."""
    n, m, p = 3, 4, 8
    rank_total = 88
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,0):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1, (0,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]
    
    def add_strassen(a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        nonlocal col
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            for (ar, ac), coeff in a_coeffs.items():
                U[(a_row_base + ar) * m + (a_col_base + ac)][col] += coeff
            for (br, bc), coeff in b_coeffs.items():
                V[(b_row_base + br) * p + (b_col_base + bc)][col] += coeff
            for (cr, cc), coeff in c_coeffs.items():
                W[(c_col_base + cc) * n + (c_row_base + cr)][col] += coeff
            col += 1
    
    def add_schoolbook_1x2x2(a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        nonlocal col
        terms = [
            ({(0,0):1}, {(0,0):1}, {(0,0):1}),
            ({(0,1):1}, {(1,0):1}, {(0,0):1}),
            ({(0,0):1}, {(0,1):1}, {(0,1):1}),
            ({(0,1):1}, {(1,1):1}, {(0,1):1}),
        ]
        for a_coeffs, b_coeffs, c_coeffs in terms:
            for (ar, ac), coeff in a_coeffs.items():
                U[(a_row_base + ar) * m + (a_col_base + ac)][col] += coeff
            for (br, bc), coeff in b_coeffs.items():
                V[(b_row_base + br) * p + (b_col_base + bc)][col] += coeff
            for (cr, cc), coeff in c_coeffs.items():
                W[(c_col_base + cc) * n + (c_row_base + cr)][col] += coeff
            col += 1
    
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    add_strassen(0, 0, 0, 2, 0, 2)
    add_strassen(0, 2, 2, 2, 0, 2)
    add_strassen(0, 0, 0, 4, 0, 4)
    add_strassen(0, 2, 2, 4, 0, 4)
    add_strassen(0, 0, 0, 6, 0, 6)
    add_strassen(0, 2, 2, 6, 0, 6)
    
    add_schoolbook_1x2x2(2, 0, 0, 0, 2, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0, 2, 0)
    add_schoolbook_1x2x2(2, 0, 0, 2, 2, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2, 2, 2)
    add_schoolbook_1x2x2(2, 0, 0, 4, 2, 4)
    add_schoolbook_1x2x2(2, 2, 2, 4, 2, 4)
    add_schoolbook_1x2x2(2, 0, 0, 6, 2, 6)
    add_schoolbook_1x2x2(2, 2, 2, 6, 2, 6)
    
    assert col == rank_total
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Verify
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    # Should not happen, but fallback
                    rank_sb = n * m * p
                    U_sb = np.zeros((n * m, rank_sb))
                    V_sb = np.zeros((m * p, rank_sb))
                    W_sb = np.zeros((p * n, rank_sb))
                    c = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_sb[ii * m + jj, c] = 1
                                V_sb[jj * p + kk, c] = 1
                                W_sb[kk * n + ii, c] = 1
                                c += 1
                    return U_sb, V_sb, W_sb
    
    return U_arr, V_arr, W_arr
