"""Matrix multiplication 4x4x6 via 4x4x4 Strassen + 4x4x2 Strassen-block."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np

def _strassen_2x2():
    """2x2x2 Strassen factors. U: (4,7), V: (4,7), W: (4,7)."""
    n = m = p = 2
    r = 7
    U = [[0.0]*r for _ in range(4)]
    V = [[0.0]*r for _ in range(4)]
    W = [[0.0]*r for _ in range(4)]
    prods = [
        ([(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)]),
        ([(1,0,1.0),(1,1,1.0)], [(0,0,1.0)], [(1,0,1.0),(1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0),(1,1,-1.0)], [(0,1,1.0),(1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(1,0,1.0)]),
        ([(0,0,1.0),(0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0),(0,1,1.0)]),
        ([(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0),(1,1,-1.0)], [(1,0,1.0),(1,1,1.0)], [(0,0,1.0)]),
    ]
    for c,(ub,vb,wb) in enumerate(prods):
        for (i,j,cv) in ub: U[i*m+j][c] += cv
        for (j,k,cv) in vb: V[j*p+k][c] += cv
        for (i,k,cv) in wb: W[k*n+i][c] += cv
    return U, V, W

def _build_444():
    """4x4x4 Strassen via Kronecker of 2x2 Strassen. Returns U(16,49), V(16,49), W(16,49)."""
    Us, Vs, Ws = _strassen_2x2()
    r = 49
    U = [[0.0]*r for _ in range(16)]
    V = [[0.0]*r for _ in range(16)]
    W = [[0.0]*r for _ in range(16)]
    for i1 in range(2):
        for i2 in range(2):
            i = 2*i1+i2
            for j1 in range(2):
                for j2 in range(2):
                    j = 2*j1+j2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7*r1+r2
                            U[i*4+j][col] = Us[i1*2+j1][r1]*Us[i2*2+j2][r2]
    for j1 in range(2):
        for j2 in range(2):
            j = 2*j1+j2
            for k1 in range(2):
                for k2 in range(2):
                    k = 2*k1+k2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7*r1+r2
                            V[j*4+k][col] = Vs[j1*2+k1][r1]*Vs[j2*2+k2][r2]
    for k1 in range(2):
        for k2 in range(2):
            k = 2*k1+k2
            for i1 in range(2):
                for i2 in range(2):
                    i = 2*i1+i2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7*r1+r2
                            W[k*4+i][col] = Ws[k1*2+i1][r1]*Ws[k2*2+i2][r2]
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return float(np.max(np.abs(T - 1.0)))

def _try_zero_column(Ua, Va, Wa, n, m, p, R, col, tol=1e-9):
    """Try to zero out column `col` by adjusting other columns.
    Returns (success, new_Ua, new_Va, new_Wa) where success is bool."""
    # Tensor contribution of column col: T_col[i,j,k] = Ua[i*m+j,col]*Va[j*p+k,col]*Wa[k*n+i,col]
    # We need to find delta_U, delta_V, delta_W (zero in column col) such that
    # sum_{c != col} (U_c + dU_c)(V_c + dV_c)(W_c + dW_c) = T_col
    # This is complex. Instead, use a simpler approach:
    # Project the column's tensor onto the span of other columns' outer products.
    # 
    # Alternative: use the fact that if we can express T_col as a sum of at most
    # rank-1 tensors from the other columns, we can zero it.
    # 
    # Practical approach: try to solve for adjustments in one matrix at a time.
    # If we fix V and W, we need sum_c U_c * (V_c outer W_c contracted) = T_col
    # This is a linear system in U.
    
    # Build the "effective" tensor for each other column
    # For column c: T_c[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
    # If we adjust U by dU, the change is: sum_c dU[i,j,c] * V[j,k,c] * W[k,i,c]
    # We want this to equal T_col[i,j,k]
    
    # Linear system: for each (i,j), sum_c dU[i,j,c] * (V[j,:,c] * W[:,i,c] reshaped) = T_col[i,j,:]
    # Actually: sum_c dU[i,j,c] * V[j,k,c] * W[k,i,c] = T_col[i,j,k] for each k
    
    # This is a linear system: A_{(i,j,k), (i',j',c)} = delta_{i,i'} delta_{j,j'} V[j,k,c] W[k,i,c]
    # But this is huge. Let's use a different approach.
    
    # Simpler: iterate. For each other column c, try to add a fraction of its
    # "pattern" to compensate.
    
    # Actually, let's use the rank-1 decomposition perspective:
    # T_col is a tensor of rank at most 1 (it's u_col * v_col * w_col outer product).
    # We need to express it using the other columns.
    # 
    # Try: for each pair (i,j), solve for dU[i,j,:] such that
    # sum_c dU[i,j,c] * V[j,:,c] * W[:,i,c] = T_col[i,j,:]
    # This is a linear system in dU[i,j,:] with p unknowns and p equations (one per k).
    # Matrix: M[i,j] where M[i,j][k,c] = V[j*p+k,c] * W[k*n+i,c] for c != col
    # RHS: T_col[i,j,k] = U[i*m+j,col]*Va[j*p+k,col]*Wa[k*n+i,col]
    
    dU = np.zeros_like(Ua)
    for i in range(n):
        for j in range(m):
            # Build system: for each k, sum_{c != col} dU[i,j,c] * V[j,k,c] * W[k,i,c] = T_col[i,j,k]
            # System matrix: p x (R-1)
            # RHS: p-vector
            rhs = np.zeros(p)
            for k in range(p):
                rhs[k] = Ua[i*m+j, col] * Va[j*p+k, col] * Wa[k*n+i, col]
            
            if np.max(np.abs(rhs)) < tol:
                continue  # This (i,j) slice is already zero in T_col
            
            # Build A: p x (R-1)
            other_cols = [c for c in range(R) if c != col]
            A = np.zeros((p, len(other_cols)))
            for idx_c, c in enumerate(other_cols):
                for k in range(p):
                    A[k, idx_c] = Va[j*p+k, c] * Wa[k*n+i, c]
            
            # Solve least squares
            try:
                sol, residuals, rank_A, sv = np.linalg.lstsq(A, rhs, rcond=1e-12)
            except:
                return False, Ua, Va, Wa
            
            # Check if solution is good
            if len(residuals) > 0 and residuals[0] > 1e-6 * max(1.0, np.dot(rhs, rhs)):
                continue  # Can't represent this slice well
            
            for idx_c, c in enumerate(other_cols):
                dU[i*m+j, c] += sol[idx_c]
    
    # Check if zeroing works
    Ua_new = Ua.copy()
    Ua_new[:, col] = 0.0
    Ua_new += dU
    
    # Verify
    Ur = Ua_new.reshape(n, m, R)
    Vr = Va.reshape(m, p, R)
    Wr = Wa.reshape(p, n, R)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = float(np.max(np.abs(T - 1.0)))
    
    if err < tol:
        return True, Ua_new, Va, Wa
    return False, Ua, Va, Wa

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    Us, Vs, Ws = _strassen_2x2()
    U444, V444, W444 = _build_444()
    
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442  # 77

    U = [[0.0]*total_rank for _ in range(n*m)]  # 16 x 77
    V = [[0.0]*total_rank for _ in range(m*p)]  # 24 x 77
    W = [[0.0]*total_rank for _ in range(p*n)]  # 24 x 77

    # Part 1: 4x4x4 Strassen for columns 0-3 of B -> columns 0-3 of C
    for idx in range(16):
        for c in range(rank_444):
            if U444[idx][c] != 0:
                U[idx][c] = U444[idx][c]
    for j in range(4):
        for k in range(4):
            for c in range(rank_444):
                if V444[j*4+k][c] != 0:
                    V[j*6+k][c] = V444[j*4+k][c]
    for k in range(4):
        for i in range(4):
            for c in range(rank_444):
                if W444[k*4+i][c] != 0:
                    W[k*4+i][c] = W444[k*4+i][c]

    # Part 2: 4x4x2 Strassen-block for columns 4-5 of B -> columns 4-5 of C
    products = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    
    for p_idx, (I, J) in enumerate(products):
        col_off = rank_444 + p_idx*7
        bot = J
        
        for i0 in range(2):
            for j0 in range(2):
                i = 2*I + i0
                j = 2*J + j0
                for s in range(7):
                    v = Us[2*i0+j0][s]
                    if v: U[i*4+j][col_off+s] += v
        for j0 in range(2):
            for k0 in range(2):
                jg = 2*bot + j0
                kg = 4 + k0
                for s in range(7):
                    v = Vs[2*j0+k0][s]
                    if v: V[jg*6+kg][col_off+s] += v
        for i0 in range(2):
            for k0 in range(2):
                ig = 2*I + i0
                kg = 4 + k0
                for s in range(7):
                    v = Ws[2*k0+i0][s]
                    if v: W[kg*4+ig][col_off+s] += v

    err = _verify(U, V, W, n, m, p, total_rank)
    
    if err > 1e-6:
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Now attempt rank reduction via coordinate descent
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    R = total_rank
    best_rank = R
    
    # Try to zero columns one at a time, iterating
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        for col in range(R-1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            # Skip if column is already zero
            if np.max(np.abs(Ua[:, col])) < 1e-12 and np.max(np.abs(Va[:, col])) < 1e-12 and np.max(np.abs(Wa[:, col])) < 1e-12:
                continue
            success, Ua_new, Va_new, Wa_new = _try_zero_column(Ua, Va, Wa, n, m, p, R, col)
            if success:
                Ua, Va, Wa = Ua_new, Va_new, Wa_new
                R -= 1
                best_rank = R
                changed = True
                # Remove the zeroed column from arrays
                mask = np.ones(R+1, dtype=bool)
                mask[col] = False
                Ua = Ua[:, mask]
                Va = Va[:, mask]
                Wa = Wa[:, mask]
                break  # Restart from beginning after a change
    
    # Build final result
    if best_rank < total_rank:
        U_out = [[0]*best_rank for _ in range(n*m)]
        V_out = [[0]*best_rank for _ in range(m*p)]
        W_out = [[0]*best_rank for _ in range(p*n)]
        for i in range(n*m):
            for c in range(best_rank):
                U_out[i][c] = float(Ua[i, c])
        for i in range(m*p):
            for c in range(best_rank):
                V_out[i][c] = float(Va[i, c])
        for i in range(p*n):
            for c in range(best_rank):
                W_out[i][c] = float(Wa[i, c])
        
        # Final verification
        final_err = _verify(U_out, V_out, W_out, n, m, p, best_rank)
        if final_err > 1e-6:
            # Fallback to original
            return {"U": U, "V": V, "W": W}
        return {"U": U_out, "V": V_out, "W": W_out}
    
    return {"U": U, "V": V, "W": W}
