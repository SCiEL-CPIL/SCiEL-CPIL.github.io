import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Find a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy:
    1. Construct a rank-36 solution using Strassen for 2x2x2 blocks + schoolbook for 2x2x1.
    2. Attempt to reduce the rank to 35, 34, etc. using least-squares optimization.
    3. Verify exactness with integer/half-integer snapping.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    time_limit = 140  # Leave margin for 150s limit
    
    # --- Helper to verify a solution ---
    def verify_solution(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                val = 0.0
                                for mu in range(R):
                                    val += U[a*m+b, mu] * V[c*p+d, mu] * W[k*n+i, mu]
                                if b == c and a == i and d == k:
                                    if abs(val - 1.0) > 1e-9:
                                        return False
                                else:
                                    if abs(val - 0.0) > 1e-9:
                                        return False
        return True

    def snap_to_half_integers(U, V, W):
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        return U_snapped, V_snapped, W_snapped

    # --- Precompute constraint structure ---
    cons_list = []
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            target = 1.0 if (b == c and a == i and d == k) else 0.0
                            cons_list.append((i, k, a, b, c, d, target))
    
    cons_idx = np.array(cons_list)
    cons_i = cons_idx[:, 0].astype(int)
    cons_k = cons_idx[:, 1].astype(int)
    cons_a = cons_idx[:, 2].astype(int)
    cons_b = cons_idx[:, 3].astype(int)
    cons_c = cons_idx[:, 4].astype(int)
    cons_d = cons_idx[:, 5].astype(int)
    cons_target = cons_idx[:, 6]
    
    def residual_vector_fast(x, R):
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        
        U_flat = x[0:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape(nm, R)
        V = V_flat.reshape(mp, R)
        W = W_flat.reshape(pn, R)
        
        U_sel = U[cons_a * m + cons_b, :]
        V_sel = V[cons_c * p + cons_d, :]
        W_sel = W[cons_k * n + cons_i, :]
        
        res = np.einsum('ir,ir,ir->i', U_sel, V_sel, W_sel) - cons_target
        return res

    # --- Build rank-36 solution using mixed-radix Winograd ---
    def get_initial_solution_rank36():
        rank = 36
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # M1: (A00+A11)(B00+B11)
            U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M2: (A10+A11)B00
            U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            
            # M3: A00(B10-B11)
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M4: A11(B01-B00)
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1; W[k0*n + (i0+1), col] += 1
            col += 1
            
            # M5: (A00+A01)B11
            U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1; W[(k0+1)*n + i0, col] += 1
            col += 1
            
            # M6: (A10-A00)(B00+B01)
            U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M7: (A01-A11)(B10+B11)
            U[i0*m + (j0+1), col] += 1; U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1
            
            return col
        
        def add_schoolbook_2x2x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(2):
                    U[(i0 + di)*m + (j0 + dj), col] += 1
                    V[(j0 + dj)*p + k0, col] += 1
                    W[k0*n + (i0 + di), col] += 1
                    col += 1
            return col
        
        # 4 Strassen 2x2x2 blocks for p-dim 0-3, m-dim 0-3
        add_strassen_222(0, 0, 0)
        add_strassen_222(0, 2, 0)
        add_strassen_222(0, 0, 2)
        add_strassen_222(0, 2, 2)
        # 2 schoolbook 2x2x1 blocks for p-dim 4
        add_schoolbook_2x2x1(0, 0, 4)
        add_schoolbook_2x2x1(0, 2, 4)
        
        return U, V, W

    U36, V36, W36 = get_initial_solution_rank36()
    
    # Verify the rank-36 solution
    if verify_solution(U36, V36, W36):
        U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
        if verify_solution(U_snap, V_snap, W_snap):
            best_solution = (U_snap, V_snap, W_snap)
            
            # Attempt to find lower ranks with more attempts and better init
            for R in range(35, 25, -1):
                if time.time() - start_time > time_limit:
                    break
                
                # Try multiple initializations
                for attempt in range(8):
                    if time.time() - start_time > time_limit - 10:
                        break
                    
                    if attempt == 0:
                        # Truncate rank 36 solution
                        U_init = U36[:, :R].copy()
                        V_init = V36[:, :R].copy()
                        W_init = W36[:, :R].copy()
                    elif attempt == 1:
                        # Perturbed truncation
                        U_init = U36[:, :R].copy() + np.random.normal(0, 0.1, (nm, R))
                        V_init = V36[:, :R].copy() + np.random.normal(0, 0.1, (mp, R))
                        W_init = W36[:, :R].copy() + np.random.normal(0, 0.1, (pn, R))
                    elif attempt == 2:
                        # Use first R columns with some zeroing
                        U_init = U36[:, :R].copy()
                        V_init = V36[:, :R].copy()
                        W_init = W36[:, :R].copy()
                        # Zero out some random entries to encourage sparsity
                        for _ in range(R):
                            idx = np.random.randint(0, nm)
                            U_init[idx, np.random.randint(0, R)] = 0
                    else:
                        # Random with structure
                        U_init = np.random.normal(0, 0.3, (nm, R))
                        V_init = np.random.normal(0, 0.3, (mp, R))
                        W_init = np.random.normal(0, 0.3, (pn, R))
                    
                    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                    
                    try:
                        result = opt.least_squares(residual_vector_fast, x0, args=(R,), method='lm', max_nfev=8000)
                        if result.success or result.cost < 1e-6:
                            x_opt = result.x
                            size_U = nm * R
                            size_V = mp * R
                            size_W = pn * R
                            U_opt = x_opt[0:size_U].reshape(nm, R)
                            V_opt = x_opt[size_U:size_U+size_V].reshape(mp, R)
                            W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(pn, R)
                            
                            res = residual_vector_fast(x_opt, R)
                            max_res = np.max(np.abs(res))
                            
                            if max_res < 1e-3:
                                # High-precision polish
                                if time.time() - start_time < time_limit - 15:
                                    try:
                                        result2 = opt.least_squares(residual_vector_fast, x_opt, args=(R,), method='lm', max_nfev=5000, xtol=1e-14, ftol=1e-14)
                                        if result2.success:
                                            x_opt = result2.x
                                            U_opt = x_opt[0:size_U].reshape(nm, R)
                                            V_opt = x_opt[size_U:size_U+size_V].reshape(mp, R)
                                            W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(pn, R)
                                            res = residual_vector_fast(x_opt, R)
                                            max_res = np.max(np.abs(res))
                                    except Exception:
                                        pass
                            
                            if max_res < 1e-5:
                                U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                                if verify_solution(U_snap, V_snap, W_snap):
                                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    except Exception:
                        pass
            
            return {"U": best_solution[0].tolist(), "V": best_solution[1].tolist(), "W": best_solution[2].tolist()}
        else:
            return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
    else:
        # Fallback: schoolbook
        rank = nm * mp
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
