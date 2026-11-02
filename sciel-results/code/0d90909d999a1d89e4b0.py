import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]
_DEADLINE = time.time() + 140.0

def _validate_exact(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for c in range(rank):
                    val += U_arr[u_idx, c] * V_arr[v_idx, c] * W_arr[w_idx, c]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _find_2x3x4_rank14():
    n, m, p = 2, 3, 4
    target_rank = 14
    rng = np.random.default_rng(42)
    
    for trial in range(50):
        if time.time() > _DEADLINE - 20:
            break
            
        U = rng.standard_normal((n*m, target_rank)) * 0.1
        V = rng.standard_normal((m*p, target_rank)) * 0.1
        W = rng.standard_normal((p*n, target_rank)) * 0.1
        
        for it in range(500):
            if time.time() > _DEADLINE - 10:
                break
                
            # Update U
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    A_sys = np.zeros((p, target_rank))
                    b_vec = np.ones(p)
                    for k in range(p):
                        for c in range(target_rank):
                            A_sys[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_sys, b_vec, rcond=None)
                        U[u_idx, :] = sol
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    A_sys = np.zeros((n, target_rank))
                    b_vec = np.ones(n)
                    for i in range(n):
                        for c in range(target_rank):
                            A_sys[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_sys, b_vec, rcond=None)
                        V[v_idx, :] = sol
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    A_sys = np.zeros((m, target_rank))
                    b_vec = np.ones(m)
                    for j in range(m):
                        for c in range(target_rank):
                            A_sys[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_sys, b_vec, rcond=None)
                        W[w_idx, :] = sol
                    except:
                        pass
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(target_rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-10:
                # Try to rationalize
                def to_frac(x):
                    if abs(x - round(x)) < 1e-9:
                        return int(round(x))
                    for den in range(1, 21):
                        num = x * den
                        if abs(num - round(num)) < 1e-9:
                            return Fraction(int(round(num)), den)
                    return Fraction(x).limit_denominator(100)
                
                U_fr = [[to_frac(x) for x in row] for row in U]
                V_fr = [[to_frac(x) for x in row] for row in V]
                W_fr = [[to_frac(x) for x in row] for row in W]
                
                # Verify exactness with fractions
                exact_ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = Fraction(0)
                            for c in range(target_rank):
                                val += U_fr[i*m+j][c] * V_fr[j*p+k][c] * W_fr[k*n+i][c]
                            if val != 1:
                                exact_ok = False
                                break
                        if not exact_ok: break
                    if not exact_ok: break
                if exact_ok:
                    return U_fr, V_fr, W_fr
    return None

def construct():
    n, m, p = DIMENSIONS
    # Try to find 2x3x4 rank 14 solution
    sol_234 = _find_2x3x4_rank14()
    
    if sol_234:
        U14, V14, W14 = sol_234
        # Construct 2x3x8 solution
        U28 = []
        for row in U14:
            U28.append(row + row) # Duplicate columns
        
        V28 = []
        for j in range(m):
            for k in range(p): # p=8
                # k 0-3 use first half, 4-7 use second half (same as first)
                k_14 = k % 4
                v_idx_14 = j * 4 + k_14
                V28.append(V14[v_idx_14] + V14[v_idx_14]) # Wait, V14 rows are 12.
                # V14 has 12 rows. V28 needs 24 rows.
                # Row j*8+k in V28 corresponds to row j*4+(k%4) in V14?
                # Yes, because the algorithm is identical for both blocks.
                # So V28 row is just a copy of the corresponding V14 row?
                # No, V28 has 28 columns. V14 has 14 columns.
                # So we duplicate the 14 columns to make 28.
                # And the row content is the same as V14.
        
        # Let's re-verify the construction.
        # U28: 6x28. U14 is 6x14. U28 = [U14, U14]. Correct.
        # V28: 24x28. V14 is 12x14.
        # Row r in V28 (0..23). r = j*8 + k.
        # Corresponding row in V14 is j*4 + (k%4).
        # Content is V14[j*4+(k%4)] duplicated to 28 cols.
        
        V28 = []
        for j in range(m):
            for k in range(p):
                k_14 = k % 4
                v_idx_14 = j * 4 + k_14
                V28.append(V14[v_idx_14] + V14[v_idx_14])
        
        W28 = []
        for k in range(p):
            for i in range(n):
                k_14 = k % 4
                w_idx_14 = k_14 * 2 + i
                W28.append(W14[w_idx_14] + W14[w_idx_14])
        
        # Final check
        if _validate_exact(U28, V28, W28):
            return {"U": U28, "V": V28, "W": W28}
    
    # Fallback to schoolbook
    schoolbook_rank = n * m * p
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    return {"U": U_school, "V": V_school, "W": W_school}
