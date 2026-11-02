import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    n, m, p = DIMENSIONS
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def try_reduce_rank(U_list, V_list, W_list, current_rank):
        """Find linear dependencies among columns and eliminate them."""
        if current_rank <= 1:
            return U_list, V_list, W_list, current_rank
        
        Ua = np.array(U_list, dtype=float)
        Va = np.array(V_list, dtype=float)
        Wa = np.array(W_list, dtype=float)
        
        # Build the 56 x current_rank matrix where each column r contributes
        # to coordinate (i,j,k) with value U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want to find which columns are linearly dependent in the sense that
        # their contribution vectors are in the span of others.
        
        # For each column r, compute its "fingerprint" vector in R^56
        # fingerprint_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But this is multiplicative, not additive. The tensor entry is sum_r of these.
        
        # Actually, the decomposition says:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # 
        # A column r is "redundant" if there exist coefficients c_s (s != r) such that
        # for all (i,j,k): U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = sum_{s!=r} c_s * U[i*m+j, s]*V[j*p+k, s]*W[k*n+i, s]
        # This is a nonlinear condition in general.
        
        # Simpler approach: check if any column has all-zero entries in U, V, or W
        # (it contributes nothing). Or check for exact duplicate columns.
        
        # Check for zero columns
        keep = []
        for r in range(current_rank):
            if np.all(Ua[:, r] == 0) or np.all(Va[:, r] == 0) or np.all(Wa[:, r] == 0):
                continue
            keep.append(r)
        
        if len(keep) < current_rank:
            U_list = [U_list[i] for i in keep]
            # Rebuild: filter rows
            U_new = [[U_list[i][r] for i in range(n*m)] for r in range(len(keep))]
            V_new = [[V_list[i][r] for i in range(m*p)] for r in range(len(keep))]
            W_new = [[W_list[i][r] for i in range(p*n)] for r in range(len(keep))]
            return U_new, V_new, W_new, len(keep)
        
        # Check for duplicate columns (same U col, same V col, same W col)
        seen = {}
        keep = []
        for r in range(current_rank):
            key = (tuple(Ua[:, r]), tuple(Va[:, r]), tuple(Wa[:, r]))
            if key in seen:
                continue
            seen[key] = r
            keep.append(r)
        
        if len(keep) < current_rank:
            U_new = [[U_list[i][r] for i in range(n*m)] for r in keep]
            V_new = [[V_list[i][r] for i in range(m*p)] for r in keep]
            W_new = [[W_list[i][r] for i in range(p*n)] for r in keep]
            return U_new, V_new, W_new, len(keep)
        
        return U_list, V_list, W_list, current_rank
    
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
            
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> c00, c11
        U[a][col] += 1; U[d][col] += 1
        V[e][col] += 1; V[h][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2 = (c+d)(e) -> c10, c11(-)
        U[c][col] += 1; U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        
        # M3 = (a)(f-h) -> c01, c11
        U[a][col] += 1
        V[f][col] += 1; V[h][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4 = (d)(g-e) -> c00, c10
        U[d][col] += 1
        V[g][col] += 1; V[e][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # M5 = (a+b)(h) -> c00(-), c01
        U[a][col] += 1; U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> c11
        U[c][col] += 1; U[a][col] -= 1
        V[e][col] += 1; V[f][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> c00
        U[b][col] += 1; U[d][col] -= 1
        V[g][col] += 1; V[h][col] += 1
        W[c00][col] += 1
        col += 1
        
        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
            
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        U[a][col] += 1; V[e][col] += 1; W[c00][col] += 1; col += 1
        U[b][col] += 1; V[g][col] += 1; W[c00][col] += 1; col += 1
        U[c][col] += 1; V[e][col] += 1; W[c10][col] += 1; col += 1
        U[d][col] += 1; V[g][col] += 1; W[c10][col] += 1; col += 1
        
        return True

    # Build decomposition: two groups of (2x2)*(2x7)
    # Group 1: j=0,1
    for k0 in [0, 2, 4]:
        add_strassen_222(0, 0, k0)
    add_schoolbook_221(0, 0, 6)
    
    # Group 2: j=2,3
    for k0 in [0, 2, 4]:
        add_strassen_222(0, 2, k0)
    add_schoolbook_221(0, 2, 6)

    # Verify the decomposition is correct using vectorized numpy
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Vectorized check: for each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Build index arrays
    max_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                reconstructed = np.dot(U_arr[a_idx] * V_arr[b_idx], W_arr[c_idx])
                err = abs(reconstructed - 1.0)
                if err > max_error:
                    max_error = err
    
    if max_error > 1e-9:
        return {"U": [[0]*rank for _ in range(n*m)], 
                "V": [[0]*rank for _ in range(m*p)], 
                "W": [[0]*rank for _ in range(p*n)]}
    
    # Try to reduce rank by finding and eliminating linearly dependent columns
    U, V, W, final_rank = try_reduce_rank(U, V, W, col)
    
    # Final verification after reduction
    if final_rank < col:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        max_error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    reconstructed = np.dot(U_arr[a_idx] * V_arr[b_idx], W_arr[c_idx])
                    err = abs(reconstructed - 1.0)
                    if err > max_error:
                        max_error = err
        if max_error > 1e-9:
            # Revert to original if reduction broke correctness
            U = [[0] * col for _ in range(n * m)]
            V = [[0] * col for _ in range(m * p)]
            W = [[0] * col for _ in range(p * n)]
            # Rebuild original
            col = 0
            for k0 in [0, 2, 4]:
                add_strassen_222(0, 0, k0)
            add_schoolbook_221(0, 0, 6)
            for k0 in [0, 2, 4]:
                add_strassen_222(0, 2, k0)
            add_schoolbook_221(0, 2, 6)
            final_rank = col
    
    return {"U": U, "V": V, "W": W}
