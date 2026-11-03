import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        T_expected = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i,j,k] = 1.0
        
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            u_mat = u_c.reshape(n, m)
            v_mat = v_c.reshape(m, p)
            w_mat = w_c.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i,j]) < 1e-15:
                        continue
                    for k in range(p):
                        T_computed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        return np.max(np.abs(T_computed - T_expected)) < tol

    # Strassen coefficients for 2x2x2
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-e)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    U_coeffs = [
        {0: 1, 3: 1},       # M1: a, d
        {2: 1, 3: 1},       # M2: c, d
        {0: 1},             # M3: a
        {3: 1},             # M4: d
        {0: 1, 1: 1},       # M5: a, b
        {2: 1, 0: -1},      # M6: c, -a
        {1: 1, 3: -1}       # M7: b, -d
    ]
    
    V_coeffs = [
        {0: 1, 3: 1},       # M1: e, h
        {0: 1},             # M2: e
        {1: 1, 3: -1},      # M3: f, -h
        {2: 1, 0: -1},      # M4: g, -e
        {3: 1},             # M5: h
        {0: 1, 1: 1},       # M6: e, f
        {2: 1, 3: 1}        # M7: g, h
    ]
    
    W_coeffs = [
        {0: 1, 3: 1},       # M1: C00, C11
        {2: 1, 3: -1},      # M2: C10, -C11
        {1: 1, 3: 1},       # M3: C01, C11
        {0: 1, 2: 1},       # M4: C00, C10
        {0: -1, 1: 1},      # M5: -C00, C01
        {3: 1},             # M6: C11
        {0: 1}              # M7: C00
    ]
    
    def build_decomposition(i_off, j_off, k_off):
        """
        Build a decomposition with one Strassen block at (i_off, j_off, k_off)
        covering i in {i_off, i_off+1}, j in {j_off, j_off+1}, k in {k_off, k_off+1}
        and schoolbook for all other (i,j,k) triples.
        """
        rank = 7 + 24  # 7 Strassen + 24 schoolbook (36 total - 2*2*2 = 36-8=28... wait)
        # Actually: total schoolbook terms = n*m*p = 72
        # Strassen covers 2*2*2 = 8 terms with 7 multiplications
        # Remaining: 72 - 8 = 64 terms with 64 multiplications
        # Total rank = 7 + 64 = 71
        
        rank = 7 + (n*m*p - 8)
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Strassen block
        for l in range(7):
            for li, coeff in U_coeffs[l].items():
                i_local = li // 2
                j_local = li % 2
                i_global = i_off + i_local
                j_global = j_off + j_local
                U[i_global*m + j_global, col] += coeff
            for lj, coeff in V_coeffs[l].items():
                j_local = lj // 2
                k_local = lj % 2
                j_global = j_off + j_local
                k_global = k_off + k_local
                V[j_global*p + k_global, col] += coeff
            for lc, coeff in W_coeffs[l].items():
                i_local = lc // 2
                k_local = lc % 2
                i_global = i_off + i_local
                k_global = k_off + k_local
                W[k_global*n + i_global, col] += coeff
            col += 1
        
        # Schoolbook for remaining
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Skip if in Strassen block
                    if (i_off <= i < i_off+2) and (j_off <= j < j_off+2) and (k_off <= k < k_off+2):
                        continue
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def build_decomposition_pair(i_off1, j_off1, k_off1, i_off2, j_off2, k_off2):
        """
        Build a decomposition with two non-overlapping Strassen blocks.
        """
        # Check non-overlap
        overlap_i = not (i_off1+1 < i_off2 or i_off2+1 < i_off1)
        overlap_j = not (j_off1+1 < j_off2 or j_off2+1 < j_off1)
        overlap_k = not (k_off1+1 < k_off2 or k_off2+1 < k_off1)
        
        if overlap_i and overlap_j and overlap_k:
            return None  # They overlap in all three dimensions
        
        # Count schoolbook terms
        # Each Strassen block covers 8 terms
        # If they don't overlap in at least one dimension, total covered = 16
        # If they overlap in exactly one dimension, covered = 8 + 8 - 4 = 12
        # If they overlap in two dimensions, covered = 8 + 8 - 2 = 14
        # If they overlap in all three, covered = 8 (same block)
        
        covered1 = set()
        for i in range(i_off1, i_off1+2):
            for j in range(j_off1, j_off1+2):
                for k in range(k_off1, k_off1+2):
                    covered1.add((i,j,k))
        
        covered2 = set()
        for i in range(i_off2, i_off2+2):
            for j in range(j_off2, j_off2+2):
                for k in range(k_off2, k_off2+2):
                    covered2.add((i,j,k))
        
        total_covered = len(covered1 | covered2)
        schoolbook_count = n*m*p - total_covered
        rank = 14 + schoolbook_count
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # First Strassen block
        for l in range(7):
            for li, coeff in U_coeffs[l].items():
                i_local = li // 2
                j_local = li % 2
                i_global = i_off1 + i_local
                j_global = j_off1 + j_local
                U[i_global*m + j_global, col] += coeff
            for lj, coeff in V_coeffs[l].items():
                j_local = lj // 2
                k_local = lj % 2
                j_global = j_off1 + j_local
                k_global = k_off1 + k_local
                V[j_global*p + k_global, col] += coeff
            for lc, coeff in W_coeffs[l].items():
                i_local = lc // 2
                k_local = lc % 2
                i_global = i_off1 + i_local
                k_global = k_off1 + k_local
                W[k_global*n + i_global, col] += coeff
            col += 1
        
        # Second Strassen block
        for l in range(7):
            for li, coeff in U_coeffs[l].items():
                i_local = li // 2
                j_local = li % 2
                i_global = i_off2 + i_local
                j_global = j_off2 + j_local
                U[i_global*m + j_global, col] += coeff
            for lj, coeff in V_coeffs[l].items():
                j_local = lj // 2
                k_local = lj % 2
                j_global = j_off2 + j_local
                k_global = k_off2 + k_local
                V[j_global*p + k_global, col] += coeff
            for lc, coeff in W_coeffs[l].items():
                i_local = lc // 2
                k_local = lc % 2
                i_global = i_off2 + i_local
                k_global = k_off2 + k_local
                W[k_global*n + i_global, col] += coeff
            col += 1
        
        # Schoolbook for remaining
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i,j,k) in covered1 or (i,j,k) in covered2:
                        continue
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    best_result = None
    best_rank = n*m*p  # schoolbook
    
    # Search single Strassen block placements
    for i_off in range(n-1):
        for j_off in range(m-1):
            for k_off in range(p-1):
                if time_left() < 10:
                    break
                result = build_decomposition(i_off, j_off, k_off)
                if validate(result["U"], result["V"], result["W"]):
                    rank = result["U"].shape[1]
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {
                            "U": result["U"].tolist(),
                            "V": result["V"].tolist(),
                            "W": result["W"].tolist()
                        }
    
    # Search pairs of non-overlapping Strassen blocks
    for i_off1 in range(n-1):
        for j_off1 in range(m-1):
            for k_off1 in range(p-1):
                if time_left() < 5:
                    break
                for i_off2 in range(n-1):
                    for j_off2 in range(m-1):
                        for k_off2 in range(p-1):
                            if time_left() < 2:
                                break
                            # Skip if same block
                            if i_off1 == i_off2 and j_off1 == j_off2 and k_off1 == k_off2:
                                continue
                            # Check if they overlap in all three dimensions
                            overlap_i = not (i_off1+1 < i_off2 or i_off2+1 < i_off1)
                            overlap_j = not (j_off1+1 < j_off2 or j_off2+1 < j_off1)
                            overlap_k = not (k_off1+1 < k_off2 or k_off2+1 < k_off1)
                            if overlap_i and overlap_j and overlap_k:
                                continue
                            
                            result = build_decomposition_pair(i_off1, j_off1, k_off1, i_off2, j_off2, k_off2)
                            if result is None:
                                continue
                            if validate(result["U"], result["V"], result["W"]):
                                rank = result["U"].shape[1]
                                if rank < best_rank:
                                    best_rank = rank
                                    best_result = {
                                        "U": result["U"].tolist(),
                                        "V": result["V"].tolist(),
                                        "W": result["W"].tolist()
                                    }
    
    if best_result is not None:
        return best_result
    else:
        return schoolbook()
