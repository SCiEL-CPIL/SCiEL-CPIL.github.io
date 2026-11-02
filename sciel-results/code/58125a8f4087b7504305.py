import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    def build_base_terms():
        rank_terms = []
        
        def add_strassen_2x2x2(a_offset, b_offset, c_offset, sign=1):
            i0, j0 = a_offset
            j1, k0 = b_offset
            c_i0, c_k0 = c_offset
            terms = [
                ({(i0, j0): 1, (i0+1, j0+1): 1},
                 {(j0, k0): 1, (j0+1, k0+1): 1},
                 {(i0, k0): 1, (i0+1, k0+1): 1}),
                ({(i0+1, j0): 1, (i0+1, j0+1): 1},
                 {(j0, k0): 1},
                 {(i0+1, k0): 1, (i0+1, k0+1): -1}),
                ({(i0, j0): 1},
                 {(j0, k0+1): 1, (j0+1, k0+1): -1},
                 {(i0, k0+1): 1, (i0+1, k0+1): 1}),
                ({(i0+1, j0+1): 1},
                 {(j0+1, k0): 1, (j0, k0): -1},
                 {(i0, k0): 1, (i0+1, k0): 1}),
                ({(i0, j0): 1, (i0, j0+1): 1},
                 {(j0+1, k0+1): 1},
                 {(i0, k0): -1, (i0, k0+1): 1}),
                ({(i0+1, j0): 1, (i0, j0): -1},
                 {(j0, k0): 1, (j0, k0+1): 1},
                 {(i0+1, k0+1): 1}),
                ({(i0, j0+1): 1, (i0+1, j0+1): -1},
                 {(j0+1, k0): 1, (j0+1, k0+1): 1},
                 {(i0, k0): 1}),
            ]
            for u, v, w in terms:
                u_scaled = {k: val * sign for k, val in u.items()}
                v_scaled = {k: val * sign for k, val in v.items()}
                w_scaled = {k: val * sign for k, val in w.items()}
                rank_terms.append((u_scaled, v_scaled, w_scaled))

        def add_schoolbook_2x2x1(a_offset, b_offset, c_offset, sign=1):
            i0, j0 = a_offset
            j1, k0 = b_offset
            c_i0, c_k0 = c_offset
            terms = [
                ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
                ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
                ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
                ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
            ]
            for u, v, w in terms:
                u_scaled = {k: val * sign for k, val in u.items()}
                v_scaled = {k: val * sign for k, val in v.items()}
                w_scaled = {k: val * sign for k, val in w.items()}
                rank_terms.append((u_scaled, v_scaled, w_scaled))

        add_strassen_2x2x2((0, 0), (0, 0), (0, 0))
        add_strassen_2x2x2((0, 2), (2, 0), (0, 0))
        add_strassen_2x2x2((0, 0), (0, 2), (0, 2))
        add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))
        add_strassen_2x2x2((0, 2), (2, 2), (0, 2))
        add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))
        
        return rank_terms

    def terms_to_matrices(terms):
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j, col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k, col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i, col] = coeff

        return U, V, W

    def verify_decomposition(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        u_val = U[i * m + j, r]
                        v_val = V[j * p + k, r]
                        w_val = W[k * n + i, r]
                        total += u_val * v_val * w_val
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def try_reduce_rank(U, V, W):
        rank = U.shape[1]
        # Try to remove each column and solve for W
        for col_to_remove in range(rank):
            if time.time() > deadline - 10:
                break
            
            # Remove column
            idx = [c for c in range(rank) if c != col_to_remove]
            U_rem = U[:, idx]
            V_rem = V[:, idx]
            
            # Build linear system for W_rem
            # Equation: sum_s U_rem[i*m+j, s] * V_rem[j*p+k, s] * W_rem[k*n+i, s] = 1
            # Variables: W_rem[k*n+i, s]
            # Number of variables: (p*n) * (rank-1)
            # Number of equations: n*m*p
            
            num_vars = (p * n) * (rank - 1)
            num_eqs = n * m * p
            
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            # Map equation index to (i, j, k)
            # Map variable index to (k, i, s)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Coefficients for W[k*n+i, s]
                        for s in range(rank - 1):
                            u_val = U_rem[i * m + j, s]
                            v_val = V_rem[j * p + k, s]
                            if u_val != 0 and v_val != 0:
                                var_idx = (k * n + i) * (rank - 1) + s
                                A[eq_idx, var_idx] = u_val * v_val
                        eq_idx += 1
            
            # Solve A W = b
            try:
                # Use least squares
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                # Check if solution is exact
                if residuals is not None and len(residuals) > 0:
                    if residuals[0] > 1e-6:
                        continue
                
                W_rem = W_flat.reshape((p * n, rank - 1))
                
                # Verify the solution
                if verify_decomposition(U_rem, V_rem, W_rem):
                    return U_rem, V_rem, W_rem
            except np.linalg.LinAlgError:
                continue
        
        return None

    # Build base decomposition
    base_terms = build_base_terms()
    U, V, W = terms_to_matrices(base_terms)
    
    # Verify base decomposition
    if not verify_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    reduced = try_reduce_rank(U, V, W)
    if reduced:
        U, V, W = reduced
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
