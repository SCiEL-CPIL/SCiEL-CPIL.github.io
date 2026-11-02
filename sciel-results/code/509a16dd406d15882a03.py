import numpy as np
import time
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 2, 2, 6
    
    # --- Rank 21 Solution (Strassen) ---
    rank_21 = 21
    U21 = np.zeros((n * m, rank_21))
    V21 = np.zeros((m * p, rank_21))
    W21 = np.zeros((p * n, rank_21))
    
    # Strassen coefficients
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1
        [(0, 0, 1)],                 # M2
        [(0, 1, 1), (1, 1, -1)],     # M3
        [(1, 0, 1), (0, 0, -1)],     # M4
        [(1, 1, 1)],                 # M5
        [(0, 0, 1), (0, 1, 1)],      # M6
        [(1, 0, 1), (1, 1, 1)]       # M7
    ]
    
    W_map = [
        [(0, 0, 1), (1, 1, 1)],   # M1
        [(1, 0, 1), (1, 1, -1)],  # M2
        [(0, 1, 1), (1, 1, 1)],   # M3
        [(0, 0, 1), (1, 0, 1)],   # M4
        [(0, 0, -1), (0, 1, 1)],  # M5
        [(1, 1, 1)],              # M6
        [(0, 0, 1)]               # M7
    ]
    
    for b in range(3):
        for s in range(7):
            r = b * 7 + s
            for idx, val in U_coeffs[s].items():
                U21[idx, r] = val
            for j_local, k_local, val in V_coeffs[s]:
                global_k = b * 2 + k_local
                v_idx = j_local * p + global_k
                V21[v_idx, r] = val
            for i, j_local, val in W_map[s]:
                global_k = b * 2 + j_local
                w_idx = global_k * n + i
                W21[w_idx, r] = val
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W form a valid rank-R decomposition."""
        R = U.shape[1]
        U_mat = U.reshape(n, m, R)
        V_mat = V.reshape(m, p, R)
        W_mat = W.reshape(p, n, R)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[i, j, :] * V_mat[j, k, :] * W_mat[k, i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_find_rank_20(time_limit):
        """Try to find a rank-20 decomposition by structured search."""
        R = 20
        
        # Convert to Fraction for exact arithmetic
        U_frac = [[Fraction(int(U21[i, r]), 1) for r in range(21)] for i in range(n * m)]
        V_frac = [[Fraction(int(V21[i, r]), 1) for r in range(21)] for i in range(m * p)]
        W_frac = [[Fraction(int(W21[i, r]), 1) for r in range(21)] for i in range(p * n)]
        
        def verify_exact(U_f, V_f, W_f):
            """Verify decomposition using exact arithmetic."""
            R = len(U_f[0])
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = Fraction(0, 1)
                        for r in range(R):
                            total += U_f[i * m + j][r] * V_f[j * p + k][r] * W_f[k * n + i][r]
                        if total != 1:
                            return False
            return True
        
        # Strategy: For each pair of columns (c1, c2), try to find a scalar t
        # such that replacing column c1 with (column c1 + t * column c2)
        # and then checking if any column becomes zero (or can be eliminated)
        # 
        # More systematically: we look for a linear dependency among the 21 columns.
        # If column c can be written as a linear combination of the other 20 columns,
        # we can eliminate it.
        
        # For each column c, check if it's in the span of the others.
        # This is a linear algebra problem. We'll use a simpler approach:
        # Try all pairs (c1, c2) and see if column c1 - t*column c2 is zero
        # for some t, which would mean they're linearly dependent.
        
        # Actually, a better approach: try to find if any 2x2 minor condition
        # allows us to merge two terms.
        
        # Let's try a different approach: for each pair of blocks (b1, b2)
        # and each pair of local Strassen indices (s1, s2), try to find t
        # such that the perturbation creates a dependency.
        
        # The key insight: if we can find t such that for some column c,
        # U[:,c] = t * U[:,c'] and V[:,c] = t * V[:,c'] and W[:,c] = t * W[:,c']
        # then we can eliminate one of them.
        
        # But this is too restrictive. Let's try:
        # For each pair of columns (c1, c2), check if there exists t such that
        # the "tensor contribution" of column c1 equals t times column c2.
        
        # The tensor contribution of column r is the outer product:
        # T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We need T_c1 = t * T_c2 for some t.
        
        # This gives us equations: for all (i,j,k),
        # U[i*m+j, c1] * V[j*p+k, c1] * W[k*n+i, c1] = t * U[i*m+j, c2] * V[j*p+k, c2] * W[k*n+i, c2]
        
        # Let's try all pairs and solve for t.
        
        for c1 in range(21):
            if time.time() > time_limit:
                return None
            
            for c2 in range(c1 + 1, 21):
                # Check if column c1 is a scalar multiple of column c2
                # in the tensor sense
                
                # Find all (i,j,k) where the product is nonzero in c2
                found_t = None
                consistent = True
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod_c2 = U_frac[i * m + j][c2] * V_frac[j * p + k][c2] * W_frac[k * n + i][c2]
                            prod_c1 = U_frac[i * m + j][c1] * V_frac[j * p + k][c1] * W_frac[k * n + i][c1]
                            
                            if prod_c2 != 0:
                                t_candidate = prod_c1 / prod_c2
                                if found_t is None:
                                    found_t = t_candidate
                                elif found_t != t_candidate:
                                    consistent = False
                                    break
                            else:
                                # If prod_c2 == 0, then prod_c1 must also be 0
                                if prod_c1 != 0:
                                    consistent = False
                                    break
                        if not consistent:
                            break
                    if not consistent:
                        break
                
                if consistent and found_t is not None:
                    # Column c1 = found_t * column c2 in tensor sense
                    # We can eliminate column c1
                    # Create new matrices without column c1
                    cols_keep = [c for c in range(21) if c != c1]
                    U_new = [[U_frac[i][c] for c in cols_keep] for i in range(n * m)]
                    V_new = [[V_frac[i][c] for c in cols_keep] for i in range(m * p)]
                    W_new = [[W_frac[i][c] for c in cols_keep] for i in range(p * n)]
                    
                    if verify_exact(U_new, V_new, W_new):
                        # Convert back to numpy
                        U_out = np.array([[float(x) for x in row] for row in U_new])
                        V_out = np.array([[float(x) for x in row] for row in V_new])
                        W_out = np.array([[float(x) for x in row] for row in W_new])
                        
                        # Verify with floating point
                        if check_decomposition(U_out, V_out, W_out):
                            return U_out, V_out, W_out
        
        return None
    
    # Attempt rank 20 search
    U20, V20, W20 = None, None, None
    if time.time() < deadline - 10:
        # Give it up to 10 seconds to try
        result = try_find_rank_20(deadline - 10)
        if result is not None:
            U20, V20, W20 = result
            # Verify
            if check_decomposition(U20, V20, W20):
                return {
                    "U": U20.tolist(),
                    "V": V20.tolist(),
                    "W": W20.tolist()
                }
    
    # Fallback to Rank 21
    return {
        "U": U21.tolist(),
        "V": V21.tolist(),
        "W": W21.tolist()
    }
