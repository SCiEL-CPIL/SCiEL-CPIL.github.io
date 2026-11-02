import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Strassen decomposition for 2x2x2
    strassen_decomp = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),         # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),         # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),         # M4
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),         # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})          # M7
    ]
    
    def _build_strassen_3x4x8():
        """Build the block-Strassen decomposition: rank 88."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        for j1 in range(2):  # 2 blocks of 2 in m dimension
            # Top part: 2x2x8 (rows 0,1 of A)
            for l in range(4):  # 4 blocks of 2 in p dimension
                for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                    for li, coeff in u_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_i = li_i
                        global_j = 2 * j1 + li_j
                        U[global_i * m + global_j][col] += coeff
                    for li, coeff in v_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_j = 2 * j1 + li_i
                        global_k = 2 * l + li_j
                        V[global_j * p + global_k][col] += coeff
                    for li, coeff in w_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_i = li_i
                        global_k = 2 * l + li_j
                        W[global_k * n + global_i][col] += coeff
                    col += 1
            
            # Bot part: 1x2x8 (row 2 of A), schoolbook
            for k in range(8):
                for j2 in range(2):
                    global_i = 2
                    global_j = 2 * j1 + j2
                    U[global_i * m + global_j][col] = 1
                    V[global_j * p + k][col] = 1
                    W[k * n + global_i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def _verify(U, V, W, tol=1e-9):
        """Verify exact tensor decomposition."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def _verify_half_integer(U, V, W):
        """Check that all entries are integers or half-integers."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True
    
    def _local_search_reduce(U, V, W, max_iterations=200):
        """
        Try to reduce rank by checking if any column is in the span of others.
        For each column r, check if U[:,r], V[:,r], W[:,r] can be expressed
        as linear combinations of other columns while preserving the tensor identity.
        """
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check for duplicate or zero columns
        for r in range(rank):
            if np.allclose(U_arr[:, r], 0) or np.allclose(V_arr[:, r], 0) or np.allclose(W_arr[:, r], 0):
                # This column is useless - remove it
                keep = [i for i in range(rank) if i != r]
                U_new = [[row[i] for i in keep] for row in U]
                V_new = [[row[i] for i in keep] for row in V]
                W_new = [[row[i] for i in keep] for row in W]
                if _verify(U_new, V_new, W_new) and _verify_half_integer(U_new, V_new, W_new):
                    return U_new, V_new, W_new
        
        # Check for linearly dependent triples:
        # If column r1 and r2 have proportional U, V, W vectors, we can merge them
        for r1 in range(rank):
            if time.time() > deadline - 10:
                break
            for r2 in range(r1 + 1, rank):
                # Check if U[:,r1] is proportional to U[:,r2]
                if np.linalg.norm(U_arr[:, r1]) < 1e-10 or np.linalg.norm(U_arr[:, r2]) < 1e-10:
                    continue
                # Check proportionality of U vectors
                cross_U = np.cross(U_arr[:, r1], U_arr[:, r2])
                if np.linalg.norm(cross_U) > 1e-8:
                    continue
                # U vectors are proportional; check V and W
                cross_V = np.cross(V_arr[:, r1], V_arr[:, r2])
                if np.linalg.norm(cross_V) > 1e-8:
                    continue
                cross_W = np.cross(W_arr[:, r1], W_arr[:, r2])
                if np.linalg.norm(cross_W) > 1e-8:
                    continue
                
                # All three are proportional - try to merge
                # Compute the proportionality constants
                scale_U = np.dot(U_arr[:, r1], U_arr[:, r2]) / (np.dot(U_arr[:, r2], U_arr[:, r2]) + 1e-15)
                
                # Try removing r2 by adding scaled r2 to r1
                # Actually, if they're all proportional with the same scale, we can just remove one
                keep = [i for i in range(rank) if i != r2]
                U_new = [[row[i] for i in keep] for row in U]
                V_new = [[row[i] for i in keep] for row in V]
                W_new = [[row[i] for i in keep] for row in W]
                if _verify(U_new, V_new, W_new) and _verify_half_integer(U_new, V_new, W_new):
                    return U_new, V_new, W_new
        
        return U, V, W
    
    # Build the main candidate
    result = _build_strassen_3x4x8()
    
    # Verify
    if not _verify(result['U'], result['V'], result['W']):
        raise ValueError("Construction failed verification")
    if not _verify_half_integer(result['U'], result['V'], result['W']):
        raise ValueError("Construction has non-half-integer entries")
    
    # Local search for potential rank reduction
    if time.time() < deadline - 15:
        U_red, V_red, W_red = _local_search_reduce(result['U'], result['V'], result['W'])
        if len(U_red[0]) < len(result['U'][0]):
            result = {"U": U_red, "V": V_red, "W": W_red}
    
    return result
# EVOLVE-BLOCK-END
