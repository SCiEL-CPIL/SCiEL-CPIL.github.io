"""
Matrix multiplication 2x4x4 with Strassen-based rank-28 decomposition.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]


def _get_strassen_patterns():
    """
    Returns the 7 Strassen patterns for 2x2 matrix multiplication.
    Local index: r*2+c where r,c in {0,1}
    """
    patterns = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns


def construct():
    t0 = time.time()
    deadline = 145.0  # Explicit deadline guard with safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    R = 28
    
    U = np.zeros((n * m, R), dtype=int)  # 8x28
    V = np.zeros((m * p, R), dtype=int)  # 16x28
    W = np.zeros((p * n, R), dtype=int)  # 8x28
    
    strassen = _get_strassen_patterns()
    
    # Four 2x2 block products:
    # (a_col_start, b_row_start, b_col_start, c_col_start)
    blocks = [
        (0, 0, 0, 0),  # A[:,0:2] x B[0:2,0:2] -> C[:,0:2]
        (2, 2, 0, 0),  # A[:,2:4] x B[2:4,0:2] -> C[:,0:2]
        (0, 0, 2, 2),  # A[:,0:2] x B[0:2,2:4] -> C[:,2:4]
        (2, 2, 2, 2),  # A[:,2:4] x B[2:4,2:4] -> C[:,2:4]
    ]
    
    col_idx = 0
    for (a_cs, b_rs, b_cs, c_cs) in blocks:
        for pattern in strassen:
            for local_idx, coeff in pattern['a'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                U[local_r * m + (a_cs + local_c), col_idx] = coeff
            
            for local_idx, coeff in pattern['b'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                V[(b_rs + local_r) * p + (b_cs + local_c), col_idx] = coeff
            
            for local_idx, coeff in pattern['c'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                W[(c_cs + local_c) * n + local_r, col_idx] = coeff
            
            col_idx += 1
    
    # Verify correctness
    U_r = U.reshape(n, m, R)
    V_r = V.reshape(m, p, R)
    W_r = W.reshape(p, n, R)
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    T_true = np.ones((n, m, p))
    
    if np.max(np.abs(T_recon - T_true)) > 1e-9:
        # Fallback: return naive rank-32 if Strassen construction failed
        U = np.zeros((n * m, 32), dtype=int)
        V = np.zeros((m * p, 32), dtype=int)
        W = np.zeros((p * n, 32), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt rank reduction: check for linearly dependent columns
    # Use integer row reduction to find independent columns
    def _find_independent_cols(mat):
        """Find indices of linearly independent columns using Gaussian elimination over Q."""
        mat = mat.astype(float)
        cols = mat.shape[1]
        rows = mat.shape[0]
        pivot_rows = []
        pivot_cols = []
        mat_copy = mat.copy()
        
        for c in range(cols):
            # Find pivot
            pivot_row = -1
            for r in range(len(pivot_rows), rows):
                if abs(mat_copy[r, c]) > 1e-10:
                    pivot_row = r
                    break
            if pivot_row == -1:
                continue
            pivot_cols.append(c)
            pivot_rows.append(pivot_row)
            # Eliminate
            mat_copy[pivot_row, c] = 1.0
            for r in range(rows):
                if r != pivot_row and abs(mat_copy[r, c]) > 1e-10:
                    factor = mat_copy[r, c]
                    mat_copy[r, :] -= factor * mat_copy[pivot_row, :]
        
        return pivot_cols
    
    # Check if all three matrices have full column rank
    # If any matrix has dependent columns, the decomposition might be reducible
    # But we need to check if the dependent structure is consistent across all three
    
    # Try removing columns one at a time and re-verifying
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_R = R
    
    for trial in range(current_R):
        if time.time() - t0 > deadline - 5:
            break
        # Try removing column 'trial'
        mask = [i for i in range(current_R) if i != trial]
        test_U = current_U[:, mask]
        test_V = current_V[:, mask]
        test_W = current_W[:, mask]
        
        U_tr = test_U.reshape(n, m, -1)
        V_tr = test_V.reshape(m, p, -1)
        W_tr = test_W.reshape(p, n, -1)
        T_test = np.einsum('ijr,jkr,kir->ijk', U_tr, V_tr, W_tr)
        
        if np.max(np.abs(T_test - T_true)) < 1e-9:
            current_U, current_V, current_W = test_U, test_V, test_W
            current_R -= 1
    
    U_final = current_U.astype(int)
    V_final = current_V.astype(int)
    W_final = current_W.astype(int)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
