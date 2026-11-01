"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from numpy import linalg as LA

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s kill

    # --- Schoolbook decomposition (rank = n*m*p = 28) ---
    def schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # --- Fast verification using einsum ---
    def verify_fast(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-9)

    # --- Check for integer/half-integer entries ---
    def is_valid_entries(*mats):
        for mat in mats:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        return True

    # --- Convert to list format ---
    def to_list(mat):
        mat_rounded = np.round(mat * 2) / 2.0
        if np.allclose(mat_rounded, np.round(mat_rounded), atol=1e-9):
            return np.round(mat_rounded).astype(int).tolist()
        else:
            return mat_rounded.tolist()

    # --- Strassen rank-8 core for 2x2x2 (indices i,k,j in {0,1}) ---
    # Classic Strassen decomposition.
    def strassen_core():
        # Returns U_core (4x8), V_core (2x8), W_core (2x8) for the 2x2x2 block
        # U indexed by (i,k) in row-major: i*2+k
        # V indexed by (k,j): k*2+j
        # W indexed by (j,i): j*2+i
        U = np.array([
            [1, 0, 0, 1, 0, 1, 0, 1],  # (0,0)
            [0, 1, 0, 1, 0, 0, 1, 0],  # (0,1)
            [1, 0, 0, 0, 1, 0, 1, 1],  # (1,0)
            [0, 1, 0, 0, 1, 1, 1, 0],  # (1,1)
        ], dtype=float)
        V = np.array([
            [1, 0, 1, 0, 1, 0, 0, 1],  # (0,0)
            [0, 1, 0, 1, 0, 1, 1, 0],  # (0,1)
            [1, 0, 1, 0, 1, 0, 0, 1],  # (1,0)
            [0, 1, 0, 1, 0, 1, 1, 0],  # (1,1)
        ], dtype=float)
        W = np.array([
            [1, 1, 0, 0, 1, 0, 0, 0],  # (0,0)
            [0, 0, 1, 1, 0, 1, 1, 0],  # (0,1)
            [1, 1, 0, 0, 1, 0, 0, 0],  # (1,0)
            [0, 0, 1, 1, 0, 1, 1, 0],  # (1,1)
        ], dtype=float)
        return U, V, W

    # --- Build full U, V, W from core + extension columns ---
    # Extension columns (c >= 8) have zeros in the 2x2x2 sub-block.
    # For each extension column c, define:
    #   U_ext: 4-vector, V_ext: 14-vector, W_ext: 14-vector
    # Constraint: for all i,k,j in {0,1}: sum_c U[i,k,c]*V[k,j,c]*W[j,i,c] = 0
    # This is automatically satisfied if extension columns have support only on j>=2 in V
    # OR only on j>=2 in W OR are structured so that the 2x2x2 contraction vanishes.
    #
    # Simplest valid extension: for each (i,k) pair and j in 2..6, add a schoolbook-like column
    # but we can do better by sharing.
    #
    # Strategy: For j in 2..6, we need to satisfy:
    #   For each (i,k): sum_c U[i,k,c]*V[k,j,c]*W[j,i,c] = 1
    # If we use dedicated columns per (i,k,j) with j>=2, that's 2*2*5 = 20 columns + 8 = 28.
    #
    # Better: For each j in 2..6, use a rank-2 block (since m=2, we need at least 2 columns per j
    # if we want to share across i). Actually for fixed j, we need a 2x2 matrix (i,k) to be 1.
    # The identity matrix 2x2 has rank 2, so we can use 2 columns per j if we allow U and W to vary.
    #
    # Let's try: for each j in 2..6, use 2 columns c1, c2.
    # Column c1: U = [1,0,0,0], V[k,j]=1 for all k, W[j,i]=1 for all i -> gives outer product of [1,1] and [1,0,0,0]^T? No.
    # Let me think more carefully.
    #
    # For fixed j, we need: for all i,k: sum_{c in {c1,c2}} U[i,k,c]*V[k,j,c]*W[j,i,c] = 1
    # Let's set V[k,j,c1] = 1, V[k,j,c2] = 1 for all k.
    # Then we need: for all i,k: U[i,k,c1]*W[j,i,c1] + U[i,k,c2]*W[j,i,c2] = 1
    # Let U[i,k,c1] = delta_{i,0}, U[i,k,c2] = delta_{i,1} (independent of k)
    # Then: for i=0: delta_{k,?}... wait, U depends on both i and k.
    #
    # Simpler: use schoolbook for j>=2 but try to merge columns.
    # For now, let's just do schoolbook and see if we can find a lower rank via the core insight.
    
    # Actually, the key insight from the cell findings: "Correcting the Strassen tensor coefficients 
    # ensures the rank-25 decomposition is algebraically valid"
    # So rank 25 is achievable! Let's construct it explicitly.
    #
    # Rank 25 = 8 (Strassen core) + 17 (extensions for j=2..6)
    # For j in 2..6 (5 values), we need 17 columns. 17/5 = 3.4, so not uniform.
    # Alternatively: 8 + 5*2 = 18 columns for j>=2 if we use 2 per j, total 26.
    # Or: use a smarter extension.
    
    # Let me try a direct approach: assume rank r, and use the structure that
    # the tensor is block-structured in j. For j=0,1 we use Strassen (8 cols).
    # For j=2..6, we need to satisfy 2*2 = 4 equations per j.
    # If we use 2 columns per j, that's 10 more columns, total 18.
    # But maybe we can share columns across different j values.
    
    # Let's try rank 25 with a specific construction.
    # I'll use a semi-structured search: fix the Strassen core, then solve for
    # extension columns using linear algebra.
    
    def try_rank_with_core(r):
        """Try to build a rank-r decomposition using Strassen core + (r-8) extension columns."""
        if r < 8:
            return None
        num_ext = r - 8
        
        # Core
        U_core, V_core, W_core = strassen_core()
        
        # Extension: we have num_ext columns to fill.
        # For each extension column c (0-indexed in extension), we choose:
        #   U_ext[c]: 4-vector
        #   V_ext[c]: 14-vector  
        #   W_ext[c]: 14-vector
        # Subject to: for all i,k,j in {0,1}: sum_c_ext U[i,k,c]*V[k,j,c]*W[j,i,c] = 0
        # And: for all i,k,j with j>=2: sum_c_ext U[i,k,c]*V[k,j,c]*W[j,i,c] = 1
        
        # This is a nonlinear system. Let's use a simpler approach:
        # Assume extension columns have V support only on j>=2 (so core constraint auto-satisfied).
        # Then for j>=2, we need to solve a bilinear system.
        
        # Even simpler: use schoolbook-style extensions but try to merge.
        # For each j in 2..6, use 2 columns (one for i=0, one for i=1) with k shared.
        
        # Let's just do a clean schoolbook and return it. The Strassen insight tells us
        # rank 25 is possible, but constructing it explicitly is complex.
        # For this strategy, let's try a different approach: use the fact that
        # m=2 means each output column j only needs rank 2 in the (i,k) plane.
        
        # Construction: 
        # Columns 0-7: Strassen core (handles j=0,1)
        # Columns 8-17: For j=2,3,4,5,6, use 2 columns each (10 columns)
        # Total: 18 columns. Let's verify this works.
        
        if r >= 18:
            U = np.zeros((4, r))
            V = np.zeros((14, r))
            W = np.zeros((14, r))
            
            # Copy Strassen core
            U[:, :8] = U_core
            V[:, :8] = V_core
            W[:, :8] = W_core
            
            # For j in 2..6, use 2 columns
            col = 8
            for j in range(2, 7):
                # Column col: handles i=0
                U[0*2+0, col] = 1  # (i=0,k=0)
                U[0*2+1, col] = 1  # (i=0,k=1)
                U[1*2+0, col] = 0
                U[1*2+1, col] = 0
                for k in range(2):
                    V[k*2+j, col] = 1
                W[j*2+0, col] = 1  # (j,i=0)
                W[j*2+1, col] = 0
                col += 1
                
                # Column col: handles i=1
                U[0*2+0, col] = 0
                U[0*2+1, col] = 0
                U[1*2+0, col] = 1  # (i=1,k=0)
                U[1*2+1, col] = 1  # (i=1,k=1)
                for k in range(2):
                    V[k*2+j, col] = 1
                W[j*2+0, col] = 0
                W[j*2+1, col] = 1  # (j,i=1)
                col += 1
            
            # Verify
            if verify_fast(U, V, W):
                if is_valid_entries(U, V, W):
                    return {
                        "U": to_list(U),
                        "V": to_list(V),
                        "W": to_list(W)
                    }
        
        return None
    
    # Try ranks from 18 up to 28
    for r in range(18, best_rank + 1):
        if time.time() > deadline:
            break
        
        result = try_rank_with_core(r)
        if result is not None:
            best = result
            best_rank = r
            break
    
    return best
# EVOLVE-BLOCK-END
