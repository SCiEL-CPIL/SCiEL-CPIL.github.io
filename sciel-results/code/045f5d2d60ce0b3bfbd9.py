"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Build the rank-55 decomposition using Strassen for 2x2x2 blocks
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part A: j=0 (first column of A, first row of B)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2
        
        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1
        
        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1
    
    # Now attempt rank reduction by checking for linear dependencies
    # Convert to numpy arrays
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check if any column is linearly dependent on others
    # For each column r, check if W[:,r] can be expressed as linear combination of other W columns
    # while maintaining the constraint that U[:,r] and V[:,r] also satisfy the same relation
    
    # A better approach: try to find a subset of columns that spans the same tensor
    # Use the fact that if we have columns (U_r, V_r, W_r) for r=1..R,
    # we can eliminate column r if W_r = sum_{s != r} c_s * W_s
    # and U_r = sum_{s != r} c_s * U_s (but this is generally not possible)
    
    # Alternative: use the bilinear structure more carefully
    # The tensor is T = sum_r U_r (outer) V_r (outer) W_r (outer)
    # We can try to find a lower-rank representation by solving a system
    
    # Let's try a greedy approach: remove columns that are "redundant"
    # A column r is redundant if there exist coefficients c_s such that
    # W_r = sum_{s != r} c_s W_s
    # AND the resulting tensor change can be compensated by modifying other columns
    
    # For now, let's just verify the decomposition and return it
    # The key improvement would be to find a true rank reduction
    
    # Check time and return
    if time.time() > deadline - 5:
        return {"U": U, "V": V, "W": W}
    
    # Try to find linear dependencies in the W matrix
    # If W has rank < R, we might be able to reduce the decomposition
    W_rank = np.linalg.matrix_rank(W_arr)
    
    if W_rank < rank:
        # There might be redundancy - try to find a reduced set
        # Use SVD to find a basis
        try:
            U_svd, s, Vt = np.linalg.svd(W_arr, full_matrices=False)
            # Keep only significant singular values
            tol = 1e-10
            num_kept = np.sum(s > tol)
            if num_kept < rank:
                # We have redundancy, but we need to ensure the full tensor is preserved
                # This requires a more careful analysis
                pass
        except:
            pass
    
    # Return the rank-55 decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
