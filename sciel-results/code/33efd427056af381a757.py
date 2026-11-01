"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                if abs(computed - expected) > 1e-10:
                    return False, (i, j, k, computed, expected)
    return True, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Start with rank-35 Strassen decomposition
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(5):
        col_base = t * 7
        
        # Term 1: P1 = (a+d)(e+h)
        col = col_base
        U[0][col] = 1; U[3][col] = 1
        V[2*t][col] = 1; V[11+2*t][col] = 1
        W[4*t][col] = 1; W[4*t+3][col] = 1
        
        # Term 2: P2 = (c+d)e
        col = col_base + 1
        U[2][col] = 1; U[3][col] = 1
        V[2*t][col] = 1
        W[4*t+1][col] = 1; W[4*t+3][col] = -1
        
        # Term 3: P3 = a(f-h)
        col = col_base + 2
        U[0][col] = 1
        V[2*t+1][col] = 1; V[11+2*t][col] = -1
        W[4*t+2][col] = 1; W[4*t+3][col] = 1
        
        # Term 4: P4 = d(g-e)
        col = col_base + 3
        U[3][col] = 1
        V[10+2*t][col] = 1; V[2*t][col] = -1
        W[4*t][col] = 1; W[4*t+1][col] = 1
        
        # Term 5: P5 = (a+b)h
        col = col_base + 4
        U[0][col] = 1; U[1][col] = 1
        V[11+2*t][col] = 1
        W[4*t][col] = -1; W[4*t+2][col] = 1
        
        # Term 6: P6 = (c-a)(e+f)
        col = col_base + 5
        U[2][col] = 1; U[0][col] = -1
        V[2*t][col] = 1; V[2*t+1][col] = 1
        W[4*t+3][col] = 1
        
        # Term 7: P7 = (b-d)(g+h)
        col = col_base + 6
        U[1][col] = 1; U[3][col] = -1
        V[10+2*t][col] = 1; V[11+2*t][col] = 1
        W[4*t][col] = 1
    
    # Verify base decomposition
    valid, err = _verify_decomposition(U, V, W)
    if not valid:
        # Return schoolbook if Strassen fails
        rank = 40
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m + j][r] = 1
                    V[j*p + k][r] = 1
                    W[k*n + i][r] = 1
        return {"U": U, "V": W}
    
    # Try to reduce rank by finding linear dependencies
    # Strategy: For each pair of blocks, check if we can merge terms
    # by finding coefficients that make their tensor sum zero
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor representation for dependency checking
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We want to find c_r such that sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0
    
    # Flatten each rank-1 tensor into a vector
    def tensor_to_vector(U_col, V_col, W_col):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[(i * m + j) * p + k] = U_col[i*m+j] * V_col[j*p+k] * W_col[k*n+i]
        return vec
    
    # Check for linear dependencies among the 35 tensors
    # This is a 40-dim space (2*2*10), and we have 35 vectors
    # If they're linearly dependent, we can reduce rank
    
    vectors = []
    for r in range(rank):
        vectors.append(tensor_to_vector(U_arr[:, r], V_arr[:, r], W_arr[:, r]))
    
    vec_matrix = np.array(vectors).T  # (40, 35)
    
    # Find null space
    try:
        Q, R = np.linalg.qr(vec_matrix)
        # Check for near-zero diagonal entries in R
        diag = np.diag(R)
        null_indices = [i for i in range(rank) if abs(diag[i]) < 1e-8]
        
        if null_indices:
            # We found linear dependencies - can remove those columns
            # Keep track of which columns to remove
            remove_set = set(null_indices)
            new_rank = rank - len(remove_set)
            keep_indices = [r for r in range(rank) if r not in remove_set]
            
            U_new = [U[r] for r in keep_indices]  # This is wrong, need to transpose
            U_new = [[U[i][r] for r in keep_indices] for i in range(n*m)]
            V_new = [[V[i][r] for r in keep_indices] for i in range(m*p)]
            W_new = [[W[i][r] for r in keep_indices] for i in range(p*n)]
            
            valid, err = _verify_decomposition(U_new, V_new, W_new)
            if valid:
                return {"U": U_new, "V": V_new, "W": W_new}
    except Exception:
        pass
    
    # Try a more systematic approach: greedy rank reduction
    # Try removing one column at a time and checking if we can adjust others
    
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    current_rank = rank
    
    for remove_r in range(current_rank):
        if time.time() > deadline:
            break
            
        # Try removing column remove_r
        test_U = [row[:] for row in current_U]
        test_V = [row[:] for row in current_V]
        test_W = [row[:] for row in current_W]
        
        # Remove the column
        for i in range(n*m):
            test_U[i].pop(remove_r)
        for i in range(m*p):
            test_V[i].pop(remove_r)
        for i in range(p*n):
            test_W[i].pop(remove_r)
        
        # Check if the remaining columns can still decompose the tensor
        # by solving for the missing contribution
        # This is complex; instead, just verify if the reduced set works
        valid, err = _verify_decomposition(test_U, test_V, test_W)
        if valid:
            current_U = test_U
            current_V = test_V
            current_W = test_W
            current_rank -= 1
    
    if current_rank < rank:
        return {"U": current_U, "V": current_V, "W": current_W}
    
    # If no reduction found, return the original
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
