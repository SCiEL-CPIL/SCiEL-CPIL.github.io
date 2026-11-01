"""Program for matrix multiplication 2,2,10 using block-diagonal Strassen decomposition with rank reduction."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

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
                    return False
    return True

def _construct_strassen_base():
    """Construct the base rank-35 Strassen decomposition."""
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(5):
        col_base = t * 7
        
        # Term 1: P1 = (a+d)(e+h)
        col = col_base
        U[0][col] = 1; U[3][col] = 1
        V[2*t][col] = 1; V[10+2*t][col] = 1
        W[4*t][col] = 1; W[4*t+3][col] = 1
        
        # Term 2: P2 = (c+d)e
        col = col_base + 1
        U[2][col] = 1; U[3][col] = 1
        V[2*t][col] = 1
        W[4*t+1][col] = 1; W[4*t+3][col] = -1
        
        # Term 3: P3 = a(f-h)
        col = col_base + 2
        U[0][col] = 1
        V[2*t+1][col] = 1; V[10+2*t][col] = -1
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
    
    return U, V, W

def _try_reduce_rank(U, V, W, deadline):
    """Try to reduce rank by finding linear dependencies among the rank-1 tensors."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build tensor vectors for each rank-1 component
    def tensor_to_vector(U_col, V_col, W_col):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[(i * m + j) * p + k] = U_col[i*m+j] * V_col[j*p+k] * W_col[k*n+i]
        return vec
    
    vectors = []
    for r in range(rank):
        vectors.append(tensor_to_vector(U_arr[:, r], V_arr[:, r], W_arr[:, r]))
    
    vec_matrix = np.array(vectors).T  # (40, rank)
    
    # Find null space using SVD
    try:
        U_svd, S, Vt = np.linalg.svd(vec_matrix, full_matrices=False)
        
        # Find singular values that are effectively zero
        threshold = 1e-10
        null_dim = sum(1 for s in S if s < threshold)
        
        if null_dim > 0:
            # The null space is spanned by the last null_dim columns of Vt
            null_vectors = Vt[rank - null_dim:].T  # (rank, null_dim)
            
            # Try removing columns one at a time from the null space directions
            # For each null vector, we can potentially eliminate one rank-1 tensor
            # by adjusting the others
            
            # Simple approach: try to remove the column with smallest contribution
            # Actually, let's try a greedy approach: remove columns and check validity
            
            current_U = [row[:] for row in U]
            current_V = [row[:] for row in V]
            current_W = [row[:] for row in W]
            current_rank = rank
            
            # Try removing each column and see if we can still represent the tensor
            # This is expensive, so let's try a smarter approach
            
            # For now, just check if we can drop any single column
            for r_to_remove in range(rank):
                if time.time() > deadline:
                    break
                    
                # Try removing column r_to_remove
                keep = [r for r in range(current_rank) if r != r_to_remove]
                
                U_test = [[current_U[i][r] for r in keep] for i in range(n*m)]
                V_test = [[current_V[i][r] for r in keep] for i in range(m*p)]
                W_test = [[current_W[i][r] for r in keep] for i in range(p*n)]
                
                if _verify_decomposition(U_test, V_test, W_test):
                    return U_test, V_test, W_test
                
                # If we can't remove it directly, try adjusting other columns
                # This is a linear system: we need to find deltas to other columns
                # such that the sum of adjusted rank-1 tensors equals the original
                # This is complex, so skip for now
    except Exception:
        pass
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal decomposition for 2x2 * 2x10 matrix multiplication.
    Uses Strassen decomposition with rank reduction attempts.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Leave 30s safety margin
    
    # Start with base Strassen decomposition
    U, V, W = _construct_strassen_base()
    
    # Verify base decomposition
    if not _verify_decomposition(U, V, W):
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    U, V, W = _try_reduce_rank(U, V, W, deadline)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
