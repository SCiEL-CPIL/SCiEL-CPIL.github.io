"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix.
    
    The matrix multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    where A is n x m (3x3), B is m x p (3x4), C is n x p (3x4).
    
    Wait, the task says: "multiplying a 3 by 3 matrix by a 3 by 4 matrix"
    So A is 3x3 (n=3, m=3), B is 3x4 (m=3, p=4).
    Output C is 3x4 (n=3, p=4).
    
    The tensor indices: T[i, j, k] = 1 if A[i,j] * B[j,k] contributes to C[i,k].
    U has shape (n*m, rank) = (9, rank)
    V has shape (m*p, rank) = (12, rank)
    W has shape (p*n, rank) = (12, rank)
    
    The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Schoolbook: rank = n*m*p = 3*3*4 = 36.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin
    
    # Schoolbook decomposition (rank 36)
    def schoolbook_decomp():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Validate a decomposition
    def validate(U, V, W, tol=1e-10):
        """Check that U @ V @ W reconstructs the matrix multiplication tensor."""
        rank = U.shape[1]
        # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Compare with expected: T[i,j,k] = 1 if this term contributes (i.e., always 1 for valid (i,j,k))
        # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
        # since every product A[i,j]*B[j,k] contributes to C[i,k].
        
        # More precisely: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # Because C[i,k] = sum_j A[i,j] * B[j,k], and each term is a separate tensor factor.
        
        # Let's verify: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1
        # Wait no. The tensor T is defined such that the bilinear form is:
        # C = A * B, where (AB)[i,k] = sum_j A[i,j] B[j,k]
        # As a tensor: T is a 3-tensor with T[i,j,k] = 1, and the computation is
        # sum_{i,j,k} A[i,j] B[j,k] T[i,j,k] e_{i,k}
        # So T[i,j,k] = 1 for all valid indices.
        
        # The decomposition means: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > max_err:
                        max_err = abs(val - 1.0)
                    if max_err > tol:
                        return False
        return True
    
    # Try to find a lower rank decomposition
    # For 3x3x3, Lader found rank 23 (improved over schoolbook 27).
    # For 3x3x4, we can try splitting: use a rank-r decomposition for 3x3x3 plus schoolbook for 3x3x1.
    # But the dimensions are fixed: n=3, m=3, p=4.
    
    # Alternative: try random search for low-rank integer/half-integer decompositions.
    # This is very hard in general.
    
    # Let's try a structured approach:
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # One approach: use the fact that this is a "flat" tensor.
    # Another: try rank 30-35 with random integer entries.
    
    # Given the time constraint and difficulty, let's first return the schoolbook.
    # Then try a simple optimization: can we merge some terms?
    
    # Actually, let's try a different structured decomposition.
    # For p=4, we can think of it as two copies of 3x3x2.
    # But that doesn't directly help unless we find a good 3x3x2 decomposition.
    
    # Schoolbook for 3x3x2 has rank 18. Two copies give rank 36 total, same as direct.
    
    # Let me try: is there a known result for 3x3x4?
    # I don't recall one. Let's try a random search with a small number of trials.
    
    def try_random_decomposition(rank, num_trials=50, seed=42):
        """Try to find a rank-r decomposition with random integer/half-integer entries."""
        rng = np.random.RandomState(seed)
        
        for trial in range(num_trials):
            # Use small integers: -2, -1, 0, 1, 2 or half-integers
            U = rng.randint(-2, 3, size=(n*m, rank))
            V = rng.randint(-2, 3, size=(m*p, rank))
            W = rng.randint(-2, 3, size=(p*n, rank))
            
            # Check
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(val - 1.0) > 1e-10:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return U, V, W
        return None
    
    # Try ranks from 30 to 35
    for r in range(30, 36):
        if time.time() > deadline - 30:
            break
        result = try_random_decomposition(r, num_trials=20)
        if result is not None:
            U, V, W = result
            if validate(U, V, W):
                return U.tolist(), V.tolist(), W.tolist()
    
    # Fall back to schoolbook
    U, V, W = schoolbook_decomp()
    if validate(U, V, W):
        return U.tolist(), V.tolist(), W.tolist()
    
    # Should never reach here
    return U.tolist(), V.tolist(), W.tolist()
# EVOLVE-BLOCK-END
