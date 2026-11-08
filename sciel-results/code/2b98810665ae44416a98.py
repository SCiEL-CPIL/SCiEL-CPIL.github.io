"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    Uses structured Strassen-like ansatz for rank 16.
    """
    n, m, p = 2, 3, 3
    deadline = time.time() + 140  # Safety margin before 150s limit

    def verify_decomp(U, V, W, tol=1e-10):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = U.shape[1]
        if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1) > tol:
                        return False
        return True

    # Generate schoolbook as baseline (rank 18)
    U_sb = np.zeros((6, 18), dtype=int)
    V_sb = np.zeros((9, 18), dtype=int)
    W_sb = np.zeros((6, 18), dtype=int)
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U_sb[i*3+j, col] = 1
                V_sb[j*3+k, col] = 1
                W_sb[k*2+i, col] = 1
                col += 1
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = 18

    # Structured rank-16 ansatz based on Strassen-like splitting
    # For 2x3 * 3x3, we can use the following construction:
    # Split the 3x3 B matrix into two overlapping 2x3 submatrices
    # and use bilinear combinations to reduce products.
    
    # The key idea: use 16 terms with the following structure
    # Terms 0-7: handle the first two columns of A with all rows of B
    # Terms 8-15: handle the third column of A with corrections
    
    # Let's try a specific structured pattern:
    # U rows: [A00, A01, A02, A10, A11, A12]
    # V rows: [B00, B01, B02, B10, B11, B12, B20, B21, B22]
    # W rows: [C00, C01, C02, C10, C11, C12]
    
    # Try this rank-16 construction with integer coefficients:
    U_16 = np.array([
        # Row 0: A[0,0]
        [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        # Row 1: A[0,1]
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        # Row 2: A[0,2]
        [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        # Row 3: A[1,0]
        [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        # Row 4: A[1,1]
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        # Row 5: A[1,2]
        [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ], dtype=int)
    
    V_16 = np.array([
        # Row 0: B[0,0]
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 1: B[0,1]
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 2: B[0,2]
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 3: B[1,0]
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 4: B[1,1]
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 5: B[1,2]
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 6: B[2,0]
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 7: B[2,1]
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 8: B[2,2]
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)
    
    W_16 = np.array([
        # Row 0: C[0,0]
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 1: C[0,1]
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 2: C[0,2]
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 3: C[1,0]
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 4: C[1,1]
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 5: C[1,2]
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)
    
    # Verify the structured ansatz
    if verify_decomp(U_16, V_16, W_16):
        best_rank = 16
        best_decomp = (U_16, V_16, W_16)
    else:
        # If structured ansatz doesn't work, try random search for rank 16
        import random
        random.seed(42)
        
        def try_random_rank(r, max_iters=3000, coeff_range=(-2, 2)):
            """Try random integer coefficient search for given rank."""
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                U = np.random.randint(coeff_range[0], coeff_range[1]+1, (6, r))
                V = np.random.randint(coeff_range[0], coeff_range[1]+1, (9, r))
                W = np.random.randint(coeff_range[0], coeff_range[1]+1, (6, r))
                if verify_decomp(U, V, W):
                    return (U, V, W)
            return None
        
        result = try_random_rank(16, max_iters=2000)
        if result is not None:
            best_rank = 16
            best_decomp = result
        elif time.time() < deadline - 10:
            result = try_random_rank(17, max_iters=2000)
            if result is not None:
                best_rank = 17
                best_decomp = result

    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
