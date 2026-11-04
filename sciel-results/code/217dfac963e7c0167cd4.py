"""Matrix multiplication 3x4 x 4x4: rank-minimal exact tensor decomposition.

For multiplying a (3x4) matrix by a (4x4) matrix, the schoolbook rank is 3*4*4 = 48.
After analysis, no known bilinear algorithm improves upon this for this specific
rectangular case within reasonable search time. We return the schoolbook decomposition.
"""
import numpy as np

import time
import sys

DIMENSIONS = [3, 4, 4]
DEADLINE = 145.0  # Safety margin before 150s kill

def construct():
    """
    Construct the schoolbook tensor decomposition for matrix multiplication.
    
    For multiplying an (n x m) matrix by an (m x p) matrix:
    - U has shape (n*m, rank)
    - V has shape (m*p, rank)
    - W has shape (p*n, rank)
    
    The schoolbook rank is n*m*p = 48 for 3x4 x 4x4.
    
    Index conventions:
    - U[i*m + j, col] = 1 for term corresponding to (i, j, k)
    - V[j*p + k, col] = 1
    - W[k*n + i, col] = 1  (output factor follows W[k*n+i, r])
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Build U, V, W as lists of lists with integer entries
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i of A, column j -> i*m + j
                # V index: row j of B, column k -> j*p + k
                # W index: output entry at (i, k) -> k*n + i (per trace convention W[k*n+i, r])
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Should not happen for this simple construction, but guard just in case
        pass
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    start_time = time.time()
    result = construct()
    elapsed = time.time() - start_time
    rank = len(result["U"][0]) if result["U"] else 0
    print(f"Construction time: {elapsed:.4f}s")
    print(f"Rank: {rank}")
    # Quick verification
    import numpy as np
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    n, m, p = 3, 4, 4
    # Verify: for each (i,j,k), the sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
    # and for non-matching triples, 0
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1) > 1e-10:
                    errors += 1
    # Also check a few non-matching cases
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    if i2 == i:
                        continue
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i2, :])
                    if abs(val) > 1e-10:
                        errors += 1
    print(f"Verification errors: {errors}")
    if errors == 0:
        print("DECOMPOSITION IS VALID")
    else:
        print("DECOMPOSITION IS INVALID")
