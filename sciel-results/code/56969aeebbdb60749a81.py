"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 3x3 @ 3x8.
    Uses the schoolbook decomposition which is guaranteed to be valid.
    Rank = n * m * p = 3 * 3 * 8 = 72.
    """
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U is indexed by (i, j)
                U[i * m + j, idx] = 1.0
                # V is indexed by (j, k)
                V[j * p + k, idx] = 1.0
                # W is indexed by (k, i) following the trace convention W[k*n+i]
                W[k * n + i, idx] = 1.0
                idx += 1
                
    if time.time() > deadline:
        # Fallback if construction takes too long (unlikely for 72 terms)
        pass

    # Convert to lists of lists to match interface specification
    # Ensure integers where possible to satisfy "integer/half-integer entries"
    def to_int_list(mat):
        return [[int(x) if x == int(x) else x for x in row] for row in mat.tolist()]

    return {"U": to_int_list(U), "V": to_int_list(V), "W": to_int_list(W)}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    print(f"Constructed rank {len(result['U'][0])}")
