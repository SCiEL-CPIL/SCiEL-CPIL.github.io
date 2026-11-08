"""Tensor decomposition for 3x3 * 3x7 matrix multiplication."""
import time
import sys

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition for matrix multiplication.
    Uses the schoolbook algorithm which has rank n*m*p.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
    # The tensor is T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For schoolbook, each (i, j, k) term is isolated in one column.
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    # Simple check to ensure it runs within time limits and returns correctly
    start_time = time.time()
    res = construct()
    end_time = time.time()
    if end_time - start_time > 110:
        print("Warning: Construction took too long", file=sys.stderr)
    # Verify dimensions
    n, m, p = DIMENSIONS
    assert len(res['U']) == n*m
    assert len(res['V']) == m*p
    assert len(res['W']) == p*n
    rank = len(res['U'][0])
    assert len(res['V'][0]) == rank
    assert len(res['W'][0]) == rank
    print(f"Rank: {rank}")
