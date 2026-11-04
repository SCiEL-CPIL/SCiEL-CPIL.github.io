"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition has rank n*m*p
    rank = n * m * p
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Construct the schoolbook decomposition
    # Index r corresponds to tuple (i, k, j)
    # U[i*m + k, r] = 1 if r maps to (i, k, j)
    # V[k*p + j, r] = 1 if r maps to (i, k, j)
    # W[j*n + i, r] = 1 if r maps to (i, k, j)
    # Note: The problem statement says W[k*n+i, r] is the output factor?
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # Usually, matrix multiplication C = A * B is defined as C[i, j] = sum_k A[i, k] B[k, j].
    # The tensor is T[i, k, j].
    # U is for A (n x m), V is for B (m x p), W is for C (p x n)?
    # Wait, standard matrix mult: A (n x m) * B (m x p) = C (n x p).
    # The task says: "multiplying a 5 by 5 matrix by a 5 by 5 matrix".
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the output matrix C has indices i, k? Or is it a specific ordering?
    # Let's check the shapes:
    # U: (n*m, rank) -> Input A (n x m)
    # V: (m*p, rank) -> Input B (m x p)
    # W: (p*n, rank) -> Output C (p x n)?
    # If C is p x n, then C = B * A?
    # Let's assume standard C = A * B where A is n x m, B is m x p, C is n x p.
    # But W is (p*n, rank). This suggests the output matrix is stored in a p x n grid.
    # If the output is C (n x p), flattening it as row-major gives index i*p + j.
    # If the output is C^T (p x n), flattening gives j*n + i.
    # The convention "W[k*n+i, r]" suggests the row index is k and column index is i.
    # This matches a matrix of size p x n (rows k, cols i).
    # So the operation is likely C = B * A?
    # (m x p) * (n x m) is not compatible.
    # (m x p) * (p x n) -> (m x n).
    # (n x m) * (m x p) -> (n x p).
    # If W is (p*n), it corresponds to a p x n matrix.
    # Let's assume the multiplication is A (n x m) and B (m x p) resulting in C (n x p).
    # If the evaluator expects W to correspond to C^T (p x n), then W[j*n + i] corresponds to C[i, j].
    # Let's stick to the indices:
    # A[i, k] contributes to U[i*m + k, r]
    # B[k, j] contributes to V[k*p + j, r]
    # C[i, j] contributes to W[?].
    # If W is (p*n), and the convention is W[k*n+i], then row k, col i.
    # This looks like C^T if C is n x p.
    # So W[j*n + i, r] should be 1 for the term A[i, k] B[k, j].
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                r = i * (m * p) + k * p + j
                U[i * m + k][r] = 1.0
                V[k * p + j][r] = 1.0
                # Output index for C[i, j] in a p x n matrix (transposed) is j*n + i
                W[j * n + i][r] = 1.0
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
