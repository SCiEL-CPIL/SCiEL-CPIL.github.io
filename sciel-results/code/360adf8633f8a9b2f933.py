"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    # We'll try to find a decomposition with rank < n*m*p
    # Start with the schoolbook rank as fallback
    rank_full = n * m * p
    best_U, best_V, best_W = None, None, None

    # Generate the full multiplication tensor
    # T[i, j, k] = 1 if we multiply element (i,j) of A by (j,k) of B
    # The tensor has shape (n, m, p) and all entries are 1
    # Actually, the multiplication tensor T has T[i,j,k] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # For matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] which is 1 for all valid i,j,k
    
    # The rank of the multiplication tensor for n x m by m x p is at most n*m*p
    # We want to find if there's a lower rank decomposition
    
    # For small matrices, we can try to use known results or search
    # The Strassen algorithm gives rank 7 for 2x2x2
    # For 3x5x6, the known upper bound might be less than 90
    
    # Let's try a simple approach: use the structure of the problem
    # Actually, for this specific case, let's just return the schoolbook decomposition
    # unless we can quickly find something better
    
    # Try rank = n*m*p - 1, n*m*p - 2, etc. using a simple heuristic
    # But finding exact low-rank tensor decompositions is NP-hard in general
    
    # For now, return the schoolbook decomposition
    rank = rank_full
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
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
