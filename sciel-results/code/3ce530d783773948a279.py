"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal (currently assumed schoolbook) exact tensor decomposition
    for multiplying a 2 by 4 matrix by a 4 by 7 matrix.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
              U shape: (n*m, rank)
              V shape: (m*p, rank)
              W shape: (p*n, rank)
              where rank = n*m*p for the schoolbook algorithm.
    """
    n, m, p = DIMENSIONS
    # Schoolbook rank
    r = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per interface requirement "integer/half-integer entries"
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j (row of first matrix)
                U[i * m + j][col] = 1
                # V index: j*p + k (col of first / row of second)
                V[j * p + k][col] = 1
                # W index: k*n + i (col of second, row of result) 
                # Convention: W[k*n + i, r] corresponds to C[i, k]
                # The tensor entry T[i, j, k] is multiplied by the product of factors.
                # Standard decomposition: C[i, k] = sum_j A[i, j] B[j, k]
                # Term in tensor: A[i, j] * B[j, k] * delta[i, k]? No, the tensor is T_{i,j,k}.
                # The decomposition is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ?
                # Let's check the index mapping in the prompt: "output factor follows the official trace convention W[k*n+i,r]"
                # This implies W is indexed by [k, i] flattened as k*n + i.
                # So for a specific (i, j, k), we need:
                # U[i*m + j, r]
                # V[j*p + k, r]
                # W[k*n + i, r]
                # And the sum over r of U*V*W should reconstruct the multiplication.
                # For schoolbook, each (i, j, k) is a unique term.
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
