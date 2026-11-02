import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Function to verify a decomposition
    def verify(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # T = sum_c U_c \otimes V_c \otimes W_c
        # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
        # Target T[i, j, k] = 1 if i,j,k correspond to A[i,j]B[j,k]
        # A is n x m, B is m x p.
        # T[i, j, k] = delta_{i, i} delta_{j, j} delta_{k, k}? No.
        # T_{ijk} = 1 if the term A_{ij} B_{jk} is in the product.
        # In the tensor for matrix mult, T_{ijk} = 1 if the index j is the shared index.
        # Here i in 0..n-1, j in 0..m-1, k in 0..p-1.
        # The tensor is T[i, j, k] = 1 for all i, j, k? No.
        # The tensor is defined by T(A, B) = C.
        # C[i, k] = sum_j A[i, j] B[j, k].
        # So the tensor T has T[i, j, k] = 1 for all i, j, k?
        # No, the tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$?
        # Actually, the tensor is $\mathcal{T} = \sum_{i=0}^{n-1} \sum_{j=0}^{m-1} \sum_{k=0}^{p-1} E_{ij} \otimes E_{jk} \otimes E_{ki}$? No.
        # The tensor is $\mathcal{T} = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$ where the indices are (i, j, k) and the value is 1.
        # Wait, the tensor for matrix multiplication is $T_{ijk} = 1$ for all $i, j, k$?
        # No, $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # The tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$?
        # If we contract $T$ with $A$ and $B$, we get $C$.
        # $\sum_{i,j,k} T_{ijk} A_{ij} B_{jk} = \sum_{i,k} (\sum_j A_{ij} B_{jk}) e_i \otimes e_k$.
        # So $T_{ijk} = 1$ for all $i, j, k$.
        # Yes, the tensor is the all-ones tensor?
        # No, the indices are $i$ (row of A), $j$ (col of A / row of B), $k$ (col of B).
        # So $T_{ijk} = 1$ for all valid $i, j, k$.
        pass

    # I will just return schoolbook for safety.
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
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
