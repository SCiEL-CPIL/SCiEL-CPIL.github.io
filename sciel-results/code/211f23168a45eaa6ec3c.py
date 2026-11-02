"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6
    dim_u = n * m  # 4
    dim_v = m * p  # 12
    dim_w = p * n  # 12
    
    # Non-zero entries of T:
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero.append((a, b, c))
    
    # Strategy: Use structured ansatz based on the algebraic structure.
    # For 2x2 * 2x6, we have 12 outputs:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # 
    # Key idea: Group the 6 k-values into 3 pairs (k, k+3) for k=0,1,2
    # and use a Strassen-like trick on the 2x2 A matrix.
    #
    # Actually, let's try a direct approach: use the fact that we can write
    # A as a product involving fewer terms.
    #
    # Alternative: Use the "middle" approach where we compute intermediate
    # products M[r] = U[:,r] @ A and N[r] = B @ V[:,r], then C = sum W[r] * M[r] * N[r]
    #
    # Let me try rank 20 with a specific structured pattern:
    # Use 4 "base" terms for the diagonal of A, and 16 for the off-diagonal mixing.
    
    # Let's try a numerical approach: set up the problem as finding rank-r decomposition
    # using a sparse structure.
    
    # Ansatz for rank 20:
    # Columns 0-3: handle the "pure" terms A[i,j]*B[j,k] for j=0,1 and specific k
    # Columns 4-19: handle mixing terms
    
    # Actually, let me try a different structured approach:
    # For each pair of k values (k1, k2), use 2 terms that handle both.
    # 
    # Term for (k1, k2): 
    #   U: [1, 1, 1, 1] (all ones)
    #   V: [0..k1..0, 0..k2..0, 0..6+k1..0, 0..6+k2..0] with specific values
    #   W: [0..2k1..0, 0..2k1+1..0, 0..2k2..0, 0..2k2+1..0] with specific values
    #
    # This is getting complex. Let me just try rank 24 which is guaranteed to work,
    # but also try to find a lower rank.
    
    # For now, use schoolbook rank 24 which is known to work:
    rank = 24
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    # Verify
    valid = True
    nonzero_set = set(nonzero)
    for a in range(dim_u):
        for b in range(dim_v):
            for c in range(dim_w):
                expected = 1 if (a, b, c) in nonzero_set else 0
                actual = 0
                for r in range(rank):
                    actual += U[a, r] * V[b, r] * W[c, r]
                if abs(actual - expected) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        raise RuntimeError("Schoolbook decomposition failed verification")
    
    # Convert to lists of integers/half-integers
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                if abs(val - round(val)) < 1e-10:
                    row_list.append(int(round(val)))
                elif abs(val * 2 - round(val * 2)) < 1e-10:
                    row_list.append(round(val * 2) / 2)
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
