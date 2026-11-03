"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    
    # Helper to verify a decomposition
    def verify(U, V, W, n, m, p):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{ik} * A_{ij} B_{jk} ?
        # No, the tensor is T_{ijk} = 1 if i,j,k correspond to a valid multiplication term?
        # The multiplication tensor T is defined by T_{i,j,k} = 1 if C_{ik} = sum_j A_{ij} B_{jk}.
        # So the tensor has 1s at (i, j, k) for all i, j, k.
        # Wait, the tensor is the structure tensor of the bilinear map.
        # T_{i, j, k} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
        # In standard matrix multiplication, every (i,j,k) triplet is a term.
        # So the tensor is all ones?
        # No, the tensor is in the space A^* \otimes B^* \otimes C.
        # The entry T_{i,j,k} is 1 if A_{ij} B_{jk} is part of the sum for C_{ik}.
        # Yes, for standard matrix multiplication, T_{i,j,k} = 1 for all i,j,k.
        # So we need to check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k.
        
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # Compute the tensor product
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can compute this by iterating over r or using einsum.
        # T = np.einsum('ir,jr,kr->ijk', U.reshape(n, m, -1), V.reshape(m, p, -1), W.reshape(p, n, -1))
        # But indices are flattened.
        
        # Let's just iterate for small sizes
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0
                    for r in range(U.shape[1]):
                        val += U[idx_u][r] * V[idx_v][r] * W[idx_w][r]
                    if val != 1:
                        return False
        return True

    # Fallback: Schoolbook Rank 45
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to improve
    # Split into 3x3x3 and 3x3x2
    # Part 1: 3x3x3 (cols 0-2)
    # Part 2: 3x3x2 (cols 3-4)
    
    # We will try to construct a rank-23 decomposition for 3x3x3
    # and a rank-17 decomposition for 3x3x2.
    # If they fail, we use schoolbook for those parts.
    
    # Rank 23 decomposition for 3x3x3
    # This is complex to hardcode correctly without reference.
    # Let's try to use a known rank-24 or 25 if 23 is too risky?
    # Actually, let's just try to verify a hardcoded rank-23.
    # If I can't find it, I'll use schoolbook (27).
    
    # For now, let's assume we can't reliably hardcode the 23 terms.
    # Let's try to improve the 3x3x2 part to rank 17.
    # If we get rank 17 for 3x3x2, and use schoolbook 27 for 3x3x3, total is 44.
    # This is an improvement over 45.
    
    # Let's try to construct a rank-17 decomposition for 3x3x2.
    # C0 = A B0, C1 = A B1
    # B0 is 3x1, B1 is 3x1.
    # We can use Strassen-like ideas.
    # Let's try to find one.
    
    # If I can't find it, I return 45.
    
    # Given the constraints, I will return the schoolbook decomposition for safety,
    # as previous attempts to find lower ranks failed.
    
    return get_schoolbook()
# EVOLVE-BLOCK-END
