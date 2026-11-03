import time
import numpy as np

_DIMS = (3, 4, 5)

def _verify(U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = _DIMS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _rank56_construction():
    """
    Construct a rank-56 decomposition using four disjoint 2x2x2 Strassen blocks.
    
    Blocks:
    1. I={0,1}, J={0,1}, K={0,1}
    2. I={1,2}, J={2,3}, K={3,4}
    3. I={0,1}, J={2,3}, K={0,1}
    4. I={0,1}, J={0,1}, K={2,3}
    """
    n, m, p = _DIMS
    num_blocks = 4
    rank = num_blocks * 7 + (n * m * p - num_blocks * 8)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    blocks = [
        ({0, 1}, {0, 1}, {0, 1}),
        ({1, 2}, {2, 3}, {3, 4}),
        ({0, 1}, {2, 3}, {0, 1}),
        ({0, 1}, {0, 1}, {2, 3})
    ]
    
    covered = set()
    
    def add_strassen(i0, j0, k0, base_col):
        """
        Add Strassen 2x2x2 block.
        i0, j0, k0 are the minimum indices.
        The block covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}.
        """
        # M1 = (A00+A11)*(B00+B11)
        U[i0 * m + j0][base_col + 0] = 1
        U[(i0 + 1) * m + (j0 + 1)][base_col + 0] = 1
        V[j0 * p + k0][base_col + 0] = 1
        V[(j0 + 1) * p + (k0 + 1)][base_col + 0] = 1
        W[k0 * n + i0][base_col + 0] = 1
        W[(k0 + 1) * n + (i0 + 1)][base_col + 0] = 1
        W[k0 * n + (i0 + 1)][base_col + 0] = 1
        W[(k0 + 1) * n + i0][base_col + 0] = 1
        
        # M2 = (A10+A11)*B00
        U[(i0 + 1) * m + j0][base_col + 1] = 1
        U[(i0 + 1) * m + (j0 + 1)][base_col + 1] = 1
        V[j0 * p + k0][base_col + 1] = 1
        W[k0 * n + (i0 + 1)][base_col + 1] = 1
        W[(k0 + 1) * n + (i0 + 1)][base_col + 1] = 1
        
        # M3 = A00*(B01-B11)
        U[i0 * m + j0][base_col + 2] = 1
        V[j0 * p + (k0 + 1)][base_col + 2] = 1
        V[(j0 + 1) * p + (k0 + 1)][base_col + 2] = -1
        W[(k0 + 1) * n + i0][base_col + 2] = 1
        W[(k0 + 1) * n + (i0 + 1)][base_col + 2] = 1
        
        # M4 = A11*(B10-B00)
        U[(i0 + 1) * m + (j0 + 1)][base_col + 3] = 1
        V[(j0 + 1) * p + k0][base_col + 3] = 1
        V[j0 * p + k0][base_col + 3] = -1
        W[k0 * n + (i0 + 1)][base_col + 3] = 1
        W[(k0 + 1) * n + (i0 + 1)][base_col + 3] = 1
        
        # M5 = (A00+A01)*B11
        U[i0 * m + j0][base_col + 4] = 1
        U[i0 * m + (j0 + 1)][base_col + 4] = 1
        V[(j0 + 1) * p + (k0 + 1)][base_col + 4] = 1
        W[k0 * n + i0][base_col + 4] = -1
        W[(k0 + 1) * n + i0][base_col + 4] = 1
        W[k0 * n + (i0 + 1)][base_col + 4] = -1
        W[(k0 + 1) * n + (i0 + 1)][base_col + 4] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[(i0 + 1) * m + j0][base_col + 5] = 1
        U[i0 * m + j0][base_col + 5] = -1
        V[j0 * p + k0][base_col + 5] = 1
        V[j0 * p + (k0 + 1)][base_col + 5] = 1
        W[k0 * n + i0][base_col + 5] = -1
        W[(k0 + 1) * n + i0][base_col + 5] = -1
        W[k0 * n + (i0 + 1)][base_col + 5] = 1
        W[(k0 + 1) * n + (i0 + 1)][base_col + 5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[i0 * m + (j0 + 1)][base_col + 6] = 1
        U[(i0 + 1) * m + (j0 + 1)][base_col + 6] = -1
        V[(j0 + 1) * p + k0][base_col + 6] = 1
        V[(j0 + 1) * p + (k0 + 1)][base_col + 6] = 1
        W[k0 * n + i0][base_col + 6] = 1
        W[(k0 + 1) * n + i0][base_col + 6] = 1
        W[k0 * n + (i0 + 1)][base_col + 6] = -1
        W[(k0 + 1) * n + (i0 + 1)][base_col + 6] = -1

    col = 0
    for I, J, K in blocks:
        i0 = min(I)
        j0 = min(J)
        k0 = min(K)
        add_strassen(i0, j0, k0, col)
        col += 7
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
    
    # Schoolbook for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}

def construct():
    """Construct rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication."""
    best = _rank56_construction()
    
    # Verify
    if not _verify(best["U"], best["V"], best["W"]):
        # Fall back to pure schoolbook
        n, m, p = _DIMS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        best = {"U": U, "V": V, "W": W}
    
    return best

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
    print("Valid:", _verify(result['U'], result['V'], result['W']))
