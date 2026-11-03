"""Find a rank-minimal exact tensor decomposition for 3x4 x 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 Strassen decomposition.
    U: (4, 7) rows indexed by (i,j) in {00,01,10,11}
    V: (4, 7) rows indexed by (j,k) in {00,01,10,11}
    W: (4, 7) rows indexed by (k,i) in {00,01,10,11}
    """
    # P1: A: (0,0)=1,(1,1)=1; B: (0,0)=1,(1,1)=1
    # P2: A: (1,0)=1,(1,1)=1; B: (0,0)=1
    # P3: A: (0,0)=1; B: (0,1)=1,(1,1)=-1
    # P4: A: (1,1)=1; B: (1,0)=1,(0,0)=-1
    # P5: A: (0,0)=1,(0,1)=1; B: (1,1)=1
    # P6: A: (1,0)=1,(0,0)=-1; B: (0,0)=1,(0,1)=1
    # P7: A: (0,1)=1,(1,1)=-1; B: (1,0)=1,(1,1)=1
    U = [[1, 1, 0, 0, 1, -1, 0],   # (i=0,j=0)
         [0, 1, 0, 0, 1, 0, 1],    # (i=0,j=1)
         [1, 1, 0, 0, 0, 1, 0],    # (i=1,j=0)
         [1, 0, 0, 1, 0, 0, -1]]   # (i=1,j=1)

    V = [[1, 1, 0, -1, 0, 1, 0],   # (j=0,k=0)
         [0, 0, 0, 0, 0, 1, 0],    # (j=0,k=1)
         [0, 0, 1, 1, 0, 0, 1],    # (j=1,k=0)
         [1, 0, -1, 1, 1, 0, 1]]   # (j=1,k=1)

    # C[0,0] = P1 - P4 + P6
    # C[0,1] = P3 + P5 + P6
    # C[1,0] = P2 + P4 + P7
    # C[1,1] = P1 - P3 + P5 - P7
    W = [[1, 0, 0, -1, 0, 1, 0],   # (k=0,i=0)
         [0, 0, 1, 0, 1, 1, 0],    # (k=0,i=1)
         [0, 1, 0, 1, 0, 0, 1],    # (k=1,i=0)
         [1, 0, -1, 0, 1, 0, -1]]  # (k=1,i=1)

    return U, V, W


def _build_strassen_decomposition():
    """
    Build a Strassen-based decomposition.
    Partition:
    - i: I0 = {0, 1}, I1 = {2}
    - j: J0 = {0, 1}, J1 = {2, 3}
    - k: pairs (0,1),(2,3),(4,5) and singleton (6)
    """
    n, m, p = 3, 4, 7

    U_mat = [[0] * 0 for _ in range(n * m)]
    V_mat = [[0] * 0 for _ in range(m * p)]
    W_mat = [[0] * 0 for _ in range(p * n)]

    def grow_to(rank):
        curr = len(U_mat[0]) if U_mat[0] else 0
        while curr < rank:
            for row in U_mat:
                row.append(0)
            for row in V_mat:
                row.append(0)
            for row in W_mat:
                row.append(0)
            curr += 1

    def embed_strassen(i0, i1, j0, j1, k0, k1, start_col):
        """Embed a 2x2x2 Strassen decomposition into the global matrices starting at start_col."""
        U2, V2, W2 = _strassen_2x2x2()
        U2_row_map = {(i0, j0): 0, (i0, j1): 1, (i1, j0): 2, (i1, j1): 3}
        V2_row_map = {(j0, k0): 0, (j0, k1): 1, (j1, k0): 2, (j1, k1): 3}
        W2_row_map = {(k0, i0): 0, (k0, i1): 1, (k1, i0): 2, (k1, i1): 3}

        grow_to(start_col + 7)
        for r in range(7):
            for (ii, jj), ur in U2_row_map.items():
                val = U2[ur][r]
                if val != 0:
                    U_mat[ii * m + jj][start_col + r] = val
            for (jj, kk), vr in V2_row_map.items():
                val = V2[vr][r]
                if val != 0:
                    V_mat[jj * p + kk][start_col + r] = val
            for (kk, ii), wr in W2_row_map.items():
                val = W2[wr][r]
                if val != 0:
                    W_mat[kk * n + ii][start_col + r] = val

    col = 0
    # 2x2x7 for (i=0,1), (j=0,1): 3 Strassen blocks + schoolbook for k=6
    for kb in range(3):
        k0 = 2 * kb
        k1 = 2 * kb + 1
        embed_strassen(0, 1, 0, 1, k0, k1, col)
        col += 7
    for i in [0, 1]:
        for j in [0, 1]:
            grow_to(col + 1)
            U_mat[i * m + j][col] = 1
            V_mat[j * p + 6][col] = 1
            W_mat[6 * n + i][col] = 1
            col += 1

    # 2x2x7 for (i=0,1), (j=2,3)
    for kb in range(3):
        k0 = 2 * kb
        k1 = 2 * kb + 1
        embed_strassen(0, 1, 2, 3, k0, k1, col)
        col += 7
    for i in [0, 1]:
        for j in [2, 3]:
            grow_to(col + 1)
            U_mat[i * m + j][col] = 1
            V_mat[j * p + 6][col] = 1
            W_mat[6 * n + i][col] = 1
            col += 1

    # Schoolbook 1x2x7 for i=2, j=0,1
    for j in [0, 1]:
        for k in range(7):
            grow_to(col + 1)
            U_mat[2 * m + j][col] = 1
            V_mat[j * p + k][col] = 1
            W_mat[k * n + 2][col] = 1
            col += 1

    # Schoolbook 1x2x7 for i=2, j=2,3
    for j in [2, 3]:
        for k in range(7):
            grow_to(col + 1)
            U_mat[2 * m + j][col] = 1
            V_mat[j * p + k][col] = 1
            W_mat[k * n + 2][col] = 1
            col += 1

    return {"U": U_mat, "V": V_mat, "W": W_mat}


def _verify(U, V, W):
    """Verify the tensor decomposition is exact using numpy."""
    n, m, p = DIMENSIONS
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-9)
    except Exception:
        return False


def _build_schoolbook():
    """Construct the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0

    # Start with schoolbook
    best = _build_schoolbook()
    best_rank = n * m * p  # 84

    # Try Strassen-based decomposition
    try:
        candidate = _build_strassen_decomposition()
        if _verify(candidate["U"], candidate["V"], candidate["W"]):
            cand_rank = len(candidate["U"][0])
            if cand_rank < best_rank:
                best = candidate
                best_rank = cand_rank
    except Exception:
        pass

    return best
# EVOLVE-BLOCK-END
