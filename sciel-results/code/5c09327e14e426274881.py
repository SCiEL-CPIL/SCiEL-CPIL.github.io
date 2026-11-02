"""Tensor decomposition for 2x4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta(i,i')*delta(j,j')*delta(k,k')."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    T = np.zeros((n, m, p), dtype=object)
    for c in range(rank):
        uc = np.array([U[i*m+j][c] for i in range(n) for j in range(m)], dtype=object)
        vc = np.array([V[j*p+k][c] for j in range(m) for k in range(p)], dtype=object)
        wc = np.array([W[k*n+i][c] for k in range(p) for i in range(n)], dtype=object)
        # outer product reshaped
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
    # Expected: T[i,j,k] = 1 if this is the identity contraction, i.e., the multiplication tensor
    # The multiplication tensor T_{ijk} = 1 for all (i,j,k) in the standard basis? No.
    # Actually the tensor for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i,j,k] != 1:
                    return False
    return True

def _strassen_222():
    """Return U,V,W for 2x2x2 Strassen decomposition (rank 7).
    U shape (4,7), V shape (4,7), W shape (4,7).
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]."""
    # Strassen: 
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # M1: A11+A22 -> B11+B22
    # A11=(0,0), A22=(1,1), B11=(0,0), B22=(1,1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C11 gets M1
    # M2: (A21+A22) -> B11
    # A21=(1,0), A22=(1,1), B11=(0,0)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1  # C21 gets M2
    # M3: A11 -> (B21-B22)
    # A11=(0,0), B21=(1,0), B22=(1,1)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1  # C12 gets M3
    # M4: A22 -> (B12-B11)
    # A22=(1,1), B12=(0,1), B11=(0,0)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C11 gets M4
    # M5: (A11+A12) -> B22
    # A11=(0,0), A12=(0,1), B22=(1,1)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # C11 gets -M5
    # M6: (A21-A11) -> (B11+B12)
    # A21=(1,0), A11=(0,0), B11=(0,0), B12=(0,1)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = 1  # C22 gets M6
    # M7: (A12-A22) -> (B21+B22)
    # A12=(0,1), A22=(1,1), B21=(1,0), B22=(1,1)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C11 gets M7
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """Schoolbook decomposition for n x m times m x p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _combine_blocks(blocks):
    """Combine a list of (U,V,W, i_off, j_off, k_off) block decompositions.
    Each block handles subproblem A[i_off:i_off+nb, j_off:j_off+mb] * B[j_off:j_off+mb, k_off:k_off+pb]
    contributing to C[i_off:i_off+nb, k_off:k_off+pb].
    U shape (nb*mb, rank_b), V shape (mb*pb, rank_b), W shape (pb*nb, rank_b).
    Global: U_full[i*m+j, c], V_full[j*p+k, c], W_full[k*n+i, c].
    """
    n, m, p = DIMENSIONS
    total_rank = sum(len(b[0][0]) for b in blocks)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    col = 0
    for (Ub, Vb, Wb, i_off, j_off, k_off) in blocks:
        nb = len(Ub) // len(Ub[0]) if False else None
        # Determine block dimensions from the offsets and shapes
        # Ub has nb*mb rows, Vb has mb*pb rows, Wb has pb*nb rows
        rank_b = len(Ub[0])
        # We need to figure out nb, mb, pb
        # From the block definition, we know the dimensions
        # Let's pass them explicitly
        pass
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Strategy: Split the 4 columns of A (and 4 rows of B) into blocks.
    # Split j dimension (size 4) into two blocks of size 2.
    # For each j-block of size 2, we have a 2x2 * 2x7 subproblem (if we keep k as 7).
    # Actually: A is 2x4, B is 4x7. C is 2x7.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # Split j into j in {0,1} and j in {2,3}.
    # Block 1: A[:, 0:2] (2x2) * B[0:2, :] (2x7) -> 2x7
    # Block 2: A[:, 2:4] (2x2) * B[2:4, :] (2x7) -> 2x7
    # For each block, we have a 2x2 * 2x7 problem.
    # Split k (size 7) into k in {0,1}, {2,3}, {4,5}, {6}.
    # So for each j-block, we have sub-subproblems:
    #   2x2 * 2x2 (Strassen, rank 7) for k-pairs
    #   2x2 * 2x1 (schoolbook, rank 4) for k=6
    
    # Let's build this:
    # j-blocks: [0,1], [2,3]
    # k-blocks: [0,1], [2,3], [4,5], [6]
    
    blocks = []
    j_blocks = [(0, 2), (2, 4)]
    k_blocks = [(0, 2), (2, 4), (4, 6), (6, 7)]
    
    for (j0, j1) in j_blocks:
        mb = j1 - j0  # 2
        for (k0, k1) in k_blocks:
            pb = k1 - k0  # 2 or 1
            nb = n  # 2
            if mb == 2 and pb == 2:
                Ub, Vb, Wb = _strassen_222()
            elif mb == 2 and pb == 1:
                Ub, Vb, Wb = _schoolbook_nm_p(nb, mb, pb)
            elif mb == 1 and pb == 2:
                Ub, Vb, Wb = _schoolbook_nm_p(nb, mb, pb)
            else:
                Ub, Vb, Wb = _schoolbook_nm_p(nb, mb, pb)
            blocks.append((Ub, Vb, Wb, nb, mb, pb, j0, k0))
    
    # Now assemble the global U, V, W
    total_rank = sum(len(b[0][0]) for b in blocks)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    for (Ub, Vb, Wb, nb, mb, pb, j0, k0) in blocks:
        rank_b = len(Ub[0])
        for i in range(nb):
            for j in range(mb):
                for c in range(rank_b):
                    U[i*m + (j0+j)][col+c] = Ub[i*mb+j][c]
        for j in range(mb):
            for k in range(pb):
                for c in range(rank_b):
                    V[(j0+j)*p + (k0+k)][col+c] = Vb[j*pb+k][c]
        for k in range(pb):
            for i in range(nb):
                for c in range(rank_b):
                    W[(k0+k)*n + i][col+c] = Wb[k*nb+i][c]
        col += rank_b
    
    # Verify
    if time.time() < deadline:
        if not _verify(U, V, W, n, m, p):
            # Fallback to schoolbook
            U, V, W = _schoolbook_nm_p(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
