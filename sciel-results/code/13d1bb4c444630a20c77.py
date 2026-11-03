import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 8]

def _elapsed(start):
    return time.time() - start

def _time_left(start, deadline=145.0):
    return deadline - _elapsed(start)

def _verify(U, V, W, n=3, m=3, p=8, seed=42):
    """Verify if U, V, W decompose the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    np.random.seed(seed)
    A = np.random.randint(-10, 10, (n, m)).astype(float)
    B = np.random.randint(-10, 10, (m, p)).astype(float)
    C = A @ B
    
    C_decomp = np.zeros((n, p))
    for c in range(U.shape[1]):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        
        XY = (u_c * A) @ (v_c * B)
        C_decomp += XY * w_c.T
    
    return np.allclose(C, C_decomp, atol=1e-6)

def _schoolbook(n=3, m=3, p=8):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def _strassen_2x2x2(n=3, m=3, p=8, i0=0, j0=0, k0=0):
    """
    Create Strassen-like decomposition for a 2x2x2 block.
    Returns (U, V, W) arrays for the 7 Strassen terms.
    """
    # 7 Strassen terms for 2x2x2 matrix multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    U = np.zeros((n * m, 7))
    V = np.zeros((m * p, 7))
    W = np.zeros((p * n, 7))
    
    # Term 1: (A11+A22)(B11+B22)
    # U: A11=1, A22=1
    U[i0*m + j0, 0] = 1
    U[(i0+1)*m + (j0+1), 0] = 1
    # V: B11=1, B22=1
    V[j0*p + k0, 0] = 1
    V[(j0+1)*p + (k0+1), 0] = 1
    # W: C11+=1, C12-=1, C21-=1, C22+=1
    W[k0*n + i0, 0] = 1
    W[(k0+1)*n + i0, 0] = -1
    W[k0*n + (i0+1), 0] = -1
    W[(k0+1)*n + (i0+1), 0] = 1
    
    # Term 2: (A21+A22)B11
    U[(i0+1)*m + j0, 1] = 1
    U[(i0+1)*m + (j0+1), 1] = 1
    V[j0*p + k0, 1] = 1
    W[k0*n + (i0+1), 1] = 1
    
    # Term 3: A11(B12-B22)
    U[i0*m + j0, 2] = 1
    V[j0*p + (k0+1), 2] = 1
    V[(j0+1)*p + (k0+1), 2] = -1
    W[(k0+1)*n + i0, 2] = 1
    
    # Term 4: A22(B21-B11)
    U[(i0+1)*m + (j0+1), 3] = 1
    V[(j0+1)*p + k0, 3] = 1
    V[j0*p + k0, 3] = -1
    W[k0*n + (i0+1), 3] = 1
    
    # Term 5: (A11+A12)B22
    U[i0*m + j0, 4] = 1
    U[i0*m + (j0+1), 4] = 1
    V[(j0+1)*p + (k0+1), 4] = 1
    W[(k0+1)*n + i0, 4] = 1
    
    # Term 6: (A21-A11)(B11+B12)
    U[(i0+1)*m + j0, 5] = 1
    U[i0*m + j0, 5] = -1
    V[j0*p + k0, 5] = 1
    V[j0*p + (k0+1), 5] = 1
    W[k0*n + i0, 5] = -1
    W[k0*n + (i0+1), 5] = 1
    
    # Term 7: (A12-A22)(B21+B22)
    U[i0*m + (j0+1), 6] = 1
    U[(i0+1)*m + (j0+1), 6] = -1
    V[(j0+1)*p + k0, 6] = 1
    V[(j0+1)*p + (k0+1), 6] = 1
    W[(k0+1)*n + i0, 6] = -1
    W[(k0+1)*n + (i0+1), 6] = 1
    
    return U, V, W

def _combine_decompositions(decomps, n=3, m=3, p=8):
    """
    Combine multiple decompositions by removing overlapping schoolbook terms
    and adding Strassen terms.
    decomps: list of (i0, j0, k0) positions
    """
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    base_rank = n * m * p
    
    # For each Strassen block, remove the 8 schoolbook terms and add 7 Strassen terms
    cols_to_remove = set()
    for i0, j0, k0 in decomps:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk
                    col = i * m * p + j * p + k
                    cols_to_remove.add(col)
    
    keep_cols = [c for c in range(base_rank) if c not in cols_to_remove]
    
    U_new = U[:, keep_cols].copy()
    V_new = V[:, keep_cols].copy()
    W_new = W[:, keep_cols].copy()
    
    # Add Strassen terms
    strassen_cols = []
    for i0, j0, k0 in decomps:
        Us, Vs, Ws = _strassen_2x2x2(n, m, p, i0, j0, k0)
        strassen_cols.append((Us, Vs, Ws))
    
    if strassen_cols:
        U_new = np.column_stack([U_new] + [s[0] for s in strassen_cols])
        V_new = np.column_stack([V_new] + [s[1] for s in strassen_cols])
        W_new = np.column_stack([W_new] + [s[2] for s in strassen_cols])
    
    return U_new, V_new, W_new

def _blocks_overlap(pos1, pos2):
    """Check if two Strassen blocks overlap (share any cell)."""
    i1, j1, k1 = pos1
    i2, j2, k2 = pos2
    # Blocks overlap if their ranges intersect in all three dimensions
    overlap_i = abs(i1 - i2) < 2
    overlap_j = abs(j1 - j2) < 2
    overlap_k = abs(k1 - k2) < 2
    return overlap_i and overlap_j and overlap_k

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = 145.0
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    # Try single Strassen blocks at all valid positions
    # i0 in [0,1], j0 in [0,1], k0 in [0,6]
    positions = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                positions.append((i0, j0, k0))
    
    # Precompute overlap matrix
    num_pos = len(positions)
    overlap = [[False] * num_pos for _ in range(num_pos)]
    for a in range(num_pos):
        for b in range(a + 1, num_pos):
            overlap[a][b] = _blocks_overlap(positions[a], positions[b])
            overlap[b][a] = overlap[a][b]
    
    # Try single Strassen blocks
    for idx in range(num_pos):
        if _time_left(start, deadline) < 5:
            break
        pos = positions[idx]
        U_new, V_new, W_new = _combine_decompositions([pos], n, m, p)
        new_rank = U_new.shape[1]
        
        if new_rank < best_rank and _verify(U_new, V_new, W_new, n, m, p):
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = new_rank
    
    # Try pairs of non-overlapping Strassen blocks
    for idx1 in range(num_pos):
        if _time_left(start, deadline) < 10:
            break
        for idx2 in range(idx1 + 1, num_pos):
            if overlap[idx1][idx2]:
                continue
            U_new, V_new, W_new = _combine_decompositions(
                [positions[idx1], positions[idx2]], n, m, p)
            new_rank = U_new.shape[1]
            
            if new_rank < best_rank and _verify(U_new, V_new, W_new, n, m, p):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = new_rank
    
    # Try triples of non-overlapping Strassen blocks
    for idx1 in range(num_pos):
        if _time_left(start, deadline) < 15:
            break
        for idx2 in range(idx1 + 1, num_pos):
            if overlap[idx1][idx2]:
                continue
            for idx3 in range(idx2 + 1, num_pos):
                if overlap[idx1][idx3] or overlap[idx2][idx3]:
                    continue
                U_new, V_new, W_new = _combine_decompositions(
                    [positions[idx1], positions[idx2], positions[idx3]], n, m, p)
                new_rank = U_new.shape[1]
                
                if new_rank < best_rank and _verify(U_new, V_new, W_new, n, m, p):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = new_rank
    
    # Try quadruples of non-overlapping Strassen blocks
    for idx1 in range(num_pos):
        if _time_left(start, deadline) < 20:
            break
        for idx2 in range(idx1 + 1, num_pos):
            if overlap[idx1][idx2]:
                continue
            for idx3 in range(idx2 + 1, num_pos):
                if overlap[idx1][idx3] or overlap[idx2][idx3]:
                    continue
                for idx4 in range(idx3 + 1, num_pos):
                    if overlap[idx1][idx4] or overlap[idx2][idx4] or overlap[idx3][idx4]:
                        continue
                    U_new, V_new, W_new = _combine_decompositions(
                        [positions[idx1], positions[idx2], positions[idx3], positions[idx4]], n, m, p)
                    new_rank = U_new.shape[1]
                    
                    if new_rank < best_rank and _verify(U_new, V_new, W_new, n, m, p):
                        best_U, best_V, best_W = U_new, V_new, W_new
                        best_rank = new_rank
    
    # Convert to lists for the return interface
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
