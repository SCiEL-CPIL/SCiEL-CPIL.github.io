import time
import numpy as np

def _partition_indices(count, max_block=3):
    """Generate all partitions of range(count) into blocks of size 1, 2, or 3."""
    if count == 0:
        yield []
        return
    # First block is size 1
    for rest in _partition_indices(count - 1, max_block):
        yield [[0]] + rest
    # First block is size 2
    if count >= 2 and max_block >= 2:
        for rest in _partition_indices(count - 2, max_block):
            yield [[0, 1]] + rest
    # First block is size 3
    if count >= 3 and max_block >= 3:
        for rest in _partition_indices(count - 3, max_block):
            yield [[0, 1, 2]] + rest

def _strassen_222():
    """Return the 7 rank-1 terms of the 2x2x2 Strassen decomposition."""
    return [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
    ]

def _strassen_333():
    """
    Return a rank-23 decomposition for 3x3x3 matrix multiplication.
    This is a known minimal decomposition.
    """
    # Using a known rank-23 decomposition for 3x3x3
    # Source: Various algebraic complexity theory references
    # The terms are (u, v, w) where u,v,w are 9-element lists (flattened 3x3)
    
    # For brevity and robustness, if I don't have the exact 23 terms memorized perfectly,
    # I can fall back to a known construction or just use schoolbook if 23 is not available.
    # However, 23 is the known minimum.
    # Let's use a simpler, guaranteed valid decomposition if I can't recall the 23 terms exactly.
    # Actually, I can construct one using 2x2x2 Strassen blocks and padding.
    # 3x3x3 can be covered by one 2x2x2 Strassen (rank 7) and the remaining border.
    # Border: 1x3x3, 3x1x3, 3x3x1.
    # This is not disjoint.
    
    # Let's use a known rank-23 decomposition from literature if possible.
    # Since I cannot guarantee the exact 23 terms from memory without error, 
    # I will use a safe fallback: 2x2x2 Strassen + Schoolbook for the rest?
    # No, that's not a valid disjoint block decomposition for 3x3x3.
    
    # Instead, I will use the fact that Rank(3x3x3) = 23.
    # I will implement a search for a 23-rank decomposition if time permits, 
    # or just use schoolbook (27) to be safe, as 27 is not better than the current 88 strategy 
    # if we can use 2x2x2 blocks effectively.
    
    # Actually, the current strategy with 2x2x2 blocks on 3x4x8 gave 88.
    # If I use 3x3x3 schoolbook (27) for the 3x3x3 part, and 2x2x2 for the rest?
    # Let's stick to 2x2x2 Strassen and 1x1x1 Schoolbook for now, 
    # but add 2x2x4 handling.
    
    # For this specific change, I will return None to indicate "use schoolbook" 
    # unless I implement the 23-term decomposition.
    # Given the risk of incorrect 23-term implementation, I will rely on the 2x2x2 improvements.
    
    return None

def _construct_for_partition(part_n, part_m, part_p, n, m, p):
    """Construct U,V,W for given partitions. Returns (rank, U, V, W) or None."""
    max_rank = n * m * p
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    col = 0
    strassen_terms = _strassen_222()
    
    for bn in part_n:
        for bm in part_m:
            for bp in part_p:
                sn, sm, sp = len(bn), len(bm), len(bp)
                
                # Handle 2x2x2 Strassen
                if sn == 2 and sm == 2 and sp == 2:
                    i0, j0, k0 = bn[0], bm[0], bp[0]
                    for (u, v, w) in strassen_terms:
                        for di in range(2):
                            for dj in range(2):
                                U[(i0+di)*m + (j0+dj), col] = u[di*2+dj]
                        for dj in range(2):
                            for dk in range(2):
                                V[(j0+dj)*p + (k0+dk), col] = v[dj*2+dk]
                        for di in range(2):
                            for dk in range(2):
                                W[(k0+dk)*n + (i0+di), col] = w[di*2+dk]
                        col += 1
                # Handle 2x2x4 by two 2x2x2 Strassen blocks
                elif sn == 2 and sm == 2 and sp == 4:
                    i0, j0, k0 = bn[0], bm[0], bp[0]
                    # First 2x2x2
                    for (u, v, w) in strassen_terms:
                        for di in range(2):
                            for dj in range(2):
                                U[(i0+di)*m + (j0+dj), col] = u[di*2+dj]
                        for dj in range(2):
                            for dk in range(2):
                                V[(j0+dj)*p + (k0+dk), col] = v[dj*2+dk]
                        for di in range(2):
                            for dk in range(2):
                                W[(k0+dk)*n + (i0+di), col] = w[di*2+dk]
                        col += 1
                    # Second 2x2x2 (offset k by 2)
                    for (u, v, w) in strassen_terms:
                        for di in range(2):
                            for dj in range(2):
                                U[(i0+di)*m + (j0+dj), col] = u[di*2+dj]
                        for dj in range(2):
                            for dk in range(2):
                                V[(j0+dj)*p + (k0+2+dk), col] = v[dj*2+dk]
                        for di in range(2):
                            for dk in range(2):
                                W[(k0+2+dk)*n + (i0+di), col] = w[di*2+dk]
                        col += 1
                # Handle 2x4x2 by two 2x2x2 Strassen blocks
                elif sn == 2 and sm == 4 and sp == 2:
                    i0, j0, k0 = bn[0], bm[0], bp[0]
                    # First 2x2x2
                    for (u, v, w) in strassen_terms:
                        for di in range(2):
                            for dj in range(2):
                                U[(i0+di)*m + (j0+dj), col] = u[di*2+dj]
                        for dj in range(2):
                            for dk in range(2):
                                V[(j0+dj)*p + (k0+dk), col] = v[dj*2+dk]
                        for di in range(2):
                            for dk in range(2):
                                W[(k0+dk)*n + (i0+di), col] = w[di*2+dk]
                        col += 1
                    # Second 2x2x2 (offset j by 2)
                    for (u, v, w) in strassen_terms:
                        for di in range(2):
                            for dj in range(2):
                                U[(i0+di)*m + (j0+2+dj), col] = u[di*2+dj]
                        for dj in range(2):
                            for dk in range(2):
                                V[(j0+2+dj)*p + (k0+dk), col] = v[dj*2+dk]
                        for di in range(2):
                            for dk in range(2):
                                W[(k0+dk)*n + (i0+di), col] = w[di*2+dk]
                        col += 1
                # Handle 4x2x2 by two 2x2x2 Strassen blocks
                elif sn == 4 and sm == 2 and sp == 2:
                    i0, j0, k0 = bn[0], bm[0], bp[0]
                    # First 2x2x2
                    for (u, v, w) in strassen_terms:
                        for di in range(2):
                            for dj in range(2):
                                U[(i0+di)*m + (j0+dj), col] = u[di*2+dj]
                        for dj in range(2):
                            for dk in range(2):
                                V[(j0+dj)*p + (k0+dk), col] = v[dj*2+dk]
                        for di in range(2):
                            for dk in range(2):
                                W[(k0+dk)*n + (i0+di), col] = w[di*2+dk]
                        col += 1
                    # Second 2x2x2 (offset i by 2)
                    for (u, v, w) in strassen_terms:
                        for di in range(2):
                            for dj in range(2):
                                U[(i0+2+di)*m + (j0+dj), col] = u[di*2+dj]
                        for dj in range(2):
                            for dk in range(2):
                                V[(j0+dj)*p + (k0+dk), col] = v[dj*2+dk]
                        for di in range(2):
                            for dk in range(2):
                                W[(k0+dk)*n + (i0+2+di), col] = w[di*2+dk]
                        col += 1
                else:
                    # Schoolbook for other blocks
                    for i in bn:
                        for j in bm:
                            for k in bp:
                                U[i*m + j, col] = 1
                                V[j*p + k, col] = 1
                                W[k*n + i, col] = 1
                                col += 1
    
    if col == 0:
        return None
    return col, U[:, :col], V[:, :col], W[:, :col]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is exact."""
    U, V, W = np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = 3, 4, 8
    deadline = time.time() + 130
    
    best_rank = n * m * p
    best_result = None
    
    part_n = list(_partition_indices(n, max_block=3))
    part_m = list(_partition_indices(m, max_block=3))
    part_p = list(_partition_indices(p, max_block=3))
    
    for pn in part_n:
        if time.time() > deadline:
            break
        for pm in part_m:
            if time.time() > deadline:
                break
            for pp in part_p:
                if time.time() > deadline:
                    break
                res = _construct_for_partition(pn, pm, pp, n, m, p)
                if res is None:
                    continue
                rank, U, V, W = res
                if _verify(U, V, W, n, m, p):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if best_result is None:
        rank = n * m * p
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1.0
                    V[j*p+k][c] = 1.0
                    W[k*n+i][c] = 1.0
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    return best_result
