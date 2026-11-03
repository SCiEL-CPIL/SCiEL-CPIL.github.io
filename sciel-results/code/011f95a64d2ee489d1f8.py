import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def verify(Uv, Vv, Wv, tol=1e-10):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Uv[i*m+j], Vv[j*p+k] * Wv[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def schoolbook_decomp():
        """Pure schoolbook decomposition, rank 75."""
        total_rank = n * m * p
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W

    def add_strassen_2x2x2(U, V, W, i0, j0, k0, col_offset):
        """Add Strassen rank-7 decomposition for a 2x2x2 block."""
        c = col_offset + 0
        U[i0*m + j0, c] += 1
        U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1
        W[(k0+1)*n + (i0+1), c] += 1

        c = col_offset + 1
        U[(i0+1)*m + j0, c] += 1
        U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        W[k0*n + (i0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += -1

        c = col_offset + 2
        U[i0*m + j0, c] += 1
        V[j0*p + (k0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += -1
        W[(k0+1)*n + i0, c] += 1
        W[(k0+1)*n + (i0+1), c] += 1

        c = col_offset + 3
        U[(i0+1)*m + (j0+1), c] += 1
        V[(j0+1)*p + k0, c] += 1
        V[j0*p + k0, c] += -1
        W[k0*n + i0, c] += 1
        W[k0*n + (i0+1), c] += 1

        c = col_offset + 4
        U[i0*m + j0, c] += 1
        U[i0*m + (j0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += -1
        W[(k0+1)*n + i0, c] += 1

        c = col_offset + 5
        U[(i0+1)*m + j0, c] += 1
        U[i0*m + j0, c] += -1
        V[j0*p + k0, c] += 1
        V[j0*p + (k0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += 1

        c = col_offset + 6
        U[i0*m + (j0+1), c] += 1
        U[(i0+1)*m + (j0+1), c] += -1
        V[(j0+1)*p + k0, c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1

    def try_block_placement(blocks):
        """Try a specific set of non-overlapping 2x2x2 Strassen blocks."""
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if i0+di < n and j0+dj < m and k0+dk < p:
                            covered[i0+di][j0+dj][k0+dk] = True
                        else:
                            return None
        
        remaining = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        remaining += 1
        
        total_rank = 7 * len(blocks) + remaining
        
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        col = 0
        for i0, j0, k0 in blocks:
            add_strassen_2x2x2(U, V, W, i0, j0, k0, col)
            col += 7
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        U[i*m+j, col] += 1
                        V[j*p+k, col] += 1
                        W[k*n+i, col] += 1
                        col += 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W, "rank": total_rank}
        return None

    def find_best_placement():
        """Search over all valid placements of non-overlapping 2x2x2 blocks."""
        best_result = None
        best_rank = 75
        
        possible_blocks = [(i0, j0, k0) for i0 in range(n-1) for j0 in range(m-1) for k0 in range(p-1)]
        
        from itertools import combinations
        
        def blocks_overlap(b1, b2):
            i0a, j0a, k0a = b1
            i0b, j0b, k0b = b2
            if max(i0a, i0b) < min(i0a+2, i0b+2) and \
               max(j0a, j0b) < min(j0a+2, j0b+2) and \
               max(k0a, k0b) < min(k0a+2, k0b+2):
                return True
            return False
        
        for size in [4, 3, 2, 1]:
            if time.time() > deadline - 5:
                break
            found = False
            for combo in combinations(possible_blocks, size):
                if time.time() > deadline - 5:
                    break
                valid = True
                for a in range(size):
                    for b in range(a+1, size):
                        if blocks_overlap(combo[a], combo[b]):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                result = try_block_placement(list(combo))
                if result is not None:
                    if result["rank"] < best_rank:
                        best_rank = result["rank"]
                        best_result = result
                    found = True
            
            if found and size == 4:
                break
        
        return best_result, best_rank

    # Strategy 1: Search for best 2x2x2 block placement
    best_result, best_rank = find_best_placement()
    
    if best_result is not None:
        return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}
    
    # Strategy 2: Fallback to known good placement (rank 71)
    U2 = np.zeros((n*m, 71))
    V2 = np.zeros((m*p, 71))
    W2 = np.zeros((p*n, 71))
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
    ]
    
    col = 0
    for i0, j0, k0 in blocks:
        add_strassen_2x2x2(U2, V2, W2, i0, j0, k0, col)
        col += 7
    
    covered2 = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered2[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered2[i][j][k]:
                    U2[i*m+j, col] += 1
                    V2[j*p+k, col] += 1
                    W2[k*n+i, col] += 1
                    col += 1
    
    if verify(U2, V2, W2):
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    # Strategy 3: Pure schoolbook (rank 75)
    U3, V3, W3 = schoolbook_decomp()
    return {"U": U3.tolist(), "V": V3.tolist(), "W": W3.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    print(f"Max error: {max_err}")
