import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS

    def add_strassen_block(U, V, W, i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    all_positions = [(i0, j0, k0) for i0 in range(n-1) for j0 in range(m-1) for k0 in range(p-1)]
    
    best_rank = n * m * p
    best_result = None

    def verify_and_set_rank(U, V, W, rank):
        """Verify the decomposition is correct and return rank if valid"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        return None
        return rank

    # Try all combinations of up to 5 blocks (allowing overlaps)
    # For each, build U,V,W and verify
    for num_blocks in range(1, 6):
        if time.time() > deadline:
            break
        for combo in itertools.combinations(all_positions, num_blocks):
            if time.time() > deadline:
                break
            rank = num_blocks * 7 + (n*m*p - num_blocks*8)  # optimistic: assume no overlap
            # Actual rank depends on overlap; compute after building
            U = [[0.0]*75 for _ in range(n*m)]  # max possible rank
            V = [[0.0]*75 for _ in range(m*p)]
            W = [[0.0]*75 for _ in range(p*n)]
            
            col = 0
            for i0, j0, k0 in combo:
                add_strassen_block(U, V, W, i0, j0, k0, col)
                col += 7
            
            # Add schoolbook for entries not correctly covered
            # First, compute what's currently covered
            covered = set()
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(col))
                        if abs(val - 1.0) > 1e-9:
                            # Need correction
                            U[i*m+j][col] = 1
                            V[j*p+k][col] = 1
                            W[k*n+i][col] = 1
                            col += 1
                        else:
                            covered.add((i,j,k))
            
            # Trim to actual rank
            U = [row[:col] for row in U]
            V = [row[:col] for row in V]
            W = [row[:col] for row in W]
            
            if col < best_rank:
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(col))
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    best_rank = col
                    best_result = {"U": U, "V": V, "W": W}

    if best_result is None:
        rank = n * m * p
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i*m + j*p + k
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
        best_result = {"U": U, "V": V, "W": W}

    return best_result

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
                max_err = max(max_err, abs(val - 1.0))
    print(f"Max error: {max_err}")
